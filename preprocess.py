import argparse
import json
import logging
import multiprocessing
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import ants
import nibabel as nib
import numpy as np

log = logging.getLogger("preproc")

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"


def setup_logging(log_file: Path) -> None:
    logger = logging.getLogger("preproc")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_file)
    fh.setFormatter(logging.Formatter(LOG_FORMAT))
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(fh)
    logger.addHandler(sh)


def _pool_init(log_file: Path, itk_threads: int) -> None:
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(itk_threads)
    setup_logging(log_file)


SUPPORTED_SUFFIXES = ("T1w", "T2w", "FLAIR")
TEMPLATE_SPACE = "MNI152NLin2009cAsym"
DEFAULT_TEMPLATE_DIR = Path("/opt/templateflow") / f"tpl-{TEMPLATE_SPACE}"
DEFAULT_TEMPLATE_BRAIN = (
    DEFAULT_TEMPLATE_DIR / f"tpl-{TEMPLATE_SPACE}_res-01_desc-brain_T1w.nii.gz"
)
DEFAULT_TEMPLATE_T1W = DEFAULT_TEMPLATE_DIR / f"tpl-{TEMPLATE_SPACE}_res-01_T1w.nii.gz"


def is_supported_anat_file(path: Path, bids_dir: Path) -> bool:
    rel = path.relative_to(bids_dir)
    if path.parent.name != "anat":
        return False
    if not any(part.startswith("sub-") for part in rel.parts):
        return False
    return any(path.name.endswith(f"_{suffix}.nii.gz") for suffix in SUPPORTED_SUFFIXES)


def find_anat_files(bids_dir: Path, subject: str | None = None) -> list[Path]:
    files = []
    for root, dirs, names in os.walk(bids_dir, followlinks=True):
        root_path = Path(root)
        for name in names:
            if not name.endswith(".nii.gz"):
                continue
            path = root_path / name
            if is_supported_anat_file(path, bids_dir):
                files.append(path)
    if subject:
        files = [path for path in files if subject in path.relative_to(bids_dir).parts]
    return sorted(files)


def output_paths(input_path: Path, bids_dir: Path, out_dir: Path) -> tuple[Path, Path, Path]:
    rel = input_path.relative_to(bids_dir)
    suffix = next(s for s in SUPPORTED_SUFFIXES if input_path.name.endswith(f"_{s}.nii.gz"))
    stem = input_path.name.replace(f"_{suffix}.nii.gz", "")
    anat_dir = out_dir / rel.parent
    anat_dir.mkdir(parents=True, exist_ok=True)
    preproc = anat_dir / f"{stem}_space-{TEMPLATE_SPACE}_desc-preproc_{suffix}.nii.gz"
    mask = anat_dir / f"{stem}_space-{TEMPLATE_SPACE}_desc-brain_mask_{suffix}.nii.gz"
    xfm = anat_dir / f"{stem}_from-native_to-{TEMPLATE_SPACE}_mode-image_desc-{suffix}_xfm.mat"
    return preproc, mask, xfm


def reorient_to_ras(img: nib.Nifti1Image) -> nib.Nifti1Image:
    return nib.as_closest_canonical(img)


def nib_to_ants(img: nib.Nifti1Image) -> ants.ANTsImage:
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
        nib.save(img, tmp.name)
        ants_img = ants.image_read(tmp.name)
    Path(tmp.name).unlink(missing_ok=True)
    return ants_img


def ants_to_nib(ants_img: ants.ANTsImage) -> nib.Nifti1Image:
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
        ants.image_write(ants_img, tmp.name)
        img = nib.load(tmp.name)
        img = nib.Nifti1Image(img.get_fdata(), img.affine, img.header)
    Path(tmp.name).unlink(missing_ok=True)
    return img


def n4_correction(img: nib.Nifti1Image) -> nib.Nifti1Image:
    corrected = ants.n4_bias_field_correction(nib_to_ants(img))
    return ants_to_nib(corrected)


def synthstrip(img: nib.Nifti1Image) -> tuple[nib.Nifti1Image, nib.Nifti1Image]:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        nib.save(img, tmp / "input.nii.gz")
        subprocess.run(
            [
                "mri_synthstrip",
                "-i", str(tmp / "input.nii.gz"),
                "-o", str(tmp / "brain.nii.gz"),
                "-m", str(tmp / "mask.nii.gz"),
            ],
            check=True,
        )
        brain = nib.load(tmp / "brain.nii.gz")
        mask  = nib.load(tmp / "mask.nii.gz")
        brain = nib.Nifti1Image(brain.get_fdata(), brain.affine, brain.header)
        mask  = nib.Nifti1Image(mask.get_fdata(),  mask.affine,  mask.header)
    return brain, mask


def resample_1mm(img: nib.Nifti1Image) -> nib.Nifti1Image:
    resampled = ants.resample_image(nib_to_ants(img), (1, 1, 1), use_voxels=False, interp_type=4)
    return ants_to_nib(resampled)


def resample_mask_1mm(mask: nib.Nifti1Image) -> nib.Nifti1Image:
    resampled = ants.resample_image(nib_to_ants(mask), (1, 1, 1), use_voxels=False, interp_type=1)
    return ants_to_nib(resampled)


def rigid_register_to_template(
    img: nib.Nifti1Image,
    mask: nib.Nifti1Image,
    template_brain_path: Path,
    transform_path: Path,
) -> tuple[nib.Nifti1Image, nib.Nifti1Image]:
    fixed = ants.image_read(str(template_brain_path))
    moving = nib_to_ants(img)
    moving_mask = nib_to_ants(mask)

    with tempfile.TemporaryDirectory() as tmpdir:
        outprefix = str(Path(tmpdir) / "rigid_")
        tx = ants.registration(
            fixed=fixed,
            moving=moving,
            type_of_transform="Rigid",
            outprefix=outprefix,
        )
        fwdtransforms = tx["fwdtransforms"]
        rigid_transform = next((Path(path) for path in fwdtransforms if str(path).endswith(".mat")), None)
        if rigid_transform is None:
            raise RuntimeError("ANTs rigid registration did not produce a matrix transform")

        registered = ants.apply_transforms(
            fixed=fixed,
            moving=moving,
            transformlist=fwdtransforms,
            interpolator="bSpline",
        )
        registered_mask = ants.apply_transforms(
            fixed=fixed,
            moving=moving_mask,
            transformlist=fwdtransforms,
            interpolator="nearestNeighbor",
        )
        shutil.copy2(rigid_transform, transform_path)

    return ants_to_nib(registered), ants_to_nib(registered_mask)


def apply_mask_and_clip(img: nib.Nifti1Image, mask: nib.Nifti1Image) -> tuple[nib.Nifti1Image, nib.Nifti1Image]:
    mask_data = (mask.get_fdata() > 0.5).astype(np.uint8)
    brain_data = np.clip(img.get_fdata() * mask_data, 0, None).astype(np.float32)

    brain_header = img.header.copy()
    brain_header.set_data_dtype(np.float32)
    mask_header = mask.header.copy()
    mask_header.set_data_dtype(np.uint8)

    brain = nib.Nifti1Image(brain_data, img.affine, brain_header)
    bin_mask = nib.Nifti1Image(mask_data, mask.affine, mask_header)
    return brain, bin_mask


def process_file(args: tuple[Path, Path, Path, Path]) -> dict:
    input_path, bids_dir, out_dir, template_brain_path = args
    name = str(input_path.relative_to(bids_dir))
    preproc_path, mask_path, xfm_path = output_paths(input_path, bids_dir, out_dir)

    if preproc_path.exists() and mask_path.exists() and xfm_path.exists():
        log.info("%s — already processed, skipping", name)
        return {"file": name, "status": "skipped"}

    log.info("%s — starting", name)
    try:
        img = nib.load(input_path)

        log.info("%s — reorienting to RAS", name)
        img = reorient_to_ras(img)

        log.info("%s — N4 bias field correction", name)
        img = n4_correction(img)

        log.info("%s — SynthStrip skull stripping", name)
        brain, mask = synthstrip(img)

        zooms = brain.header.get_zooms()[:3]
        if not np.allclose(zooms, 1.0, atol=1e-3):
            log.info("%s — resampling to 1 mm isotropic", name)
            brain = resample_1mm(brain)
            mask  = resample_mask_1mm(mask)
        else:
            log.info("%s — already 1 mm isotropic, skipping resample", name)

        brain, mask = apply_mask_and_clip(brain, mask)

        log.info("%s — rigid registration to %s", name, TEMPLATE_SPACE)
        brain, mask = rigid_register_to_template(brain, mask, template_brain_path, xfm_path)

        log.info("%s — applying transformed mask and clipping overshoot", name)
        brain, mask = apply_mask_and_clip(brain, mask)

        nib.save(brain, preproc_path)
        nib.save(mask, mask_path)
        log.info("%s — done → %s", name, preproc_path.name)
        return {"file": name, "status": "success"}
    except Exception as e:
        log.error("%s — failed: %s", name, e, exc_info=True)
        return {"file": name, "status": "failed", "error": str(e)}


def write_dataset_description(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    desc = {
        "Name": "anat-preproc",
        "BIDSVersion": "1.8.0",
        "DatasetType": "derivative",
        "GeneratedBy": [{"Name": "anat-preproc", "Version": "0.1.0"}],
        "PipelineDescription": {
            "Steps": [
                "Reorient to RAS",
                "N4 bias field correction (ANTs)",
                "Skull stripping (SynthStrip)",
                "Resample to 1mm isotropic",
                f"Rigid registration to {TEMPLATE_SPACE} template brain",
                "Apply brain mask and clip negative interpolation overshoot after resampling operations",
            ]
        },
        "TemplateSpace": TEMPLATE_SPACE,
        "TemplateBrain": DEFAULT_TEMPLATE_BRAIN.name,
        "TemplateT1w": DEFAULT_TEMPLATE_T1W.name,
    }
    with open(out_dir / "dataset_description.json", "w") as f:
        json.dump(desc, f, indent=2)


def main() -> None:
    default_bids = Path(__file__).parent / "data" / "raw"
    default_out  = Path(__file__).parent / "data" / "processed"
    default_logs = Path(__file__).parent / "data" / "logs"

    parser = argparse.ArgumentParser(description="Anat preprocessing pipeline")
    parser.add_argument("--bids",        default=default_bids, type=Path)
    parser.add_argument("--subject",     default=None,         type=str)
    parser.add_argument("--output",      default=default_out,  type=Path)
    parser.add_argument("--log_dir",     default=default_logs, type=Path)
    parser.add_argument("--n_workers",   default=48,           type=int)
    parser.add_argument("--itk_threads", default=2,            type=int)
    parser.add_argument("--template_brain", default=DEFAULT_TEMPLATE_BRAIN, type=Path)
    args = parser.parse_args()

    bids_dir = args.bids.resolve()
    out_dir = args.output.resolve()
    template_brain = args.template_brain.resolve()

    if not template_brain.exists():
        raise FileNotFoundError(f"Template brain not found: {template_brain}")

    write_dataset_description(out_dir)

    log_dir = args.log_dir.resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "preprocessing.log"
    setup_logging(log_file)

    files = find_anat_files(bids_dir, args.subject)
    if not files:
        log.info("No T1w/T2w/FLAIR files found.")
        return

    log.info("Found %d anat file(s). Workers: %d", len(files), args.n_workers)
    tasks = [(f, bids_dir, out_dir, template_brain) for f in files]

    if args.n_workers > 1:
        os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(args.itk_threads)
        with multiprocessing.Pool(args.n_workers, initializer=_pool_init, initargs=(log_file, args.itk_threads)) as pool:
            results = pool.map(process_file, tasks)
    else:
        results = [process_file(task) for task in tasks]

    # --- write status summary ---
    succeeded = [r for r in results if r["status"] == "success"]
    failed    = [r for r in results if r["status"] == "failed"]
    skipped   = [r for r in results if r["status"] == "skipped"]

    status = {
        "timestamp": datetime.now().isoformat(),
        "total": len(results),
        "successful": len(succeeded),
        "failed": len(failed),
        "skipped": len(skipped),
        "subjects": results,
    }
    status_file = log_dir / "processing_status.json"
    with open(status_file, "w") as f:
        json.dump(status, f, indent=2)

    log.info(
        "Finished — %d succeeded, %d failed, %d skipped. Status: %s",
        len(succeeded), len(failed), len(skipped), status_file,
    )

    if failed:
        log.error("Failed subjects: %s", [r["file"] for r in failed])
        sys.exit(1)


if __name__ == "__main__":
    main()
