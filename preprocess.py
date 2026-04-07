import argparse
import json
import logging
import multiprocessing
import os
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


def find_anat_files(bids_dir: Path, subject: str | None = None) -> list[Path]:
    files = []
    for suffix in SUPPORTED_SUFFIXES:
        files += list(bids_dir.glob(f"*/anat/*_{suffix}.nii.gz"))
        files += list(bids_dir.glob(f"*/*/anat/*_{suffix}.nii.gz"))
    if subject:
        files = [f for f in files if f.parts[len(bids_dir.parts)] == subject]
    return sorted(files)


def output_paths(input_path: Path, bids_dir: Path, out_dir: Path) -> tuple[Path, Path]:
    rel = input_path.relative_to(bids_dir)
    suffix = next(s for s in SUPPORTED_SUFFIXES if input_path.name.endswith(f"_{s}.nii.gz"))
    stem = input_path.name.replace(f"_{suffix}.nii.gz", "")
    anat_dir = out_dir / rel.parent
    anat_dir.mkdir(parents=True, exist_ok=True)
    preproc = anat_dir / f"{stem}_desc-preproc_{suffix}.nii.gz"
    mask    = anat_dir / f"{stem}_desc-brain_mask.nii.gz"
    return preproc, mask


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


def crop_to_brain(img: nib.Nifti1Image, mask: nib.Nifti1Image) -> tuple[nib.Nifti1Image, nib.Nifti1Image]:
    mask_data = mask.get_fdata()
    coords = np.argwhere(mask_data > 0.5)
    if coords.size == 0:
        raise ValueError("empty brain mask — SynthStrip may have failed")
    lo, hi = coords.min(0), coords.max(0) + 1

    def crop_img(src: nib.Nifti1Image) -> nib.Nifti1Image:
        data = src.get_fdata()
        affine = src.affine.copy()
        cropped = data[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        affine[:3, 3] = affine[:3, :3] @ lo + affine[:3, 3]
        return nib.Nifti1Image(cropped, affine, src.header)

    return crop_img(img), crop_img(mask)


def process_file(args: tuple[Path, Path, Path]) -> dict:
    input_path, bids_dir, out_dir = args
    name = input_path.name
    preproc_path, mask_path = output_paths(input_path, bids_dir, out_dir)

    if preproc_path.exists() and mask_path.exists():
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

        brain_data = brain.get_fdata() * mask.get_fdata()
        brain_data = np.clip(brain_data, 0, None)
        brain = nib.Nifti1Image(brain_data, brain.affine, brain.header)

        log.info("%s — cropping to brain bounding box", name)
        brain, mask = crop_to_brain(brain, mask)

        nib.save(brain, preproc_path)
        nib.save(mask,  mask_path)
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
                "Tight crop to brain bounding box",
            ]
        },
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
    args = parser.parse_args()

    bids_dir = args.bids.resolve()
    out_dir  = args.output.resolve()

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
    tasks = [(f, bids_dir, out_dir) for f in files]

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
