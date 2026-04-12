import argparse
import csv
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

# ── SynthSeg ──────────────────────────────────────────────────────────────────

DEFAULT_SYNTHSEG_OUTPUT = Path(__file__).parent / "data" / "derivatives" / "synthseg"

# Static FreeSurfer label → name mapping for the _dseg.nii.gz sidecar TSV.
# These are the integer voxel values written into the segmentation labelmap.
# NOTE: label 85 (Optic-Chiasm) is NOT produced by SynthSeg — omitted.
SYNTHSEG_LABEL_TABLE: dict[int, str] = {
    2:   "Left-Cerebral-White-Matter",
    3:   "Left-Cerebral-Cortex",
    4:   "Left-Lateral-Ventricle",
    5:   "Left-Inf-Lateral-Vent",
    7:   "Left-Cerebellum-White-Matter",
    8:   "Left-Cerebellum-Cortex",
    10:  "Left-Thalamus",
    11:  "Left-Caudate",
    12:  "Left-Putamen",
    13:  "Left-Pallidum",
    14:  "3rd-Ventricle",
    15:  "4th-Ventricle",
    16:  "Brain-Stem",
    17:  "Left-Hippocampus",
    18:  "Left-Amygdala",
    24:  "CSF",
    26:  "Left-Accumbens-area",
    28:  "Left-VentralDC",
    41:  "Right-Cerebral-White-Matter",
    42:  "Right-Cerebral-Cortex",
    43:  "Right-Lateral-Ventricle",
    44:  "Right-Inf-Lateral-Vent",
    46:  "Right-Cerebellum-White-Matter",
    47:  "Right-Cerebellum-Cortex",
    49:  "Right-Thalamus",
    50:  "Right-Caudate",
    51:  "Right-Putamen",
    52:  "Right-Pallidum",
    53:  "Right-Hippocampus",
    54:  "Right-Amygdala",
    58:  "Right-Accumbens-area",
    60:  "Right-VentralDC",
    77:  "WM-hypointensities",
    251: "CC_Posterior",
    252: "CC_Mid_Posterior",
    253: "CC_Central",
    254: "CC_Mid_Anterior",
    255: "CC_Anterior",
}

# Column name groups for computing summary metrics from the --vol CSV.
# Names are exactly as output by mri_synthseg (empirically verified on FS 7.4.1).
# VentralDC deliberately excluded from sGMV per Bethlehem 2022.
_GMV_COLS = ("left cerebral cortex", "right cerebral cortex")
_WMV_COLS = ("left cerebral white matter", "right cerebral white matter")
_SGMV_COLS = (
    "left thalamus",     "right thalamus",
    "left caudate",      "right caudate",
    "left putamen",      "right putamen",
    "left pallidum",     "right pallidum",
    "left hippocampus",  "right hippocampus",
    "left amygdala",     "right amygdala",
    "left accumbens area", "right accumbens area",
)
_VENTCSF_COLS = (
    "left lateral ventricle",           "right lateral ventricle",
    "left inferior lateral ventricle",  "right inferior lateral ventricle",
    "3rd ventricle",                    "4th ventricle",
)

# DKT parcellation region names as produced by mri_synthseg --parc (FS 7.4.1,
# empirically verified). Each appears as "ctx-lh-{name}" and "ctx-rh-{name}"
# in the volumes CSV. 34 regions per hemisphere.
DKT_REGION_NAMES: list[str] = [
    "bankssts", "caudalanteriorcingulate", "caudalmiddlefrontal", "cuneus",
    "entorhinal", "fusiform", "inferiorparietal", "inferiortemporal",
    "isthmuscingulate", "lateraloccipital", "lateralorbitofrontal", "lingual",
    "medialorbitofrontal", "middletemporal", "parahippocampal", "paracentral",
    "parsopercularis", "parsorbitalis", "parstriangularis", "pericalcarine",
    "postcentral", "posteriorcingulate", "precentral", "precuneus",
    "rostralanteriorcingulate", "rostralmiddlefrontal", "superiorfrontal",
    "superiorparietal", "superiortemporal", "supramarginal", "frontalpole",
    "temporalpole", "transversetemporal", "insula",
]


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


# ── SynthSeg helpers ──────────────────────────────────────────────────────────

def _detect_gpu_count() -> int:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        return len(result.stdout.strip().splitlines())
    except (subprocess.SubprocessError, FileNotFoundError):
        return 0


def write_tsv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _is_valid_tsv(path: Path) -> bool:
    """Returns True only if path exists and contains at least one data row."""
    if not path.exists():
        return False
    try:
        lines = path.read_text().splitlines()
        return len(lines) >= 2
    except OSError:
        return False


def synthseg_output_paths(
    input_path: Path, bids_dir: Path, synthseg_dir: Path
) -> tuple[Path, Path, Path, Path]:
    rel = input_path.relative_to(bids_dir)
    suffix = next(s for s in SUPPORTED_SUFFIXES if input_path.name.endswith(f"_{s}.nii.gz"))
    stem = input_path.name.replace(f"_{suffix}.nii.gz", "")
    anat_dir = synthseg_dir / rel.parent
    anat_dir.mkdir(parents=True, exist_ok=True)
    # res-1mm reflects SynthSeg's documented behaviour: output is always 1mm
    # isotropic regardless of input resolution.
    seg     = anat_dir / f"{stem}_res-1mm_desc-synthseg_dseg.nii.gz"
    dseg    = anat_dir / f"{stem}_desc-synthseg_dseg.tsv"
    volumes = anat_dir / f"{stem}_desc-synthseg_volumes.tsv"
    qc      = anat_dir / f"{stem}_desc-synthseg_qc.tsv"
    return seg, dseg, volumes, qc


def run_synthseg(
    input_path: Path,
    seg_path: Path,
    vol_csv: Path,
    qc_csv: Path,
    threads: int,
    cpu_only: bool,
) -> None:
    """
    Calls mri_synthseg. GPU is used automatically when CUDA_VISIBLE_DEVICES is
    set (done by _synthseg_pool_init per worker). --cpu forces CPU-only mode.
    Raises subprocess.TimeoutExpired if the process exceeds 600s.
    """
    cmd = [
        "mri_synthseg",
        "--i",       str(input_path),
        "--o",       str(seg_path),
        "--parc",
        "--robust",
        "--vol",     str(vol_csv),
        "--qc",      str(qc_csv),
        "--threads", str(threads),
    ]
    if cpu_only:
        cmd.append("--cpu")
    subprocess.run(cmd, check=True, timeout=600)


def parse_synthseg_volumes(vol_csv: Path) -> list[dict]:
    """
    Parses the mri_synthseg --vol CSV (string column names, one data row per
    input file) and returns rows for the _volumes.tsv derivative.

    Output sections (in order):
      1. Raw structural volumes (all columns except subject, total intracranial,
         and DKT parcellation columns)
      2. Bilateral DKT sums (ctx-lh-{name} + ctx-rh-{name} per region)
      3. Computed summary metrics: GMV, WMV, sGMV, VentCSF, TCV
         - sGMV excludes VentralDC per Bethlehem 2022 FreeSurfer convention
         - TCV = GMV + WMV + sGMV (cerebrum only, no cerebellum)
    """
    with open(vol_csv, newline="") as fh:
        reader = csv.DictReader(fh)
        row = next(iter(reader))

    vols: dict[str, float] = {}
    for k, v in row.items():
        if k and k.strip() not in ("subject", ""):
            try:
                vols[k.strip()] = float(v)
            except (ValueError, TypeError):
                pass

    dkt_cols = {f"ctx-lh-{r}" for r in DKT_REGION_NAMES} | {f"ctx-rh-{r}" for r in DKT_REGION_NAMES}

    output_rows: list[dict] = []

    # 1. Raw structural volumes (ICV first as normalization reference, then rest)
    if "total intracranial" in vols:
        output_rows.append({"region": "total intracranial", "volume_mm3": round(vols["total intracranial"], 4)})
    for col, vol in vols.items():
        if col == "total intracranial" or col in dkt_cols:
            continue
        output_rows.append({"region": col, "volume_mm3": round(vol, 4)})

    # 2. Bilateral DKT sums
    for region in DKT_REGION_NAMES:
        lv = vols.get(f"ctx-lh-{region}", 0.0)
        rv = vols.get(f"ctx-rh-{region}", 0.0)
        output_rows.append({"region": f"ctx-{region}", "volume_mm3": round(lv + rv, 4)})

    # 3. Computed summary metrics
    gmv     = sum(vols.get(c, 0.0) for c in _GMV_COLS)
    wmv     = sum(vols.get(c, 0.0) for c in _WMV_COLS)
    sgmv    = sum(vols.get(c, 0.0) for c in _SGMV_COLS)
    ventcsf = sum(vols.get(c, 0.0) for c in _VENTCSF_COLS)
    tcv     = gmv + wmv + sgmv

    for metric, value in [
        ("GMV",     gmv),
        ("WMV",     wmv),
        ("sGMV",    sgmv),
        ("VentCSF", ventcsf),
        ("TCV",     tcv),
    ]:
        output_rows.append({"region": metric, "volume_mm3": round(value, 4)})

    return output_rows


def parse_synthseg_qc(qc_csv: Path) -> list[dict]:
    with open(qc_csv, newline="") as fh:
        reader = csv.DictReader(fh)
        row = next(iter(reader))
    return [
        {"structure": k.strip(), "qc_score": v}
        for k, v in row.items()
        if k and k.strip() != "subject"
    ]


def write_synthseg_dseg_tsv(path: Path) -> None:
    """Writes the static FreeSurfer label→name lookup as a BIDS _dseg.tsv sidecar."""
    rows = [{"index": label, "name": name} for label, name in SYNTHSEG_LABEL_TABLE.items()]
    write_tsv(rows, path)


def write_synthseg_dataset_description(synthseg_dir: Path, source_bids_dir: Path) -> None:
    synthseg_dir.mkdir(parents=True, exist_ok=True)
    try:
        source_rel = str(source_bids_dir.relative_to(synthseg_dir.parent.parent))
    except ValueError:
        source_rel = str(source_bids_dir)
    desc = {
        "Name": "synthseg",
        "BIDSVersion": "1.8.0",
        "DatasetType": "derivative",
        "GeneratedBy": [{"Name": "mri_synthseg", "Version": "7.4.1"}],
        "SourceDatasets": [{"URL": f"../{source_rel}"}],
        "PipelineDescription": {
            "Steps": [
                "Reorient to RAS",
                "Whole-brain segmentation and parcellation (mri_synthseg --parc --robust)",
            ],
            "Notes": (
                "Output segmentation NIfTIs (_res-1mm_desc-synthseg_dseg.nii.gz) are "
                "always 1mm isotropic regardless of input resolution. This is documented "
                "SynthSeg behaviour and does not affect volume accuracy (volumes are "
                "computed internally before resampling)."
            ),
        },
    }
    with open(synthseg_dir / "dataset_description.json", "w") as f:
        json.dump(desc, f, indent=2)


def _synthseg_pool_init(log_file: Path, n_gpus: int) -> None:
    # Pin each worker to a distinct GPU so workers don't compete for GPU 0.
    # CUDA_VISIBLE_DEVICES must be set before the mri_synthseg subprocess starts
    # so TensorFlow/PyTorch in the child process initialises on the right device.
    worker_id = multiprocessing.current_process()._identity[0] - 1
    if n_gpus > 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(worker_id % n_gpus)
    setup_logging(log_file)


def process_synthseg_file(task: tuple) -> dict:
    input_path, bids_dir, synthseg_dir, synthseg_threads, cpu_only = task
    name = str(input_path.relative_to(bids_dir))
    seg_path, dseg_tsv, volumes_tsv, qc_tsv = synthseg_output_paths(
        input_path, bids_dir, synthseg_dir
    )

    if (seg_path.exists() and dseg_tsv.exists()
            and _is_valid_tsv(volumes_tsv) and _is_valid_tsv(qc_tsv)):
        log.info("%s — SynthSeg already done, skipping", name)
        return {"file": name, "status": "skipped"}

    log.info("%s — SynthSeg starting", name)
    try:
        img = nib.load(input_path)

        log.info("%s — reorienting to RAS for SynthSeg", name)
        img = reorient_to_ras(img)

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            reoriented_path = tmp / "reoriented.nii.gz"
            vol_csv         = tmp / "volumes.csv"
            qc_csv_path     = tmp / "qc.csv"
            nib.save(img, reoriented_path)

            log.info("%s — running mri_synthseg", name)
            run_synthseg(reoriented_path, seg_path, vol_csv, qc_csv_path, synthseg_threads, cpu_only)

            log.info("%s — parsing volumes and writing outputs", name)
            vol_rows = parse_synthseg_volumes(vol_csv)
            qc_rows  = parse_synthseg_qc(qc_csv_path)

        write_tsv(vol_rows, volumes_tsv)
        write_tsv(qc_rows,  qc_tsv)
        write_synthseg_dseg_tsv(dseg_tsv)

        log.info("%s — SynthSeg done → %s", name, seg_path.name)
        return {"file": name, "status": "success"}

    except subprocess.TimeoutExpired:
        log.error("%s — SynthSeg timed out after 600s", name)
        return {"file": name, "status": "failed", "error": "timeout"}
    except Exception as e:
        log.error("%s — SynthSeg failed: %s", name, e, exc_info=True)
        return {"file": name, "status": "failed", "error": str(e)}


# ── Orchestration ─────────────────────────────────────────────────────────────

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
    parser.add_argument("--synthseg",         action="store_true", default=False,
                        help="Run optional SynthSeg segmentation branch.")
    parser.add_argument("--cpu",              action="store_true", default=False,
                        help="Force SynthSeg to run on CPU (passes --cpu to mri_synthseg).")
    parser.add_argument("--synthseg_threads", default=8,           type=int,
                        help="CPU threads per mri_synthseg call (default: 8).")
    parser.add_argument("--synthseg_output",  default=DEFAULT_SYNTHSEG_OUTPUT, type=Path,
                        help="Output directory for SynthSeg derivatives.")
    parser.add_argument("--synthseg_workers", default=None,        type=int,
                        help="Worker pool size for SynthSeg. Defaults to GPU count (or n_workers on CPU).")
    args = parser.parse_args()

    bids_dir = args.bids.resolve()
    out_dir = args.output.resolve()
    template_brain = args.template_brain.resolve()
    synthseg_dir = args.synthseg_output.resolve()

    if not template_brain.exists():
        raise FileNotFoundError(f"Template brain not found: {template_brain}")

    # Determine SynthSeg worker count: one worker per GPU on a multi-GPU node
    # so each worker is pinned to its own GPU (see _synthseg_pool_init).
    # CPU mode uses n_workers. Explicit --synthseg_workers overrides either.
    n_gpus = _detect_gpu_count()
    synthseg_workers = n_gpus if (n_gpus > 0 and not args.cpu) else args.n_workers
    if args.synthseg_workers is not None:
        synthseg_workers = args.synthseg_workers

    write_dataset_description(out_dir)
    if args.synthseg:
        write_synthseg_dataset_description(synthseg_dir, bids_dir)

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
    if args.synthseg:
        synthseg_tasks = [
            (f, bids_dir, synthseg_dir, args.synthseg_threads, args.cpu)
            for f in files
        ]

    if args.n_workers > 1:
        os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(args.itk_threads)
        with multiprocessing.Pool(args.n_workers, initializer=_pool_init, initargs=(log_file, args.itk_threads)) as pool:
            results = pool.map(process_file, tasks)
    else:
        results = [process_file(task) for task in tasks]

    if args.synthseg:
        log.info(
            "SynthSeg: %d file(s), %d worker(s), %d GPU(s), cpu_only=%s",
            len(synthseg_tasks), synthseg_workers, n_gpus, args.cpu,
        )
        if synthseg_workers > 1:
            with multiprocessing.Pool(
                synthseg_workers,
                initializer=_synthseg_pool_init,
                initargs=(log_file, n_gpus if not args.cpu else 0),
            ) as pool:
                synthseg_results = pool.map(process_synthseg_file, synthseg_tasks)
        else:
            synthseg_results = [process_synthseg_file(t) for t in synthseg_tasks]

    # --- write preproc status summary ---
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

    # --- write SynthSeg status summary ---
    if args.synthseg:
        ss_succeeded = [r for r in synthseg_results if r["status"] == "success"]
        ss_failed    = [r for r in synthseg_results if r["status"] == "failed"]
        ss_skipped   = [r for r in synthseg_results if r["status"] == "skipped"]
        synthseg_status = {
            "timestamp": datetime.now().isoformat(),
            "total": len(synthseg_results),
            "successful": len(ss_succeeded),
            "failed": len(ss_failed),
            "skipped": len(ss_skipped),
            "subjects": synthseg_results,
        }
        synthseg_status_file = log_dir / "synthseg_status.json"
        with open(synthseg_status_file, "w") as f:
            json.dump(synthseg_status, f, indent=2)
        log.info(
            "SynthSeg — %d succeeded, %d failed, %d skipped. Status: %s",
            len(ss_succeeded), len(ss_failed), len(ss_skipped), synthseg_status_file,
        )
        if ss_failed:
            log.error("SynthSeg failed subjects: %s", [r["file"] for r in ss_failed])
            sys.exit(1)


if __name__ == "__main__":
    main()
