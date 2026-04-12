# Anat Preprocessing

This container preprocesses BIDS anatomical images and writes MNI-space derivatives.

## Pipeline

The current sequence is:

1. Reorient to RAS
2. N4 bias field correction with ANTs
3. Skull stripping with SynthStrip
4. Resample to 1 mm isotropic when needed
5. Rigid registration to TemplateFlow `MNI152NLin2009cAsym`
6. Apply the transformed mask and clip negative interpolation overshoot

Registration uses the baked-in TemplateFlow brain image:

- `tpl-MNI152NLin2009cAsym_res-01_desc-brain_T1w.nii.gz`

The Docker build also bakes in the matching non-brain T1w template:

- `tpl-MNI152NLin2009cAsym_res-01_T1w.nii.gz`

## Outputs

For each supported input (`T1w`, `T2w`, `FLAIR`), the pipeline writes:

- MNI-space preprocessed image
- MNI-space brain mask
- Rigid transform matrix from native space to MNI space

Outputs are resampled onto the fixed MNI template grid. Smaller brains will be surrounded by zeros. Anatomy outside the template field of view after rigid alignment will be clipped at the template boundary.

## Docker

Build the image with Docker:

```bash
docker build -t anat-preproc .
```

Run it against a BIDS directory:

```bash
docker run --rm \
  -v "$PWD/data/raw:/app/data/raw" \
  -v "$PWD/data/processed:/app/data/processed" \
  -v "$PWD/data/logs:/app/data/logs" \
  anat-preproc
```

If you need to override the baked-in template brain path for debugging, pass `--template_brain` to the container entrypoint.

## Structural Extraction

The repo also includes a zip-to-BIDS extractor for the studio datasets:

- `aabc`: keep only main `T1w` and `T2w` plus shipped JSON sidecars
- `hcpya`: keep only `T1w_MPR1` and `T2w_SPC1`, written without a session label

By default it reads from `/teamspace/studios/this_studio/{aabc,hcpya}` and writes separate BIDS trees to:

- `data/aabc_bids`
- `data/hcpya_bids`

Run both datasets:

```bash
python extract_bids.py
```

Run one dataset only:

```bash
python extract_bids.py --dataset aabc
python extract_bids.py --dataset hcpya
```
