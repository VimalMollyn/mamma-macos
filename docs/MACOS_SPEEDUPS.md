# macOS (Apple Silicon) speedups

The pipeline runs on macOS/MPS out of the box (see the port changes). Two
optional accelerations target the slow steps on Apple GPUs.

## ma_2d on MPS (automatic)

The MammaNet landmark model runs on MPS automatically when CUDA is absent
(~10x faster than CPU; the detectron2 detector stays on CPU). No setup.

## EfficientTAM for ma_masks (opt-in)

`ma_masks` (SAM2) is the slowest step. SAM2 Hiera-Large is compute-bound on
MPS (~1.5 s/frame, no clean win — autocast/batching don't help on Apple GPU).
[EfficientTAM](https://github.com/yformer/EfficientTAM) is a lighter,
SAM2-compatible video model: its encoder is ~9x faster on MPS, cutting
`ma_masks` roughly in half end-to-end, with mask IoU ~0.97 vs SAM2 (final
SMPL-X fit within ~mm translation / ~2° pose).

### Enable it

```bash
# 1. Install EfficientTAM into the venv
uv pip install "git+https://github.com/yformer/EfficientTAM.git"

# 2. Its wheel omits the config yamls — fetch them into the package
ETAM=$(python -c "import efficient_track_anything,os;print(os.path.dirname(efficient_track_anything.__file__))")
mkdir -p "$ETAM/configs/efficienttam"
base="https://raw.githubusercontent.com/yformer/EfficientTAM/main/efficient_track_anything/configs/efficienttam"
for c in efficienttam_ti efficienttam_s efficienttam_ti_512x512 efficienttam_s_512x512; do
  wget -q -O "$ETAM/configs/efficienttam/$c.yaml" "$base/$c.yaml"
done

# 3. Weights (public, no login)
mkdir -p data/weights/efficienttam
wget -O data/weights/efficienttam/efficienttam_ti.pt \
  https://huggingface.co/yunyangx/efficient-track-anything/resolve/main/efficienttam_ti.pt
```

Then turn it on via `.env.local` (gitignored; leaves shipped presets on SAM2):

```bash
MAMMA_MA_MASKS_SAM_VERSION=efficienttam_ti
```

The runner appends `--sam_version efficienttam_ti` to `ma_masks`, overriding
the preset. Variants: `efficienttam_ti` (fastest, recommended), `efficienttam_s`
(closer to SAM2 quality), and `*_512x512` (lower input res, even faster — only
if coarse masks suffice). Place the matching `<variant>.pt` under
`data/weights/efficienttam/`, or set `MAMMA_ETAM_CHECKPOINT` / `--sam_checkpoint`.

### SAM2-variant fallback (no new dependency)

If you'd rather not add EfficientTAM, a smaller SAM2 checkpoint also helps
(encoder, MPS): hiera_base_plus ~2x, hiera_small ~4x. Download from
`dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_<variant>.pt`
and point `MAMMA_SAM2_CHECKPOINT` at it (the architecture config is auto-selected
by SAM2's hydra from the variant).
