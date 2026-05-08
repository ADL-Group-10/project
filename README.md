# NVD Car Detection in Snow — YOLOv9

Car detection in snowy conditions using YOLOv9 on the Nordic Vehicle Dataset (NVD).
Course project for D7047E Advanced Deep Learning, LTU Group 10.

Three-variant experiment:
- **V1** — baseline augmentation (geometric + HSV jitter)
- **V2** — V1 + snow-specific augmentation (perspective, desaturation, blur, brightness jitter, snow overlay, motion blur for snowfall, optional color inversion)
- **V3** — domain shift: V2's snow stack split per-split, with `light_snow` intensity on train+val and `heavy_snow` intensity on test

---

### Data Pipeline

Owned by `src/data/data_pipeline.py` (`DataPipeline`) and `src/data/snow_augmentation.py` (`SnowAugmentation`).

#### Data design

NVD raw `.mp4` / `.png` sequences → annotated frames extracted via decord random access (mp4) or direct copy (png) → CVAT corner bboxes converted to YOLO normalized center format → written to `outputs/yolo/{images,labels}/{train,val,test}/` plus `dataset.yaml`. Idempotent: re-runs skip extraction. On top of that on-disk tree, `SnowAugmentation` builds the Albumentations transform list that V2 and V3 apply per image.

#### Data splits

| Sequence | Split | Source |
|---|---|---|
| 2022-12-04 Bjenberg 02 | train | .mp4 |
| 2022-12-23 Asjo 01_HD 5x stab | train | .mp4 |
| 2022-12-02 Asjo 01_stabilized | train | .mp4 |
| 2022-12-03 Nyland 01_stabilized | val | .mp4 |
| 2022-12-23 Bjenberg 02_stabilized | test | .png |

Change at `config.yaml -> paths.splits.{train,val,test}`.

#### What to toggle and where

| Goal | Where | Key |
|---|---|---|
| Run pipeline | terminal | `python -m src.data.data_pipeline` |
| Preview transforms | open after running | `outputs/yolo/samples/comparison.png` |
| Active variant | `config.yaml` | `variants.active: v1\|v2\|v3` |
| Snow density | `config.yaml` | `augmentation.snow.snow_overlay.{snow_point_lower, snow_point_upper, p}` |
| Snowfall streaks | `config.yaml` | `augmentation.snow.motion_blur.{blur_limit, p}` |
| Camera tilt | `config.yaml` | `augmentation.snow.perspective.{scale, p}` |
| Whiteout / dim / fog | `config.yaml` | `augmentation.snow.{desaturation, brightness_jitter, blur}` |
| Color inversion (off by default) | `config.yaml` | `augmentation.snow.invert.{enabled, threshold, p}` |
| V3 per-split intensity | `config.yaml` | `domain_shift.light_snow.*` (train+val), `domain_shift.heavy_snow.*` (test) |
| Add a new transform | code | `SnowAugmentation.get()` + mirror config block under `augmentation.snow.<name>` |

Every block under `augmentation.snow.*` has the same shape: `enabled: true|false`, `p` (probability per image), plus transform-specific params. Flip `enabled` to skip without deleting config.

#### Transformation details

`SnowAugmentation.get()` returns the list in this order — geometric first so bboxes get remapped against the original pixel grid before color / noise transforms touch the image.

| # | Block | Albumentations | Simulates |
|---|---|---|---|
| 1 | `perspective` | `A.Perspective` | camera tilt; remaps bboxes |
| 2 | `desaturation` | `A.HueSaturationValue` (sat only) | whiteout / loss of color |
| 3 | `blur` | `A.GaussianBlur` | fog / out-of-focus |
| 4 | `brightness_jitter` | `A.RandomBrightnessContrast` | overcast darkening |
| 5 | `snow_overlay` | `A.RandomSnow` | synthetic snowflakes |
| 6 | `motion_blur` | `A.MotionBlur` | snowfall streaks (after overlay so flakes also streak) |
| 7 | `invert` | `A.Solarize` (threshold-based) | pixel inversion above threshold |

V1 (`augment="base"`) is just `HorizontalFlip` + `HueSaturationValue` from `augmentation.standard`. V2 (`augment="snow"`) layers the full stack on top.

Knob defaults under `augmentation.snow.*` (V2):

| Knob | Default | "More" |
|---|---|---|
| `snow_overlay.snow_point_upper` | 0.3 | 0.5–0.7 |
| `snow_overlay.snow_point_lower` | 0.1 | 0.3 |
| `snow_overlay.p` | 0.4 | 0.8 |
| `motion_blur.blur_limit` | [5, 11] | [11, 21] |
| `motion_blur.p` | 0.4 | 0.7 |
| `perspective.scale` | [0.05, 0.10] | [0.10, 0.18] |
| `perspective.p` | 0.4 | 0.6 |
| `desaturation.saturation_limit` | [-0.6, -0.2] | [-0.9, -0.5] |
| `brightness_jitter.brightness_limit` | [-0.15, 0.05] | [-0.30, -0.05] |
| `blur.blur_limit` | [3, 7] | [5, 13] |
| `invert.threshold` | [128, 220] | [100, 180] |
| `invert.enabled` | `false` | `true` |
| `invert.p` | 0.05 | ≤ 0.10 |

`invert` uses `A.Solarize` — inverts only pixels above `threshold`. Off by default because color statistics (snow is bright / desaturated) are part of the learning signal, and real inference images are never inverted. `perspective` may drop bboxes that fall below 30% visible after warp (`BboxParams(min_visibility=0.3)`).

#### Domain shift (V3)

V3 swaps the single pipeline for two: `light_snow` (train+val) and `heavy_snow` (test). Light ≈ ½ × V2; heavy ≈ 2 × V2.

| Knob | light_snow (train+val) | V2 baseline | heavy_snow (test) |
|---|---|---|---|
| `perspective.scale` | [0.02, 0.05] | [0.05, 0.10] | [0.08, 0.15] |
| `perspective.p` | 0.2 | 0.4 | 0.6 |
| `motion_blur.blur_limit` | [3, 5] | [5, 11] | [9, 17] |
| `motion_blur.p` | 0.2 | 0.4 | 0.6 |
| `snow_overlay.snow_point_upper` | 0.15 | 0.3 | 0.5 |
| `snow_overlay.p` | 0.2 | 0.4 | 0.6 |
| `desaturation.saturation_limit` | [-0.3, -0.1] | [-0.6, -0.2] | [-0.8, -0.4] |
| `brightness_jitter.brightness_limit` | [-0.08, 0.03] | [-0.15, 0.05] | [-0.25, -0.05] |
| `blur.blur_limit` | [3, 5] | [3, 7] | [5, 11] |
| `invert.threshold` | [180, 230] | [128, 220] | [100, 200] |
| `invert.p` | 0.02 | 0.05 | 0.10 |

`invert.enabled` stays `false` in all three. Activate V3 with `pipeline.run(augment="snow", domain_shift=True)`.
