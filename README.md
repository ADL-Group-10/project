# NVD Car Detection in Snow — YOLOv9

Car detection in snowy conditions using YOLOv9 on the Nordic Vehicle Dataset (NVD).
Course project for D7047E Advanced Deep Learning, LTU Group 10.

Three-variant experiment:
- **V1** — baseline augmentation (geometric + HSV jitter)
- **V2** — V1 + snow-specific augmentation (perspective, desaturation, blur, brightness jitter, snow overlay, motion blur for snowfall, optional color inversion)
- **V3** — domain shift: V2's snow stack split per-split, with `light_snow` intensity on train+val and `heavy_snow` intensity on test

---

### Data Pipeline

Converts the Nordic Vehicle Dataset (NVD) into YOLO-ready format for YOLOv9 training.

#### Files

```
src/data/
    data_pipeline.py       # DataPipeline — main orchestrator
    snow_augmentation.py   # SnowAugmentation — snow-specific transforms
    visualizer.py          # AnnotationVisualizer — dual-box + zoom visualization
    __init__.py
```

#### How it works

**On first run**, `DataPipeline` detects that processed YOLO data doesn't exist and automatically runs the setup chain:

1. **Parse** — Reads CVAT 1.1 XML annotations per video sequence. Resolution is read from the XML `<meta><original_size>` tag (not hardcoded). **Rotated bounding boxes** are handled: CVAT stores `(xtl, ytl, xbr, ybr, rotation)` where `rotation` is degrees around the box center. The parser rotates all 4 corners and computes the axis-aligned envelope — matching the [official NVD repo](https://github.com/Amirhossein-Nayebi/Nordic-Vehicle-Dataset)'s `AnnotationBox.GetStraightBoundingBox()`. Across the 5 NVD sequences, 99.5% of boxes have non-trivial rotation (up to 360°).
2. **Extract** — Pulls frames from .mp4 using decord (0-based frame indexing, matching CVAT). For .png sequences, copies directly.
3. **Convert** — Transforms rotated CVAT boxes to YOLO format:

   ```
   CVAT:  xtl=913.9, ytl=276.3, xbr=936.0, ybr=319.5, rotation=243.29°
   After rotation → straight bbox: [900.7, 278.3, 949.2, 317.5] (49×39 px)
   YOLO:  0  0.4817  0.2758  0.0253  0.0363   (class_id, x_center, y_center, w, h)
   ```
4. **Organize** — Places frames and labels into `images/{train,val,test}/` and `labels/{train,val,test}/`
5. **Generate** — Writes `dataset.yaml` for YOLOv9
6. **Validate** — Checks image↔label pairing, label format, no empty splits

Only annotated frames are saved — unannotated frames are skipped.
Subsequent runs skip setup entirely (idempotent).

#### Usage

```python
from src.data import DataPipeline

pipeline = DataPipeline("config.yaml")

# Three augmentation levels
path, aug = pipeline.run(augment="none")    # resize + normalize only
path, aug = pipeline.run(augment="base")    # + geometric, HSV jitter
path, aug = pipeline.run(augment="snow")    # + snow-aware transforms

# Inspect
pipeline.summary()                          # per-split image/bbox counts
```

#### Annotation visualization

`AnnotationVisualizer` (`src/data/visualizer.py`) is a standalone class for drawing bounding boxes with zoom insets. It accepts either CVAT annotation dicts (draws both rotated polygon and axis-aligned box) or plain YOLO boxes. The same class is used by the data pipeline for sample previews and can be reused for evaluation visualization.

```python
from src.data.visualizer import AnnotationVisualizer as AV

# From CVAT XML — dual boxes (rotated polygon + axis-aligned) + zoom
anns, w, h = AV.parse_cvat_frame("annotation.xml", frame_num=1721)
vis = AV.visualize(image, anns)

# From YOLO boxes — axis-aligned + zoom
vis = AV.visualize(image, [[0.48, 0.28, 0.05, 0.04], ...])

# Comparison grid
fig = AV.comparison_grid([("raw", img1), ("snow", img2)])
AV.save_figure(fig, "comparison.png")
```

Running `python -m src.data.data_pipeline` generates two images in `outputs/yolo/samples/`:
- `comparison_variants.png` — raw, V1, V2, V3 train, V3 test with dual-box + zoom
- `comparison_transforms.png` — individual transforms at p=1.0 with zoom insets

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
| Preview variants | open after running | `outputs/yolo/samples/comparison_variants.png` |
| Preview transforms | open after running | `outputs/yolo/samples/comparison_transforms.png` |
| Active variant | `config.yaml` | `variants.active: v1\|v2\|v3` |
| Snow density | `config.yaml` | `augmentation.snow.snow_overlay.{snow_point_lower, snow_point_upper, p}` |
| Snowfall streaks | `config.yaml` | `augmentation.snow.motion_blur.{blur_limit, p}` |
| Camera tilt | `config.yaml` | `augmentation.snow.perspective.{scale, p}` |
| Whiteout / dim / fog | `config.yaml` | `augmentation.snow.{desaturation, brightness_jitter, blur}` |
| Color inversion (off by default) | `config.yaml` | `augmentation.snow.invert.{enabled, threshold, p}` |
| V3 per-split intensity | `config.yaml` | `domain_shift.light_snow.*` (train+val), `domain_shift.heavy_snow.*` (test) |
| Add a new transform | code | `SnowAugmentation.get()` + mirror config block under `augmentation.snow.<name>` |

Every block under `augmentation.snow.*` has the same shape: `enabled: true|false`, `p` (probability per image), plus transform-specific params. Flip `enabled` to skip without deleting config.

#### Augmentation design

| Level | Transforms | Experiment |
|---|---|---|
| `none` | Resize, Normalize | Clean baseline |
| `base` | HorizontalFlip, HSV jitter, Resize, Normalize | V1 baseline |
| `snow` | Base + desaturation, blur, brightness jitter, snow/fog overlay | V2 snow-augmented |

* **Base transforms** are built inside `DataPipeline` using params from `config.yaml → augmentation.standard`
* **Snow transforms** are owned by `SnowAugmentation`, called only when `augment="snow"`, using params from `config.yaml → augmentation.snow`
* **Albumentations** handles all augmentations with `BboxParams(format="yolo", min_visibility=0.3)` so bounding boxes transform alongside images for geometric augmentations (flip, perspective)
* **decord** handles frame extraction from .mp4 using 0-based frame indexing (matching CVAT annotation frame numbers)
* **snow_overlay.p** is set high (0.85 for V2, 0.8/0.95 for V3 light/heavy) to guarantee visible snow in every V2/V3 image; other transforms fire probabilistically on top

#### Config keys used

```
paths.nvd_root          # raw NVD dataset location
paths.yolo_output       # processed YOLO output location
paths.splits            # train/val/test sequence assignment
model.img_size          # resize target (640)
model.num_classes       # 1 (car)
augmentation.standard   # base augmentation params
augmentation.snow       # snow augmentation params
```

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
| `snow_overlay.snow_point_upper` | 0.35 | 0.5–0.7 |
| `snow_overlay.snow_point_lower` | 0.15 | 0.3 |
| `snow_overlay.p` | 0.85 | 0.95 |
| `motion_blur.blur_limit` | [3, 7] | [5, 11] |
| `motion_blur.p` | 0.3 | 0.5 |
| `perspective.scale` | [0.03, 0.07] | [0.05, 0.10] |
| `perspective.p` | 0.3 | 0.5 |
| `desaturation.saturation_limit` | [-0.4, -0.1] | [-0.6, -0.3] |
| `brightness_jitter.brightness_limit` | [-0.10, 0.03] | [-0.18, -0.03] |
| `blur.blur_limit` | [3, 5] | [3, 7] |
| `invert.threshold` | [128, 220] | [100, 180] |
| `invert.enabled` | `false` | `true` |
| `invert.p` | 0.05 | ≤ 0.10 |

`invert` uses `A.Solarize` — inverts only pixels above `threshold`. Off by default because color statistics (snow is bright / desaturated) are part of the learning signal, and real inference images are never inverted. `perspective` may drop bboxes that fall below 30% visible after warp (`BboxParams(min_visibility=0.3)`).

#### Domain shift (V3)

V3 swaps the single pipeline for two: `light_snow` (train+val) and `heavy_snow` (test). Light ≈ ½ × V2; heavy ≈ 2 × V2.

| Knob | light_snow (train+val) | V2 baseline | heavy_snow (test) |
|---|---|---|---|
| `perspective.scale` | [0.02, 0.04] | [0.03, 0.07] | [0.05, 0.10] |
| `perspective.p` | 0.2 | 0.3 | 0.4 |
| `motion_blur.blur_limit` | [3, 5] | [3, 7] | [5, 11] |
| `motion_blur.p` | 0.2 | 0.3 | 0.4 |
| `snow_overlay.snow_point_upper` | 0.25 | 0.35 | 0.55 |
| `snow_overlay.p` | 0.8 | 0.85 | 0.95 |
| `desaturation.saturation_limit` | [-0.3, -0.08] | [-0.4, -0.1] | [-0.6, -0.3] |
| `brightness_jitter.brightness_limit` | [-0.07, 0.02] | [-0.10, 0.03] | [-0.18, -0.03] |
| `blur.blur_limit` | [3, 5] | [3, 5] | [3, 7] |
| `invert.threshold` | [180, 230] | [128, 220] | [100, 200] |
| `invert.p` | 0.02 | 0.05 | 0.10 |

`invert.enabled` stays `false` in all three. Activate V3 with `pipeline.run(augment="snow", domain_shift=True)`.
