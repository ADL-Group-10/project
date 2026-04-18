# project
https://excalidraw.com/#json=Q3hyRhoJXY1I4pRRY8ZIQ,XmrgmnoBZrxzP5yRJWiLbA
<img width="1580" height="996" alt="image" src="https://github.com/user-attachments/assets/c169dc25-d326-4d29-adad-ecd4209ab855" />

### Data Pipeline

Converts the Nordic Vehicle Dataset (NVD) into YOLO-ready format for YOLOv9 training.

#### Files

```
src/data/
    data_pipeline.py       # DataPipeline — main orchestrator
    snow_augmentation.py   # SnowAugmentation — snow-specific transforms
    __init__.py
```

#### How it works

**On first run**, `DataPipeline` detects that processed YOLO data doesn't exist and automatically runs the setup chain:

1. **Parse** — Reads CVAT 1.1 XML annotations per video sequence
2. **Extract** — Pulls frames from .mp4 using NVIDIA DALI (GPU-accelerated, falls back to OpenCV if unavailable). For .png sequences, copies directly
3. **Convert** — Transforms CVAT bounding boxes to YOLO format. CVAT stores boxes as absolute pixel coordinates using four corners: `xtl, ytl` (top-left) and `xbr, ybr` (bottom-right). YOLO expects center point + dimensions, all normalized to 0–1. For example on a 1920×1080 frame:
   ```
   CVAT:  xtl=1064, ytl=100, xbr=1123, ybr=212   (pixels, corners)
   YOLO:  0  0.5695  0.1444  0.0307  0.1037        (class_id, x_center, y_center, w, h)
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
pipeline.show_samples(n=5, augment="snow")  # visual sanity check
```

#### Augmentation design

| Level | Transforms | Experiment |
|-------|-----------|------------|
| `none` | Resize, Normalize | Clean baseline |
| `base` | HorizontalFlip, HSV jitter, Resize, Normalize | V1 baseline |
| `snow` | Base + desaturation, blur, brightness jitter, snow/fog overlay | V2 snow-augmented |

- **Base transforms** are built inside `DataPipeline` using params from `config.yaml → augmentation.standard`
- **Snow transforms** are owned by `SnowAugmentation`, called only when `augment="snow"`, using params from `config.yaml → augmentation.snow`
- **Albumentations** handles all augmentations
- **NVIDIA DALI** handles video decoding (GPU-accelerated frame extraction)

#### Config keys used

```yaml
paths.nvd_root          # raw NVD dataset location
paths.yolo_output       # processed YOLO output location
paths.splits            # train/val/test sequence assignment
model.img_size          # resize target (640)
model.num_classes       # 1 (car)
augmentation.standard   # base augmentation params
augmentation.snow       # snow augmentation params
```

#### Dataset split (per LTU specification)

| Sequence | Split | Source |
|----------|-------|--------|
| 2022-12-04 Bjenberg 02 | Train | .mp4 |
| 2022-12-23 Asjo 01_HD 5x stab | Train | .mp4 |
| 2022-12-02 Asjo 01_stabilized | Train | .mp4 |
| 2022-12-03 Nyland 01_stabilized | Val | .mp4 |
| 2022-12-23 Bjenberg 02_stabilized | Test | .png |