"""
count_truncated_fast.py
Only checks file sizes — no OpenCV, runs in seconds.
"""
from pathlib import Path

YOLO_ROOT      = Path('/project/outputs/yolo')
TRUNCATED_SIZE = 1_048_576  # exactly 1MB
SPLITS         = ['train', 'val', 'test']

print('Counting truncated (1MB) files across all splits...\n')

grand_total = 0
for split in SPLITS:
    img_dir = YOLO_ROOT / 'images' / split
    all_imgs = list(img_dir.glob('*.png')) + list(img_dir.glob('*.jpg'))
    truncated = [p for p in all_imgs if p.stat().st_size == TRUNCATED_SIZE]
    grand_total += len(truncated)
    print(f'[{split:5s}]  total={len(all_imgs):5d}  truncated={len(truncated):4d}')
    if truncated:
        print(f'         examples: {[p.name for p in truncated[:3]]}')

print(f'\nTotal truncated: {grand_total}')
