"""
check_and_fix_filenames.py
Run this on the cluster:  python check_and_fix_filenames.py

Diagnoses file naming mismatches between images and labels,
then optionally renames to fix them.

From the training logs we already know images are named like:
  2022-12-23_Bjenberg_02_stabilized_frame_000008.png

This script checks whether labels match that exact stem.
"""

import os
from pathlib import Path

YOLO_ROOT = Path('/project/outputs/yolo')
SPLITS    = ['train', 'val', 'test']

DRY_RUN = True   # ← Set to False to actually rename files


def audit_split(split):
    img_dir = YOLO_ROOT / 'images' / split
    lbl_dir = YOLO_ROOT / 'labels' / split

    if not img_dir.exists():
        print(f'  [MISSING] {img_dir}')
        return

    images = list(img_dir.glob('*'))
    labels = list(lbl_dir.glob('*.txt'))

    img_stems = {p.stem: p for p in images if p.suffix in ('.png', '.jpg', '.jpeg')}
    lbl_stems = {p.stem: p for p in labels}

    matched        = set(img_stems) & set(lbl_stems)
    imgs_no_label  = set(img_stems) - set(lbl_stems)
    lbls_no_image  = set(lbl_stems) - set(img_stems)

    print(f'\n[{split.upper()}]')
    print(f'  Images        : {len(img_stems)}')
    print(f'  Label files   : {len(lbl_stems)}')
    print(f'  Matched pairs : {len(matched)}')
    print(f'  Images WITHOUT label : {len(imgs_no_label)}')
    print(f'  Labels WITHOUT image : {len(lbls_no_image)}')

    # Show extension breakdown
    from collections import Counter
    img_exts = Counter(p.suffix for p in img_stems.values())
    print(f'  Image extensions: {dict(img_exts)}')

    # Show naming pattern sample
    if img_stems:
        sample = list(img_stems.keys())[:3]
        print(f'  Image stem sample: {sample}')
    if lbl_stems:
        sample = list(lbl_stems.keys())[:3]
        print(f'  Label stem sample: {sample}')

    # Detect common prefix/suffix differences that would explain mismatches
    if imgs_no_label and lbls_no_image:
        # Pick one from each side and compare
        img_ex = list(imgs_no_label)[0]
        lbl_ex = list(lbls_no_image)[0]
        print(f'\n  ⚠ MISMATCH EXAMPLE:')
        print(f'    Image stem : "{img_ex}"')
        print(f'    Label stem : "{lbl_ex}"')

        # Check common transformation patterns
        checks = {
            'spaces→underscores in label': img_ex.replace(' ', '_') == lbl_ex,
            'spaces→underscores in image': lbl_ex.replace(' ', '_') == img_ex,
            'label has extra prefix':      lbl_ex.endswith(img_ex),
            'image has extra prefix':      img_ex.endswith(lbl_ex),
            'case difference':             img_ex.lower() == lbl_ex.lower(),
        }
        for desc, match in checks.items():
            if match:
                print(f'    ✓ Pattern detected: {desc}')

    return {
        'matched': len(matched),
        'imgs_no_label': imgs_no_label,
        'lbls_no_image': lbls_no_image,
        'img_stems': img_stems,
        'lbl_stems': lbl_stems,
        'img_dir': img_dir,
        'lbl_dir': lbl_dir,
    }


def try_fix_spaces_to_underscores(result, split):
    """
    Most common issue: image filenames have spaces, labels have underscores
    (or vice versa). Try to rename image files to match label stems.
    """
    if not result:
        return

    imgs_no_label = result['imgs_no_label']
    lbl_stems     = result['lbl_stems']
    img_stems     = result['img_stems']
    img_dir       = result['img_dir']

    fixed = 0
    cant_fix = []

    for stem in list(imgs_no_label):
        # Try space → underscore
        candidate = stem.replace(' ', '_')
        if candidate in lbl_stems:
            old_path = img_stems[stem]
            new_path = img_dir / (candidate + old_path.suffix)
            if DRY_RUN:
                print(f'  [DRY RUN] Would rename: {old_path.name} → {new_path.name}')
            else:
                old_path.rename(new_path)
                print(f'  Renamed: {old_path.name} → {new_path.name}')
            fixed += 1
            continue

        # Try underscore → space
        candidate = stem.replace('_', ' ')
        if candidate in lbl_stems:
            old_path = img_stems[stem]
            new_path = img_dir / (candidate + old_path.suffix)
            if DRY_RUN:
                print(f'  [DRY RUN] Would rename: {old_path.name} → {new_path.name}')
            else:
                old_path.rename(new_path)
            fixed += 1
            continue

        cant_fix.append(stem)

    print(f'\n  Fixable by rename   : {fixed}')
    print(f'  Cannot auto-fix     : {len(cant_fix)}')
    if cant_fix:
        print(f'  Examples of unfixable mismatches:')
        for s in cant_fix[:5]:
            print(f'    "{s}"')


def check_dataset_yaml():
    """Check the dataset.yaml references correct paths."""
    yaml_path = YOLO_ROOT / 'dataset.yaml'
    if not yaml_path.exists():
        print(f'\n⚠ dataset.yaml not found at {yaml_path}')
        return
    print(f'\n=== dataset.yaml ===')
    print(yaml_path.read_text())


def check_corrupt_images(split, max_check=50):
    """Quick check for images OpenCV cannot read."""
    import cv2
    img_dir = YOLO_ROOT / 'images' / split
    if not img_dir.exists():
        return

    images  = list(img_dir.glob('*.png')) + list(img_dir.glob('*.jpg'))
    sample  = images[:max_check]
    corrupt = []

    for p in sample:
        img = cv2.imread(str(p))
        if img is None:
            corrupt.append(p.name)

    print(f'\n[{split.upper()}] Corrupt image check (first {max_check}):')
    print(f'  Checked: {len(sample)} | Corrupt: {len(corrupt)}')
    if corrupt:
        print(f'  Examples: {corrupt[:5]}')
        # Check if they're actually zero-byte files
        for name in corrupt[:3]:
            size = (img_dir / name).stat().st_size
            print(f'    {name}  →  {size} bytes')


# ── Run ───────────────────────────────────────────────────────────────────────
print('FILE NAMING AUDIT')
print('=' * 60)
print(f'DRY_RUN = {DRY_RUN}  (set to False to apply fixes)\n')

check_dataset_yaml()

results = {}
for split in SPLITS:
    results[split] = audit_split(split)

print('\n\nFIX ATTEMPT — spaces ↔ underscores')
print('=' * 60)
for split in SPLITS:
    if results[split]:
        print(f'\n[{split.upper()}]')
        try_fix_spaces_to_underscores(results[split], split)

print('\n\nCORRUPT IMAGE CHECK')
print('=' * 60)
for split in SPLITS:
    check_corrupt_images(split)

print('\n\nDONE.')
print('If DRY_RUN=True above, set DRY_RUN=False and re-run to apply fixes.')
print('After fixing, re-run your training or validation.')
