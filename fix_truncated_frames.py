"""
fix_truncated_frames.py

Every corrupt test image is exactly 1,048,576 bytes (1MB) — a hard write-cap
truncation from the frame extraction step. This script:

1. Finds ALL truncated files across every split (not just the first 50)
2. Maps each back to its source video + frame number
3. Re-extracts those specific frames from the original video using ffmpeg
4. Replaces the truncated PNGs in-place
5. Verifies the fix

Run:  python fix_truncated_frames.py
"""

import cv2
import subprocess
import re
from pathlib import Path

YOLO_ROOT  = Path('/project/outputs/yolo')
NVD_ROOT   = Path('/project/src/data/NVD')   # adjust if different
SPLITS     = ['train', 'val', 'test']

TRUNCATED_SIZE = 1_048_576   # exactly 1MB

DRY_RUN = True  # set False to actually re-extract


#  Step 1: Find every truncated file 

def find_truncated(split):
    img_dir   = YOLO_ROOT / 'images' / split
    truncated = []
    all_imgs  = list(img_dir.glob('*.png')) + list(img_dir.glob('*.jpg'))

    for p in all_imgs:
        size = p.stat().st_size
        if size == TRUNCATED_SIZE:
            truncated.append(p)
            continue
        # Also catch anything OpenCV can't read regardless of size
        img = cv2.imread(str(p))
        if img is None and size > 0:
            truncated.append(p)

    return truncated


print('STEP 1 — Finding all truncated files')
print('=' * 60)
all_truncated = {}
total = 0
for split in SPLITS:
    t = find_truncated(split)
    all_truncated[split] = t
    total += len(t)
    sizes = sorted(set(p.stat().st_size for p in t))
    print(f'  [{split:5s}]  {len(t):4d} truncated  |  unique sizes: {sizes[:5]}')

print(f'\n  Total truncated across all splits: {total}')


# ── Step 2: Parse frame number and video name from filename ───────────────────
#
# Filename pattern:  {video_folder_underscored}_frame_{NNNNNN}.png
# Examples:
#   2022-12-23_Bjenberg_02_stabilized_frame_000091.png
#   2022-12-04_Bjenberg_02_frame_000763.png
#   2022-12-02_Asjo_01_stabilized_frame_001046.png
#
# The video folder name has spaces instead of underscores, e.g.:
#   "2022-12-23 Bjenberg 02_stabilized"
# But that depends on how NVD is stored. We probe both variants.

FRAME_RE = re.compile(r'^(.+)_frame_(\d+)$')

def stem_to_video_and_frame(stem):
    """
    Returns (video_folder_name, frame_index) or (None, None).
    Tries underscore→space variants to find the actual folder.
    """
    m = FRAME_RE.match(stem)
    if not m:
        return None, None

    folder_part = m.group(1)           # e.g. '2022-12-23_Bjenberg_02_stabilized'
    frame_idx   = int(m.group(2))     # e.g. 91

    # Try exact match first, then space variants
    candidates = [
        folder_part,
        folder_part.replace('_', ' '),
        # partial space: only first two underscores → spaces
    ]

    for c in candidates:
        if (NVD_ROOT / c).exists():
            return c, frame_idx

    # Fuzzy: find any NVD subfolder whose name matches when normalised
    norm = folder_part.lower().replace('_', '').replace(' ', '').replace('-', '')
    for d in NVD_ROOT.iterdir():
        if d.is_dir():
            dnorm = d.name.lower().replace('_', '').replace(' ', '').replace('-', '')
            if dnorm == norm:
                return d.name, frame_idx

    return None, None


def find_video_file(nvd_folder):
    """Find the .mp4 / .avi / .mov inside the NVD folder."""
    folder = NVD_ROOT / nvd_folder
    for ext in ('*.mp4', '*.avi', '*.mov', '*.mkv', '*.MP4', '*.AVI'):
        hits = list(folder.glob(ext))
        if hits:
            return hits[0]
    return None


# ── Step 3: Re-extract frames ─────────────────────────────────────────────────

print('\n\nSTEP 2 — Mapping truncated files to source videos')
print('=' * 60)

extraction_plan = []   # list of (truncated_path, video_file, frame_idx)
unmappable      = []

for split, files in all_truncated.items():
    for p in files:
        folder_name, frame_idx = stem_to_video_and_frame(p.stem)
        if folder_name is None:
            unmappable.append(p)
            continue
        video = find_video_file(folder_name)
        if video is None:
            unmappable.append(p)
            print(f'  ⚠ No video found in NVD/{folder_name}')
            continue
        extraction_plan.append((p, video, frame_idx))

print(f'  Mapped to video + frame : {len(extraction_plan)}')
print(f'  Could not map           : {len(unmappable)}')
if unmappable:
    print(f'  Unmappable examples:')
    for p in unmappable[:5]:
        print(f'    {p.name}')

# Show NVD structure if nothing mapped
if len(extraction_plan) == 0:
    print('\n  NVD directory contents:')
    if NVD_ROOT.exists():
        for d in sorted(NVD_ROOT.iterdir()):
            print(f'    {d.name}/')
            for f in list(d.iterdir())[:3]:
                print(f'      {f.name}')
    else:
        print(f'  ⚠ NVD_ROOT does not exist: {NVD_ROOT}')
        print('  Update NVD_ROOT at the top of this script.')


def extract_frame_ffmpeg(video_path, frame_idx, out_path):
    """
    Use ffmpeg to extract one specific frame by index.
    frame_idx is 0-based (matching how they were originally extracted).
    """
    # Select frame by PTS: frame N is at time N/fps
    # Safer: use select filter which is frame-accurate
    cmd = [
        'ffmpeg', '-y',
        '-i', str(video_path),
        '-vf', f'select=eq(n\\,{frame_idx})',
        '-vframes', '1',
        '-q:v', '2',
        str(out_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


print('\n\nSTEP 3 — Re-extracting frames')
print('=' * 60)
print(f'DRY_RUN = {DRY_RUN}')

fixed   = 0
failed  = []

for out_path, video, frame_idx in extraction_plan:
    if DRY_RUN:
        print(f'  [DRY] Would extract frame {frame_idx:06d} from {video.name} → {out_path.name}')
        fixed += 1
        continue

    tmp_path = out_path.with_suffix('.tmp.png')
    ok = extract_frame_ffmpeg(video, frame_idx, tmp_path)

    if not ok or not tmp_path.exists():
        failed.append((out_path.name, 'ffmpeg failed'))
        continue

    # Verify the new file is readable
    img = cv2.imread(str(tmp_path))
    if img is None:
        failed.append((out_path.name, 'still unreadable after extraction'))
        tmp_path.unlink(missing_ok=True)
        continue

    # Replace the truncated file
    tmp_path.rename(out_path)
    fixed += 1
    print(f'  ✓ Fixed: {out_path.name}  ({img.shape[1]}x{img.shape[0]})')

print(f'\n  Fixed   : {fixed}')
print(f'  Failed  : {len(failed)}')
for name, reason in failed[:10]:
    print(f'    ✗ {name}: {reason}')


# ── Step 4: Verify ────────────────────────────────────────────────────────────

if not DRY_RUN:
    print('\n\nSTEP 4 — Verification')
    print('=' * 60)
    remaining = 0
    for split in SPLITS:
        t = find_truncated(split)
        remaining += len(t)
        print(f'  [{split}] still truncated: {len(t)}')
    if remaining == 0:
        print('\n  ✅ All truncated files fixed. Re-run training/validation.')
    else:
        print(f'\n  ⚠ {remaining} files still broken. Check failed list above.')
else:
    print(f'\nSet DRY_RUN = False and re-run to apply the fix.')
