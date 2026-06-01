#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#
# This code accompanies the research paper: Upchurch, Paul, and Ransen
# Niu. "A Dense Material Segmentation Dataset for Indoor and Outdoor
# Scene Parsing." ECCV 2022.
#
# This example shows how to use the rotation descriptor and verify images.
#

import argparse
import json
import gzip
import os
import cv2
import numpy as np
import PIL.Image as Image
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm


def rotation_descriptor(ipath):
    # read as grayscale directly — avoids decoding 3 channels then averaging
    orig_img = cv2.imread(ipath, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    img = cv2.resize(orig_img, (90, 90), interpolation=cv2.INTER_AREA)
    dx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
    dy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
    angles = np.rad2deg(np.arctan2(
        dy.reshape(3, 30, 3, 30).mean(axis=(1, 3)),
        dx.reshape(3, 30, 3, 30).mean(axis=(1, 3)),
    )) % 360
    return orig_img.shape, angles


def angle_subtract(a, b):
    # convert angle difference to [-180, 180] range via atan2
    return np.rad2deg(np.arctan2(
        np.sin(np.deg2rad(a - b)),
        np.cos(np.deg2rad(a - b)),
    ))


def rotation_descriptor_match(a, b, percentile, tolerance, max_tolerance):
    # a matches b if percentile elements are within the tolerance and
    # no element is beyond the maximum tolerance.
    absdiff = np.abs(angle_subtract(a, b))
    return (absdiff <= tolerance).mean() >= percentile / 100 and np.all(absdiff <= max_tolerance)


def process_datum(datum, data_path, existing_files, skip_rotation):
    """
    Process a single datum entry. Returns a list of (status, path) tuples
    where status is one of: 'passed', or a key from the issues dict.

    Missing-file checks are done via set lookup before this function is
    called in bulk, so only data with both files present reaches the
    heavier checks below.
    """
    p1 = os.path.join(data_path, datum['image_path'])
    p2 = os.path.join(data_path, datum['label_path'])

    # size match?
    shape, rd = rotation_descriptor(p1)
    label = cv2.imread(p2, cv2.IMREAD_UNCHANGED)
    if label is None or shape[0] != label.shape[0] or shape[1] != label.shape[1]:
        return [('incorrect size', p1)]

    results = []

    # rotation match? (skippable via --skip-rotation)
    if not skip_rotation:
        ref = np.asarray(datum['rotation_descriptor'])
        if not rotation_descriptor_match(rd, ref, 50, 5, 30):
            results.append(('rotation may not match', p1))

    # inconsistent size between PIL and cv2?
    pil_size = Image.open(p1).size
    if shape[0] != pil_size[1] or shape[1] != pil_size[0]:
        results.append(('PIL and cv2 size disagreement', p1))

    return results if results else [('passed', p1)]


def main(args):
    # resolve once so walk output and constructed paths always match
    data_path = os.path.abspath(args.data_path)

    data = json.loads(
        gzip.open(os.path.join(data_path, 'info.json.gz'), 'rb').read()
    )
    print(f'Dataset path is {data_path}.')
    print(f'Dataset describes {len(data)} images.')
    if args.skip_rotation:
        print('Rotation check skipped (--skip-rotation).')

    # walk full directory tree once — avoids per-file stat() calls
    print('Scanning directory...')
    existing_files = {
        os.path.join(root, f)
        for root, _, files in os.walk(data_path)
        for f in files
    }
    print(f'Found {len(existing_files)} files in dataset directory.')

    issues = {
        'image not found': [],
        'image found, label not found': [],
        'incorrect size': [],
        'rotation may not match': [],
        'PIL and cv2 size disagreement': [],
    }

    # filter out missing files up front — no need to dispatch to workers
    pending = []
    for datum in data:
        p1 = os.path.join(data_path, datum['image_path'])
        p2 = os.path.join(data_path, datum['label_path'])
        if p1 not in existing_files:
            issues['image not found'].append(p1)
        elif p2 not in existing_files:
            issues['image found, label not found'].append(p1)
        else:
            pending.append(datum)

    skipped = len(data) - len(pending)
    if skipped:
        print(f'Skipping {skipped} entries with missing files.')
    print(f'Processing {len(pending)} entries with worker pool.')

    num_workers = args.workers if args.workers > 0 else cpu_count()
    print(f'Using {num_workers} worker processes.')

    passed = 0
    worker_fn = partial(
        process_datum,
        data_path=data_path,
        existing_files=existing_files,
        skip_rotation=args.skip_rotation,
    )

    with Pool(processes=num_workers) as pool:
        results_iter = pool.imap(worker_fn, pending, chunksize=args.chunksize)
        for result_list in tqdm(results_iter, total=len(pending), desc='Verifying images', unit='img'):
            for status, path in result_list:
                if status == 'passed':
                    passed += 1
                else:
                    issues[status].append(path)

    print(f'{passed} passed with no issues.')
    print('Summary of issues found:')
    for k, v in issues.items():
        print(f'  {k}: {len(v)}')
    print('See image_issues.json for details.')

    issues['instructions'] = (
        'An incorrect size is resolved by resizing the image to match the label map. '
        'Potential rotation mismatches are resolved by rotating the image so its orientation '
        'matches the label map. A size disagreement between PIL and cv2 is likely due to EXIF '
        'tags, which is resolved by stripping EXIF from the image and rotating the image so its '
        'orientation matches the label map.'
    )
    open('image_issues.json', 'w').write(json.dumps(issues, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path', type=str, required=True, help='path to DMS dataset'
    )
    parser.add_argument(
        '--workers', type=int, default=0,
        help='number of worker processes (default: 0 = use all CPU cores)'
    )
    parser.add_argument(
        '--chunksize', type=int, default=8,
        help='number of items per worker chunk (default: 8)'
    )
    parser.add_argument(
        '--skip-rotation', action='store_true',
        help='skip rotation descriptor check (faster, but will not detect orientation mismatches)'
    )
    args = parser.parse_args()
    main(args)