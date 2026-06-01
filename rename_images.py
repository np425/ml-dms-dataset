import argparse
import gzip
import json
import os
from tqdm import tqdm


def main(args):
    data_path = os.path.abspath(args.data_path)

    with gzip.open(os.path.join(data_path, 'info.json.gz'), 'rb') as f:
        data = json.loads(f.read())

    print(f'Loaded {len(data)} entries.')

    renamed = 0
    skipped = 0
    missing = 0

    for datum in tqdm(data, desc='Renaming', unit='img'):
        image_id = datum['openimages_metadata']['ImageID']
        subset = datum['openimages_metadata']['Subset']

        src = os.path.join(data_path, 'images', subset, f'{image_id}.jpg')
        dst = os.path.join(data_path, datum['image_path'])

        if not os.path.exists(src):
            missing += 1
            continue

        if os.path.exists(dst):
            skipped += 1
            continue

        os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.rename(src, dst)
        renamed += 1

    print(f'Renamed:  {renamed}')
    print(f'Skipped (already exists): {skipped}')
    print(f'Missing (not downloaded): {missing}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True, help='path to DMS dataset (contains info.json.gz)')
    args = parser.parse_args()
    main(args)