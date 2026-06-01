import argparse
import json
import gzip
import os
import boto3
import botocore
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from functools import partial

BUCKET_NAME = 'open-images-dataset'
s3_client = None

def init_worker():
    global s3_client
    s3_client = boto3.resource(
        's3', config=botocore.config.Config(signature_version=botocore.UNSIGNED)
    )

def download_one_image(datum, download_folder):
    relative_path = datum['image_path']
    local_path = os.path.join(download_folder, relative_path)

    if os.path.exists(local_path):
        return local_path, True, 'skipped'

    try:
        subset = datum['openimages_metadata']['Subset']
        image_id = datum['openimages_metadata']['ImageID']
        s3_key = f'{subset}/{image_id}.jpg'
        s3_client.Bucket(BUCKET_NAME).download_file(s3_key, local_path)
        return local_path, True, 's3'
    except Exception:
        try:
            urllib.request.urlretrieve(datum['openimages_metadata']['OriginalURL'], local_path)
            return local_path, True, 'url'
        except Exception as e:
            return local_path, False, str(e)

def main(args):
    with gzip.open(os.path.join(args.data_path, 'info.json.gz'), 'rb') as f:
        data = json.loads(f.read())

    # pre-create all subdirectories before spawning threads
    subsets = {datum['openimages_metadata']['Subset'] for datum in data}
    for subset in subsets:
        os.makedirs(os.path.join(args.download_folder, subset), exist_ok=True)

    print(f'Downloading {len(data)} images to {args.download_folder} '
          f'with {args.num_workers} workers...')

    init_worker()  # threads share process, so init once here

    worker = partial(download_one_image, download_folder=args.download_folder)
    failures = []

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(worker, datum): datum for datum in data}
        with tqdm(total=len(data), desc='Downloading') as pbar:
            for future in as_completed(futures):
                path, ok, detail = future.result()
                if not ok:
                    failures.append((path, detail))
                pbar.update(1)

    print(f'Done. {len(data) - len(failures)} succeeded, {len(failures)} failed.')
    if failures:
        fail_log = os.path.join(args.download_folder, 'failed_downloads.json')
        with open(fail_log, 'w') as f:
            json.dump(failures, f, indent=2)
        print(f'Failed paths written to {fail_log}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--download_folder', default='./DMS_v1')
    parser.add_argument('--num_workers', type=int, default=32)
    args = parser.parse_args()
    main(args)