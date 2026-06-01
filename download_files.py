#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#
# This code accompanies the research paper: Upchurch, Paul, and Ransen
# Niu. "A Dense Material Segmentation Dataset for Indoor and Outdoor
# Scene Parsing." ECCV 2022.
#
# Async image downloader using aiohttp + aioboto3.
# Install deps: pip install aioboto3 aiohttp tqdm
#

import argparse
import asyncio
import gzip
import json
import os

import aioboto3
import aiohttp
import botocore
from tqdm.asyncio import tqdm

BUCKET_NAME = 'open-images-dataset'
BOTOCORE_CONFIG = botocore.config.Config(
    signature_version=botocore.UNSIGNED,
    connect_timeout=5,
    read_timeout=30,
    retries={'max_attempts': 3},
)


async def download_s3(s3_bucket, s3_key, local_path):
    await s3_bucket.download_file(s3_key, local_path)


async def download_url(session, url, local_path, chunk_size=65536):
    async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as response:
        response.raise_for_status()
        with open(local_path, 'wb') as f:
            async for chunk in response.content.iter_chunked(chunk_size):
                f.write(chunk)


async def download_one(datum, download_folder, s3_bucket, http_session, semaphore, retries=3):
    local_path = os.path.join(download_folder, datum['image_path'])

    if os.path.exists(local_path):
        return local_path, True, 'skipped'

    subset = datum['openimages_metadata']['Subset']
    image_id = datum['openimages_metadata']['ImageID']
    s3_key = f'{subset}/{image_id}.jpg'
    url = datum['openimages_metadata']['OriginalURL']

    async with semaphore:
        # try S3 first with retries
        last_exc = None
        for attempt in range(retries):
            try:
                await download_s3(s3_bucket, s3_key, local_path)
                return local_path, True, 's3'
            except Exception as e:
                last_exc = e
                if attempt < retries - 1:
                    await asyncio.sleep(2 ** attempt)

        # fallback to direct URL with retries
        for attempt in range(retries):
            try:
                await download_url(http_session, url, local_path)
                return local_path, True, 'url'
            except Exception as e:
                last_exc = e
                if attempt < retries - 1:
                    await asyncio.sleep(2 ** attempt)

    return local_path, False, str(last_exc)


async def run(args):
    with gzip.open(os.path.join(args.data_path, 'info.json.gz'), 'rb') as f:
        data = json.loads(f.read())

    print(f'Loaded {len(data)} entries.')

    if args.missing_only:
        data = [d for d in data if not os.path.exists(os.path.join(args.download_folder, d['image_path']))]
        print(f'{len(data)} images missing locally, downloading those only.')

    # pre-create subdirectories
    subsets = {d['openimages_metadata']['Subset'] for d in data}
    for subset in subsets:
        os.makedirs(os.path.join(args.download_folder, subset), exist_ok=True)

    # semaphore caps concurrent downloads
    semaphore = asyncio.Semaphore(args.concurrency)
    failures = []

    async with aioboto3.Session().resource(
        's3', config=BOTOCORE_CONFIG
    ) as s3:
        bucket = await s3.Bucket(BUCKET_NAME)

        connector = aiohttp.TCPConnector(limit=args.concurrency)
        async with aiohttp.ClientSession(connector=connector) as http_session:
            tasks = [
                download_one(datum, args.download_folder, bucket, http_session, semaphore)
                for datum in data
            ]
            results = await tqdm.gather(*tasks, desc='Downloading', unit='img')

    for path, ok, detail in results:
        if not ok:
            failures.append({'path': path, 'error': detail})

    succeeded = len(data) - len(failures)
    print(f'Done. {succeeded} succeeded, {len(failures)} failed.')

    if failures:
        fail_log = os.path.join(args.download_folder, 'failed_downloads.json')
        with open(fail_log, 'w') as f:
            json.dump(failures, f, indent=2)
        print(f'Failures written to {fail_log}. Re-run with --retry-failed to retry.')


async def run_retry(args):
    fail_log = os.path.join(args.download_folder, 'failed_downloads.json')
    if not os.path.exists(fail_log):
        print(f'No failure log found at {fail_log}.')
        return

    with open(fail_log) as f:
        failures = json.load(f)

    # rebuild minimal datums from failure log paths for re-download
    # requires re-loading info.json.gz to get full metadata
    with gzip.open(os.path.join(args.data_path, 'info.json.gz'), 'rb') as f:
        data = json.loads(f.read())

    failed_paths = {entry['path'] for entry in failures}
    retry_data = [
        d for d in data
        if os.path.join(args.download_folder, d['image_path']) in failed_paths
    ]
    print(f'Retrying {len(retry_data)} previously failed downloads...')

    # temporarily replace data and re-run
    args._retry_data = retry_data
    await _run_with_data(retry_data, args)


async def _run_with_data(data, args):
    semaphore = asyncio.Semaphore(args.concurrency)
    failures = []

    async with aioboto3.Session().resource('s3', config=BOTOCORE_CONFIG) as s3:
        bucket = await s3.Bucket(BUCKET_NAME)
        connector = aiohttp.TCPConnector(limit=args.concurrency)
        async with aiohttp.ClientSession(connector=connector) as http_session:
            tasks = [
                download_one(datum, args.download_folder, bucket, http_session, semaphore)
                for datum in data
            ]
            results = await tqdm.gather(*tasks, desc='Retrying', unit='img')

    for path, ok, detail in results:
        if not ok:
            failures.append({'path': path, 'error': detail})

    print(f'Retry done. {len(data) - len(failures)} succeeded, {len(failures)} still failing.')
    if failures:
        fail_log = os.path.join(args.download_folder, 'failed_downloads.json')
        with open(fail_log, 'w') as f:
            json.dump(failures, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True,
                        help='path containing info.json.gz')
    parser.add_argument('--download_folder', default='./DMS_v1',
                        help='folder to store downloaded images')
    parser.add_argument('--concurrency', type=int, default=64,
                        help='max simultaneous downloads (default: 64)')
    parser.add_argument('--retry-failed', action='store_true',
                        help='retry downloads listed in failed_downloads.json')
    parser.add_argument('--missing-only', action='store_true',
                        help='skip images already present on disk, download only missing ones')
    args = parser.parse_args()

    if args.retry_failed:
        asyncio.run(run_retry(args))
    else:
        asyncio.run(run(args))


if __name__ == '__main__':
    main()