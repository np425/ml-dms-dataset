import argparse
import json
import gzip
import os
import boto3
import botocore
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from functools import partial

# Configuration
BUCKET_NAME = 'open-images-dataset'

def download_one_image(datum, download_folder):
    """Downloads a single image into the correct subdirectory."""
    # Extract path components from the dataset metadata
    # image_path usually looks like: 'train/a1b2c3d4.jpg'
    relative_path = datum['image_path']
    local_path = os.path.join(download_folder, relative_path)
    
    # Skip if already exists
    if os.path.exists(local_path):
        return True

    # Ensure the subdirectory exists (e.g., 'train/')
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    # Metadata for S3
    subset = datum['openimages_metadata']['Subset']
    image_id = datum['openimages_metadata']['ImageID']
    s3_key = f'{subset}/{image_id}.jpg'
    
    # Download logic
    try:
        s3 = boto3.resource('s3', config=botocore.config.Config(signature_version=botocore.UNSIGNED))
        s3.Bucket(BUCKET_NAME).download_file(s3_key, local_path)
        return True
    except Exception:
        # Fallback to direct URL if S3 fails
        try:
            import urllib.request
            urllib.request.urlretrieve(datum['openimages_metadata']['OriginalURL'], local_path)
            return True
        except Exception:
            return False

def main(args):
    # Load metadata
    info_path = os.path.join(args.data_path, 'info.json.gz')
    with gzip.open(info_path, 'rb') as f:
        data = json.loads(f.read())
    
    print(f"Starting download of {len(data)} images to {args.download_folder}...")
    
    # Parallel download
    worker = partial(download_one_image, download_folder=args.download_folder)
    with ProcessPoolExecutor(max_workers=args.num_processes) as executor:
        results = list(tqdm(executor.map(worker, data), total=len(data), desc="Downloading"))
        
    print(f"Successfully downloaded {sum(results)} images.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True, help='Path containing info.json.gz')
    parser.add_argument('--download_folder', default='./DMS_v1', help='Folder to store images')
    parser.add_argument('--num_processes', type=int, default=10, help='Parallel processes')
    args = parser.parse_args()
    main(args)