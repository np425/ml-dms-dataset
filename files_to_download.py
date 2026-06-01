import argparse
import gzip
import json
import os

def main(args):
    with gzip.open(os.path.join(args.data_path, 'info.json.gz'), 'rb') as f:
        data = json.loads(f.read())

    with open(args.output, 'w') as f:
        for datum in data:
            subset = datum['openimages_metadata']['Subset']
            image_id = datum['openimages_metadata']['ImageID']
            f.write(f'{subset}/{image_id}.jpg\n')

    print(f'Written {len(data)} entries to {args.output}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True, help='path containing info.json.gz')
    parser.add_argument('--output', default='files_to_download.txt', help='output file list path')
    args = parser.parse_args()
    main(args)