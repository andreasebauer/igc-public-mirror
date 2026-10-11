"""Restore the preserved dev155 data with the pinned dev156 administrative code."""
import argparse
import hashlib
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
for name in ['candidate-source', 'slim', 'objects', 'destination']:
    parser.add_argument('--' + name, required=True)
args = parser.parse_args()
source = Path(args.candidate_source).resolve(strict=True)
sys.path.insert(0, str(source))
from infinity_grid import preservation as pr
from infinity_grid.v05_controller_event_loop import _source_ids
if _source_ids(source)[0] != '1d51905e3943bfde75c84fd44f04e34ae5b56db9d370726ab2aa07e01e5047ad':
    raise RuntimeError('RESTORER_SOURCE_PIN')
slim = Path(args.slim)
import json
import zipfile
with zipfile.ZipFile(slim) as z:
    packet = json.loads(z.read('CHECKPOINT.json'))
if packet['checkpoint_sha256'] != 'c1c73c931918ab96fa87f6c5c76c80afa7b8cf234842360a314fa94ea114b671':
    raise RuntimeError('SAVED_CHECKPOINT_PIN')
result = pr.restore_checkpoint(slim, args.destination,
    hashlib.sha256(slim.read_bytes()).hexdigest(), objects=args.objects)
print(json.dumps(result, sort_keys=True))
