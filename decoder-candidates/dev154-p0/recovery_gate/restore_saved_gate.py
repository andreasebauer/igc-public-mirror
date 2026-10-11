"""Restore exact saved slim closure from locally materialized Drive dependencies."""
import argparse
from pathlib import Path
import sys

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--candidate-source',required=True,type=Path)
    parser.add_argument('--slim',required=True,type=Path)
    parser.add_argument('--objects',required=True,type=Path)
    parser.add_argument('--destination',required=True,type=Path)
    args=parser.parse_args()
    sys.path.insert(0,str(args.candidate_source))
    from infinity_grid import preservation
    result=preservation.restore_checkpoint(args.slim,args.destination,
        'f486875dca0772cd6802698d29ef8466977e49b64fc23e579eee5cc08dc53779',objects=args.objects)
    if result['checkpoint_sha256']!='b90924d17afd7847a184fa27b0140fd5dcbeb65ed4d0b4aae43f99f44e8c25ad':
        raise RuntimeError('SAVED_GATE_CHECKPOINT_MISMATCH')
    import json
    print(json.dumps(result))

if __name__=='__main__':main()
