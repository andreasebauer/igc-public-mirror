"""Verify split launch layout and optionally capture one window; never run it."""
import argparse
import json
from pathlib import Path
from verify_candidate_source import verify


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine',type=Path,required=True)
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--pins',type=Path,required=True)
    parser.add_argument('--destination',type=Path)
    parser.add_argument('--raw',type=Path)
    parser.add_argument('--prefix',type=Path)
    parser.add_argument('--sqlite-binding',type=Path)
    parser.add_argument('--window-index',type=int)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    checked=verify(args.engine,args.project,json.loads(args.pins.read_bytes()))
    capture_args=[args.destination,args.raw,args.prefix,args.sqlite_binding,args.window_index]
    if any(x is not None for x in capture_args):
        if any(x is None for x in capture_args):
            parser.error('Window capture requires destination, raw, prefix, sqlite-binding and window-index together')
        from infinity_grid.bulk_submission import prepare_window
        result=prepare_window(args.destination,args.engine,args.project,args.raw,args.prefix,
                              args.sqlite_binding,args.window_index,workers=args.workers)
        result={'identities':checked,'capture':result['capture'],'dispatched':False,
                'scientific_adoption':False}
    else:
        result=checked
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':
    main()
