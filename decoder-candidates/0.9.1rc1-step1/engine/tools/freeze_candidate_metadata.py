"""Explicit release preparation: freeze the records hash domain, not native pins."""
from pathlib import Path
import json
import sys


def main():
    source=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(source))
    from infinity_grid import __version__
    from infinity_grid.records import live_source_sha256
    from infinity_grid.canon import write_json_atomic
    path=source/'infinity_grid/_build_meta.json'
    meta=json.loads(path.read_bytes())
    meta.update(source_sha256=live_source_sha256(),target_release=__version__,
                release_id='decoder-'+__version__,parent_release='0.9.0',
                parent_source_sha256='496d7a0b69798d51f5735207978f0e1143eefb930cbe9f081f65637243f9f8d2',
                parent_package_sha256='f9636fef712db302da2b75e33c0c56bb358974119329af35c8b4a6f67a377596',
                engineering_attempt='DECODER091_STEP1_CONTRACT_REPAIR',
                status='STEP1_IMPLEMENTATION_CANDIDATE_NOT_RELEASE_QUALIFIED',
                semantic_delta='ADMISSION_AND_INTEGRITY_ONLY_NO_KERNEL_SCHEMA_CHANGE',
                publication_status='CANDIDATE_ONLY',shared_pointer_advanced=False,
                scope='Step 1 implementation; native release qualification and data adoption remain open')
    write_json_atomic(path,meta)
    print(json.dumps({'version':__version__,'records_source_sha256':meta['source_sha256'],
                      'release_qualified':False,'scientific_adoption':False}))


if __name__=='__main__':
    main()
