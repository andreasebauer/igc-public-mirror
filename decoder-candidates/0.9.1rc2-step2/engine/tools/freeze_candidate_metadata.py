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
                release_id='decoder-'+__version__,parent_release='0.9.1rc1',
                parent_source_sha256='f00d23b7ab0ae038e1414da1eae419f3a0f10e235f84f01512c42d289d8f0248',
                parent_package_sha256='0ad6d10e15e00d1c368f91bd7a1ecddc629e69d91d89c469ffadaf7d2f44131a',
                engineering_attempt='DECODER091_STEP2_DURABILITY_REPAIR',
                status='STEP2_IMPLEMENTATION_CANDIDATE_NOT_RELEASE_QUALIFIED',
                semantic_delta='ADMISSION_AND_INTEGRITY_ONLY_NO_KERNEL_SCHEMA_CHANGE',
                publication_status='CANDIDATE_ONLY',shared_pointer_advanced=False,
                scope='Step 2 implementation; native release qualification and data adoption remain open')
    write_json_atomic(path,meta)
    print(json.dumps({'version':__version__,'records_source_sha256':meta['source_sha256'],
                      'release_qualified':False,'scientific_adoption':False}))


if __name__=='__main__':
    main()
