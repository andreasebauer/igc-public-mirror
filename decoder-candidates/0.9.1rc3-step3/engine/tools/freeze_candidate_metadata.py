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
                release_id='decoder-'+__version__,parent_release='0.9.1rc2',
                parent_source_sha256='8b99be7070ab2a7fc23e3a2459187ab7d84bb4ddefc9f94b946043c39f33320d',
                parent_package_sha256='91c729545c500381ab32414d700c68207088a542e073420ddcaf22dce67b95cd',
                engineering_attempt='DECODER091_STEP3_RESOURCE_PERFORMANCE_REPAIR',
                status='STEP3_IMPLEMENTATION_CANDIDATE_NOT_RELEASE_QUALIFIED',
                semantic_delta='EXACT_BYTE_CONTEXT_INDEX_AND_RESOURCE_REPAIR_NO_IDENTITY_SCHEMA_CHANGE',
                publication_status='CANDIDATE_ONLY',shared_pointer_advanced=False,
                scope='Step 3 implementation; native release qualification and data adoption remain open')
    write_json_atomic(path,meta)
    print(json.dumps({'version':__version__,'records_source_sha256':meta['source_sha256'],
                      'release_qualified':False,'scientific_adoption':False}))


if __name__=='__main__':
    main()
