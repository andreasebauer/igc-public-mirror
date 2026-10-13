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
                release_id='decoder-'+__version__,parent_release='0.9.1rc3',
                parent_source_sha256='0413682caf717788cd74144bf17302868087212a15dee3e1ab9064d65a486b6a',
                parent_package_sha256='56b988da3c8ea563511fbe5b50f12d9e35dbdae32834608fe1528f9aec2be532',
                engineering_attempt='DECODER091_STEP4_NATIVE_RELEASE_QUALIFICATION',
                status='FROZEN_RELEASE_SOURCE_REQUIRES_EXACT_EXTERNAL_QUALIFICATION_LEDGER',
                semantic_delta='EXACT_BYTE_CONTEXT_INDEX_AND_RESOURCE_REPAIR_NO_IDENTITY_SCHEMA_CHANGE',
                publication_status='CANDIDATE_ONLY',shared_pointer_advanced=False,
                scope='Release source; exact native qualification ledger is authoritative; native release qualification and data adoption remain open')
    write_json_atomic(path,meta)
    print(json.dumps({'version':__version__,'records_source_sha256':meta['source_sha256'],
                      'release_qualified':False,'scientific_adoption':False}))


if __name__=='__main__':
    main()
