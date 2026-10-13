"""Read-only split-source identity check. Never dispatches a job."""
from pathlib import Path
import argparse
import json
import sys
import tempfile


def verify(engine, project, pins):
    if sys.flags.optimize:
        raise ValueError('OPTIMIZED_RUNTIME')
    engine=Path(engine).resolve(strict=True)
    project=Path(project).resolve(strict=True)
    sys.path.insert(0,str(engine))
    from infinity_grid import __version__,build_meta
    from infinity_grid import submission as sub
    from infinity_grid.isolated_runtime import verified_runtime,BUNDLED_MANIFEST_SHA256
    from infinity_grid.v05_controller_event_loop import _source_ids
    from infinity_grid.records import source_sha256
    verified_runtime()
    if Path(sys.modules['infinity_grid'].__file__).resolve().parent!=engine/'infinity_grid':
        raise ValueError('LOADED_ENGINE_ORIGIN')
    files=sub._tree(engine)
    if not any(n.startswith('infinity_grid/') for n in files) or any(n.startswith('project/') for n in files):
        raise ValueError('ENGINE_SOURCE_LAYOUT')
    project_files=sub._tree(project)
    if 'bulk_adapter.py' not in project_files or any(n.startswith('project/') for n in project_files):
        raise ValueError('PROJECT_SOURCE_LAYOUT')
    engine_sha,package_sha=_source_ids(engine)
    with tempfile.TemporaryDirectory(prefix='ig-candidate-pin-') as temporary:
        composite=Path(temporary)
        sub._write_files(composite,files)
        sub._write_files(composite/'project',project_files)
        combined_sha,combined_package_sha=_source_ids(composite)
    observed={'version':__version__,'engine_source_sha256':engine_sha,
              'bulk_combined_source_sha256':combined_sha,'package_sha256':package_sha,
              'records_source_sha256':source_sha256(),
              'runtime_manifest_sha256':BUNDLED_MANIFEST_SHA256}
    if build_meta()['version']!=__version__ or combined_package_sha!=package_sha:
        raise ValueError('PACKAGE_VERSION_BINDING')
    for name,value in observed.items():
        if pins.get(name)!=value:
            raise ValueError('CANDIDATE_PIN_MISMATCH:'+name)
    return dict(observed,status='PASS',scientific_adoption=False,science_dispatched=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine',type=Path,required=True)
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--pins',type=Path,required=True)
    args=parser.parse_args()
    try:
        result=verify(args.engine,args.project,json.loads(args.pins.read_bytes()))
    except Exception as exc:
        print(json.dumps({'status':'REFUSED','reason':str(exc),'science_dispatched':False}))
        return 1
    print(json.dumps(result,sort_keys=True))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
