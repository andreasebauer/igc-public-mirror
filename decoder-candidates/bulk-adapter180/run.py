def main():
    from pathlib import Path
    import sys,json,os
    HERE=Path(__file__).resolve().parent
    p=json.loads((HERE/'results/CAPTURE_PENDING.json').read_text())
    sys.path.insert(0,p.get('resume_workspace',p['workspace'])+'/source')
    from infinity_grid import v05_controller_event_loop as loop
    mode=sys.argv[1]
    if mode=='fault':
     os.environ['IG_V05_ENGINEERING_FAULT_INJECTION_ACK']='STAGE_RUNTIME_ABORT_TEST'
     os.environ['IG_V05_ENGINEERING_ABORT_AFTER_COMMITS']='5'
    try:
     result=loop.run_workspace_job(p.get('resume_workspace',p['workspace']),p['capture']['job_id'])
    except Exception as exc:
     (HERE/'results'/('NATIVE_'+mode.upper()+'.json')).write_text(json.dumps({'status':'PAUSED','type':type(exc).__name__,'reason':str(exc)},indent=2)+'\n')
     if mode!='fault' or 'ENGINEERING_FAULT_INJECTION_ABORT_AFTER_5_COMMITS' not in str(exc):raise
     print('EXPECTED_FAULT',str(exc))
    else:
     (HERE/'results'/('NATIVE_'+mode.upper()+'.json')).write_text(json.dumps(result,indent=2)+'\n')
     print(json.dumps({'status':result['status'],'reused':result.get('reused'),'source_sha256':result['source_sha256']}))

if __name__ == '__main__':
    main()
