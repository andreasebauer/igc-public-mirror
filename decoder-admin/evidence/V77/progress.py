from pathlib import Path
import json,sys
D=Path(__file__).resolve().parent;G=D/sys.argv[1];s=json.loads((G/'CAPTURE_SAVE_STATUS.json').read_text());w=Path(s['workspace']);rows=[json.loads(p.read_text()) for p in (w/'runtime/runs').rglob('nodes/*.json')];cols=[json.loads(p.read_text()) for p in (w/'runtime/runs').rglob('collections/*.json')]
passed=[x for x in rows if x.get('finished') and set(x['phases'])=={'setup','call','teardown'} and all(v['outcome']=='passed' and not v.get('xfail') for v in x['phases'].values())];bad=[x for x in rows if any(v['outcome']!='passed' or v.get('xfail') for v in x['phases'].values())]
print(json.dumps({'node_reports':len(rows),'passed':len(passed),'nonpass':len(bad),'unfinished':sum(not x.get('finished') for x in rows),'collection_receipts':len(cols),'collected_nodes':sum(len(c['nodes']) for c in cols),'monitor_exit':json.loads((G/'MONITOR_EXIT.json').read_text()) if (G/'MONITOR_EXIT.json').exists() else None,'first_nonpass':bad[:1]}))
