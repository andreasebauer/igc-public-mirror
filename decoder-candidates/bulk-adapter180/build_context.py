from pathlib import Path
import importlib.util,sys,json,hashlib
root=Path(__file__).resolve().parent
fixture=root/'inputs/fixture/Infinity_Grid_O7_COMPACT_REPLAY_ROOT_v1_2026-08-29'
p=fixture/'02_CODE/baseline_engine.py'
assert hashlib.sha256(p.read_bytes()).hexdigest()=='6847a47f68932bdfbfce7bbc4b12442cf7a262bc620511db67110ecebc919868'
s=importlib.util.spec_from_file_location('historical_context',p);m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);m.O6=m.import_o6()
ctx,_=m.lane_contexts()['HET4']
value={'schema':'IG_HET4_FROZEN_CONTEXT_V1','parents':[{'exact_color':p.exact_color,'h6':p.h6} for p in ctx.parents],'source_order':ctx.source_order,'baseline_engine_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
(root/'project/context.json').write_text(json.dumps(value,sort_keys=True,separators=(',',':'))+'\n')
print('parents',len(ctx.parents),'context bytes',(root/'project/context.json').stat().st_size)
