"""Copy verified saved repair evidence only; never execute source capsules."""
from pathlib import Path
import hashlib,io,json,zipfile
R=Path.cwd();B=R/'g_repair0177';I=B/'inputs';I.mkdir(exist_ok=True)
sha=lambda b:hashlib.sha256(b).hexdigest()
def dump(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
pins={}
for label,pattern,expected in [('d4','*MINIMAL*.zip','19714c1015a5cb5fd80d4ebf71925c6732f74ff3de32eb30f89a6951356bfbac'),('recon','*S1R*.zip','7551c4765e2ebfde65220548a9ffda81f327c464ba959795eb64ce3151819dc0'),('reaudit','*D4_Q2*.zip','8bb9cc6e53e77425fb75985f3681635cea75375d2aa227b97774ae740af9b97f')]:
 p=next((B/'recovered').rglob(pattern));assert sha(p.read_bytes())==expected
 pins[label]={'sha256':expected,'bytes':p.stat().st_size,'source_path':str(p.relative_to(R))}
 with zipfile.ZipFile(p) as z:
  for n in z.namelist():
   if n.endswith('.json') and 'attempt' not in n and not n.endswith('STATE.json'):
    target=I/label/n;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(n))
  if label=='recon':
   target=I/label/'evidence/RESIDUAL_FEATURE_ROWS.jsonl';target.write_bytes(z.read('evidence/RESIDUAL_FEATURE_ROWS.jsonl'))
  if label=='reaudit':
   n=next(n for n in z.namelist() if n.startswith('source/') and n.endswith('.zip'));raw=z.read(n)
   assert sha(raw)==json.loads(z.read('DEPENDENCIES.json'))['source_release']['sha256']
   with zipfile.ZipFile(io.BytesIO(raw)) as source:
    n=next(n for n in source.namelist() if n.endswith('/uplift_structural.py'));(I/'historical_uplift_structural.py').write_bytes(source.read(n))
(I/'uplift_structural.py').write_bytes((R/'engine/infinity_grid/uplift_structural.py').read_bytes())
(I/'CURRENT_IMPLEMENTATION_SPEC_V2.json').write_bytes((R/'engine/infinity_grid/resources/uplift/G_UPLIFT_S0_S1_IMPLEMENTATION_SPEC_V2.json').read_bytes())
(I/'G2_S0_INTERFACE_POPULATION.json').write_bytes((R/'g_closure0176/inputs/G2_S0_INTERFACE_POPULATION.json').read_bytes())
dump(B/'SOURCE_PINS.json',pins)
dump(B/'INPUT_PINS.json',{str(p.relative_to(I)):{'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size} for p in I.rglob('*') if p.is_file()})
print('All3 saved source archive hashes and embedded v0.29.2 source pin PASS')
