from pathlib import Path
import hashlib,json,ast,shutil
D=Path(__file__).parent;old=D.parent/'v55_gate'
shutil.copy2(old/'lifecycle_monitor.py',D/'lifecycle_monitor.py')
text=(old/'prepare_capture.py').read_text().replace('RC.CORRUPTION.V55.DEV142','RC.INTERRUPT.V61.DEV143').replace("'/tmp/ig_gate_v55_20260930/store'","'/tmp/ig_gate_v61_20261001/store'").replace('DEV142','DEV143').replace('V55 native isolated live-corruption observation gate','V61 native operator-interrupt cleanup gate').replace('V55 dev142 save wave targeted gate','V61 dev143 native operator interrupt').replace("{'validation_wave_selectors':1}","{}")
(D/'prepare_capture.py').write_text(text)
for p in D.glob('*.py'):ast.parse(p.read_text())
hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(D.glob('*.py'))}
(D/'OPERATOR_SCRIPT_HASHES.json').write_text(json.dumps(hashes,indent=2)+'\n')
