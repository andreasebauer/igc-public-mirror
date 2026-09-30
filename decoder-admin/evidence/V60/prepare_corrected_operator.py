from pathlib import Path
import shutil,json
D=Path(__file__).parent;B=D/'native_b';B.mkdir()
text=(D/'native_ops.py').read_text()
assert "'prerequisites':[]}}" in text
text=text.replace("'prerequisites':[]}}","'prerequisites':[],'preservation':{}}}")
text=text.replace('RC.CLEANUP.V60.DEV143','RC.CLEANUP.V60B.DEV143').replace('/tmp/ig_gate_v60_20261001','/tmp/ig_gate_v60b_20261001').replace('/tmp/ig_gate_v60_restored_20261001','/tmp/ig_gate_v60b_restored_20261001')
(B/'native_ops.py').write_text(text)
shutil.copy2(D/'RESULT.json',B/'RESULT.json')
(B/'NATIVE_PROTOCOL.txt').write_text((D/'NATIVE_PROTOCOL.txt').read_text().replace('RC.CLEANUP.V60.DEV143','RC.CLEANUP.V60B.DEV143')+'\nV60B corrects missing preservation:{} in the rejected pre-admission V60\ncontract. V60 dispatched zero tests; all original files/store remain intact.\n')
(D/'ADMISSION_REFUSAL.json').write_text(json.dumps({'error':'RESULT_CONTRACT_FIELDS','cause':'Missing required preservation field','capture_created':False,'tests_dispatched':0,'original_operator':'native_ops.py','corrected_operator':'native_b/native_ops.py','engine_changed':False},indent=2)+'\n')
