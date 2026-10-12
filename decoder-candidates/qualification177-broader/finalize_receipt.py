from pathlib import Path
import hashlib,json

HERE=Path(__file__).resolve().parent
OUT=HERE.parent/'deliverables177'
rows=json.loads((OUT/'receipt_inputs.json').read_text())
for row in rows:
    p=OUT/row['name']
    assert p.read_bytes()==Path(str(p)+'.readback').read_bytes()
    row.update(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),raw_readback_exact=True)
receipt={'schema':'IG_DELIVERY_RECEIPT_V1','artifacts':rows,
 'scope':json.loads((HERE/'RESULT_SCOPE.json').read_text()),
 'cold':json.loads((HERE/'capture177/results/FINAL_COLD.json').read_text()),
 'preservation':json.loads((HERE/'capture177/results/FINAL_PRESERVED.json').read_text()),
 'github':json.loads((HERE/'GITHUB_PUBLICATION.json').read_text())}
(OUT/'IG_08LIB_QUALIFICATION177_DELIVERY_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('Report and handoff raw readbacks exact')
