"""Independent export audit of sealed bytes; reuses earned native qualification."""
from pathlib import Path
import gzip,hashlib,json,zipfile,sys
B=Path(__file__).resolve().parent
sys.path.insert(0,'/tmp/ig_engine0204')
from infinity_grid.canon import canonical_sha256
def main():
    receipt=json.loads((B/'SEAL_RECEIPT.json').read_text()); raw=Path(receipt['path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==receipt['sha256']
    with zipfile.ZipFile(receipt['path']) as z:
        assert z.testzip() is None
        m=json.loads(z.read('SEAL_MANIFEST.json'));p=m['payload'];assert canonical_sha256(p)==m['scientific_root_sha256']==receipt['scientific_root_sha256']
        assert set(z.namelist())==set(p['files'])|{'SEAL_MANIFEST.json'}
        for n,v in p['files'].items():
            b=z.read(n);assert len(b)==v['bytes'] and hashlib.sha256(b).hexdigest()==v['sha256']
        recovery=json.loads(z.read('SOURCE_INPUT_RECOVERY.json'));raw=gzip.decompress(z.read('BOOTSTRAP100_HISTORICAL.json.gz'));assert hashlib.sha256(raw).hexdigest()==recovery['bootstrap_raw_sha256']
        inp=json.loads(raw);dag=inp['dag'];nodes=dag['nodes'];roots=dag['roots'];assert len(nodes)==16528 and len(set(roots))==len(roots)==193
        assert canonical_sha256({k:v for k,v in dag.items() if k!='science_sha256'})==dag['science_sha256']==p['dag_science_sha256']
        reached=set();active=set()
        def visit(k):
            if k in reached:return
            assert k not in active and nodes[k]['construction_digest']==k;active.add(k)
            for c in nodes[k].get('children',[]):visit(c)
            active.remove(k);reached.add(k)
        for k in roots:visit(k)
        assert reached==set(nodes)
        v1=json.loads(z.read('HISTORICAL_V1_INTERFACE_POPULATION.json'));v2=json.loads(z.read('DIAGNOSTIC_V2_POPULATION.json'))
        for v in [v1,v2]:
            assert {x['carrier_ref'] for x in v['interfaces']}==set(roots)
            assert len(v['interfaces'])==193 and len({x['interface_sha256'] for x in v['interfaces']})==192
            assert sum(len(x['one_endpoint_reservations']) for x in v['interfaces'])==1351
        a=json.loads(z.read('AUDIT.json'));assert a['all193_DAG_roundtrip'] and a['current_v2_population_reproduced']
        n=json.loads(z.read('NATIVE_RESULT.json'));assert n['status']=='COMPLETED' and n['evidence_status']=='VERIFIED'
        assert json.loads(z.read('CHECKPOINT_PRESERVED.json'))['pending_bytes']==0
    proposal=json.loads((B/'SCOPED_ADMISSION_PROPOSAL.json').read_text());cat=B.parent/'continuation0231'/'CATALOG_0151.json'
    assert hashlib.sha256(cat.read_bytes()).hexdigest()==proposal['base_catalog_sha256'] and len(json.loads(cat.read_text())['slices'])==151
    result=dict(status='PASS_INDEPENDENT_SEALED_EXPORT_AUDIT',scientific_root_sha256=receipt['scientific_root_sha256'],carriers=193,DAG_nodes=16528,interface_classes=192,reservation_rows=1351,all_archive_bytes_verified=True,DAG_canonical_identity=True,all_nodes_reachable_acyclic=True,all_carrier_population_joins_exact=True,prior_native_and_cold_193_roundtrip_evidence_bound=True,new_DAG_decodes=0,candidate_generation=0,master_slices=151,new_admissions=0,G2_promotion=False)
    (B/'AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
