"""Read the sealed, qualified historical G1 source; never construct candidates."""
from pathlib import Path
import copy,gzip,hashlib,json,zipfile
def canonical_hash(x):
    # JSON-parsed values are already in the canonicalizer's normalized domain.
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def read_bound(path,digest):
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('INPUT_IDENTITY')
    return json.loads(raw)
class ExactParentReader:
    def __init__(self,path,archive_sha256,scientific_root):
        raw=Path(path).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=archive_sha256:raise ValueError('ARCHIVE_IDENTITY')
        with zipfile.ZipFile(path) as z:
            if z.testzip() is not None:raise ValueError('ZIP_CRC')
            m=json.loads(z.read('SEAL_MANIFEST.json'));p=m['payload']
            if canonical_hash(p)!=scientific_root or m['scientific_root_sha256']!=scientific_root:raise ValueError('SEAL_IDENTITY')
            if set(z.namelist())!=set(p['files'])|{'SEAL_MANIFEST.json'}:raise ValueError('FILE_SET')
            for n,v in p['files'].items():
                b=z.read(n)
                if len(b)!=v['bytes'] or hashlib.sha256(b).hexdigest()!=v['sha256']:raise ValueError('SEALED_FILE_IDENTITY')
            self.proof=json.loads(z.read('AUDIT.json'));self.recovery=json.loads(z.read('SOURCE_INPUT_RECOVERY.json'))
            raw=gzip.decompress(z.read('BOOTSTRAP100_HISTORICAL.json.gz'))
            if hashlib.sha256(raw).hexdigest()!=self.recovery['bootstrap_raw_sha256']:raise ValueError('RAW_DAG_INPUT')
            self.input=json.loads(raw);self.dag=self.input['dag']
            self.v1=json.loads(z.read('HISTORICAL_V1_INTERFACE_POPULATION.json'));self.v2=json.loads(z.read('DIAGNOSTIC_V2_POPULATION.json'))
        if canonical_hash({k:v for k,v in self.dag.items() if k!='science_sha256'})!=p['dag_science_sha256'] or self.dag['science_sha256']!=p['dag_science_sha256']:raise ValueError('DAG_SCIENCE')
        self.nodes=self.dag['nodes'];self.roots=tuple(self.dag['roots']);self.rows={x['carrier_ref']:x for x in self.v1['interfaces']}
        if len(self.roots)!=193 or len(set(self.roots))!=193 or len(self.nodes)!=16528 or set(self.rows)!=set(self.roots):raise ValueError('CENSUS')
        self.closed=False
    def require_open(self):
        if self.closed:raise ValueError('CLOSED_READER')
    def root_refs(self):self.require_open();return self.roots
    def exact_node(self,key):
        self.require_open()
        if key not in self.nodes:raise ValueError('UNKNOWN_NODE')
        return copy.deepcopy(self.nodes[key])
    def historical_interface(self,key):
        self.require_open()
        if key not in self.rows:raise ValueError('UNKNOWN_CARRIER')
        return copy.deepcopy(self.rows[key])
    def close(self):self.closed=True
def qualify(inputs,bindings):
    v={k:read_bound(inputs[k],bindings[k]) for k in bindings if k!='seal'}
    old,cat,ad,proposal,gate=v['previous_catalog'],v['candidate_catalog'],v['scoped_admission'],v['proposal'],v['admission_gate']
    assert old['release_id']=='MASTER_DATA_V1_0151' and len(old['slices'])==151
    assert cat['release_id']=='MASTER_DATA_V1_0152' and len(cat['slices'])==152 and cat['slices'][:-1]==old['slices']
    assert cat['previous_catalog_sha256']==bindings['previous_catalog']==ad['predecessor_catalog_sha256']==proposal['base_catalog_sha256']
    assert cat['slices'][-1]['scoped_admission_sha256']==bindings['scoped_admission'] and cat['slices'][-1]['scope']==ad['scope']==proposal['proposed_slice']['scope']
    assert gate['status']=='PASS_PRIOR_NATIVE_COLD_AND_SEALED_EXPORT_GATES' and ad['admission_gate_sha256']==bindings['admission_gate']
    assert ad['generation_calls']==0 and ad['scope']['Q2_payload_available'] is False and ad['scope']['G2_promotion'] is False
    r=ExactParentReader(inputs['seal'],bindings['seal'],ad['scientific_root_sha256'])
    assert r.proof['all193_DAG_roundtrip'] and r.proof['native_exact_state_identity'] and r.proof['current_v2_population_reproduced']
    reached=set();active=set()
    def visit(k):
        if k in reached:return
        assert k not in active and r.nodes[k]['construction_digest']==k;active.add(k)
        for child in r.nodes[k].get('children',[]):visit(child)
        active.remove(k);reached.add(k)
    for k in r.root_refs():
        assert r.exact_node(k)==r.nodes[k] and r.historical_interface(k)==r.rows[k];visit(k)
    assert reached==set(r.nodes)
    assert set(x['carrier_ref'] for x in r.v2['interfaces'])==set(r.roots)
    assert len({x['interface_sha256'] for x in r.v1['interfaces']})==192 and sum(len(x['one_endpoint_reservations']) for x in r.v1['interfaces'])==1351
    for call in [lambda:r.exact_node('missing'),lambda:r.historical_interface('missing')]:
        try:call()
        except ValueError:pass
        else:raise ValueError('MISSING_ROUTE_ACCEPTED')
    r.close()
    for call in [r.root_refs,lambda:r.exact_node(r.roots[0]),lambda:r.historical_interface(r.roots[0])]:
        try:call()
        except ValueError:pass
        else:raise ValueError('CLOSED_ROUTE_ACCEPTED')
    return dict(outcome='PASS',master_release='MASTER_DATA_V1_0152',scientific_slices=152,prior_slices_unchanged=151,exact_parent_reader_qualified=True,scoped_admission_verified=True,carrier_routes_checked=193,DAG_nodes_checked=16528,interface_classes=192,reservation_rows=1351,generation_calls=0,new_DAG_decodes=0,Q2_payload_available=False,G2_promotion=False,master_cursor_updated=False)
