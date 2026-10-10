"""Synthetic serializer tests only; no Decoder imports or scientific executions."""
import ast,hashlib,importlib.util,json,pathlib,unittest
B=pathlib.Path(__file__).parent
s=importlib.util.spec_from_file_location('observer',B/'project/observer.py');o=importlib.util.module_from_spec(s);s.loader.exec_module(o)
class State:
    def __init__(self,caps=(1,)*7,skin='skin-ä',edges=((1,0,2,3),(0,1,3,2),(0,1,3,2))):
        self.total_caps=caps;self.skin=skin;self.typed_edges=edges;self.calls=[]
    def reserve_external(self,t):
        assert self.total_caps[t]>0;self.calls.append(t);caps=list(self.total_caps);caps[t]-=1
        return State(tuple(caps),'next-'+str(t),()),'excluded-witness'
class Tests(unittest.TestCase):
    def test_available(self):
        st=State();p=o.public_assembly(st);self.assertEqual(st.calls,list(range(7)))
        self.assertEqual(p['typed_top_edge_multiset'],[[0,1,3,2],[0,1,3,2],[1,0,2,3]])
        for t,r in enumerate(p['one_endpoint_reservations']):
            self.assertTrue(r['available']);self.assertEqual(r['successor_total_caps'][t],0)
    def test_unavailable(self):
        st=State((0,)*7);p=o.public_assembly(st);self.assertEqual(st.calls,[])
        self.assertTrue(all(r['successor_skin'] is None and r['successor_total_caps'] is None for r in p['one_endpoint_reservations']))
    def test_exact_hash_protocol(self):
        p=o.public_assembly(State());raw=json.dumps(p,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
        self.assertEqual(o.observer_sha256(p),hashlib.sha256(raw).hexdigest())
        self.assertNotEqual(o.observer_sha256(p),hashlib.sha256(raw+b'\n').hexdigest())
    def test_field_firewall(self):
        p=o.public_assembly(State());self.assertEqual(set(p),{'schema_id','assembly_skin','total_caps','typed_top_edge_multiset','one_endpoint_reservations'})
        self.assertNotIn('excluded-witness',json.dumps(p))
    def test_registered_census(self):
        r=json.loads((B/'PREREGISTRATION.json').read_text());self.assertEqual(len(r['specifications']),114)
        self.assertEqual(sum(x['case_count'] for x in r['specifications']),7058)
        self.assertEqual(sum(x['class_count'] for x in r['specifications']),3523)
        for row in r['specifications']:
            p=B/row['spec_path'];self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),row['spec_sha256'])
            spec=json.loads(p.read_text());e=next(i for i in spec['inputs'] if i['logical_name']=='observer_expected')
            self.assertEqual(hashlib.sha256(pathlib.Path(e['path']).read_bytes()).hexdigest(),e['sha256'])
        self.assertFalse(json.loads((B/'PREFLIGHT.json').read_text())['native_execution_permitted'])
if __name__=='__main__':unittest.main(verbosity=2)
