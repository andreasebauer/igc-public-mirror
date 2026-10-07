"""Read finite saved projections only. No exact state materialization."""
from pathlib import Path
from copy import deepcopy
import collections,hashlib,json

def digest(b):return hashlib.sha256(b).hexdigest()
def semantic(d):return digest(json.dumps(d,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode())

class ProjectionReader:
 def __init__(self,population,continuations,pins):
  self.closed=False;raw={}
  for name,path in [('population',population),('continuations',continuations)]:
   b=Path(path).read_bytes()
   if digest(b)!=pins[name]['sha256'] or len(b)!=pins[name]['bytes']:raise ValueError('SOURCE_HASH:'+name)
   raw[name]=b
  pop=json.loads(raw['population']);cont=json.loads(raw['continuations']);rows=pop['interfaces']
  refs=[r['carrier_ref'] for r in rows]
  if len(rows)!=193 or refs!=sorted(set(refs)) or semantic(refs)!=pop['carrier_ref_set_sha256']:raise ValueError('FINITE_COHORT')
  if semantic({k:v for k,v in pop.items() if k!='science_sha256'})!=pop['science_sha256']:raise ValueError('POPULATION_SEMANTIC_HASH')
  for r in rows:
   caps=r['total_free_by_type'];res=r['one_endpoint_reservations']
   if len(caps)!=7 or len(res)!=7 or any(type(x)!=int or x<=0 for x in caps):raise ValueError('CAPACITIES')
   for t,x in enumerate(res):
    if type(x['endpoint_type'])!=int or x['endpoint_type']!=t or x['available'] is not True or x['successor_total_free_by_type']!=[v-(j==t) for j,v in enumerate(caps)]:raise ValueError('RESERVATION_PROJECTION')
   payload={'schema_id':'IG_G1_PUBLIC_ONE_ENDPOINT_INTERFACE_SEMANTICS_V1','boundary_resource_skin_sha256':r['boundary_resource_skin_sha256'],'total_free_by_type':caps,'one_endpoint_reservations':res,'scope':'ONE_EXTERNAL_ENDPOINT_RESERVATION_FOR_WHOLE_CARRIER_PAIR_CONNECTION'}
   if semantic(payload)!=r['interface_sha256']:raise ValueError('INTERFACE_SEMANTICS')
  counts=collections.Counter(r['interface_sha256'] for r in rows)
  if len(counts)!=192 or sorted(counts.values())!=[1]*191+[2]:raise ValueError('INTERFACE_CLASSES')
  if pop['interface_class_histogram']!=[{'interface_sha256':k,'carrier_count':counts[k]} for k in sorted(counts)]:raise ValueError('CLASS_HISTOGRAM')
  if set(cont['rows'])!=set(refs) or any(set(x)!=set(map(str,range(7))) for x in cont['rows'].values()):raise ValueError('CONTINUATION_SCOPE')
  self._rows={r['carrier_ref']:r for r in rows};self._continuations=cont['rows'];self._classes={h:sorted(r['carrier_ref'] for r in rows if r['interface_sha256']==h) for h in counts}
  self._provenance={'authority':'SAVED_PUBLIC_PROJECTION_ONLY','population_sha256':digest(raw['population']),'population_science_sha256':pop['science_sha256'],'source_authority_sha256':pop['source_authority_sha256'],'continuations_sha256':digest(raw['continuations']),'exact_parent_DAG_available':False,'Q2_payload_available':False,'fresh_realization':False,'generation_calls':0}
 def _open(self):
  if self.closed:raise ValueError('CLOSED_READER')
 def refs(self):self._open();return sorted(self._rows)
 def interface(self,ref):self._open();return deepcopy(self._rows[ref])
 def reservation(self,ref,t):
  self._open()
  if type(t)!=int or not 0<=t<7:raise ValueError('ENDPOINT_TYPE')
  return deepcopy(self._rows[ref]['one_endpoint_reservations'][t])
 def continuation_hash(self,ref,t):
  self._open()
  if type(t)!=int or not 0<=t<7:raise ValueError('ENDPOINT_TYPE')
  return self._continuations[ref][str(t)]
 def interface_class(self,h):self._open();return list(self._classes[h])
 def provenance(self):self._open();return deepcopy(self._provenance)
 def exact_state(self,ref):self._open();raise ValueError('EXACT_PARENT_DAG_UNAVAILABLE')
 def q2_payload(self,ref):self._open();raise ValueError('Q2_PAYLOAD_UNAVAILABLE')
 def close(self):self.closed=True
