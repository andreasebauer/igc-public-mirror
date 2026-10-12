"""Fresh bounded rooted L2/J3 production from explicitly retained upstream tables.

The boundary has ordered external ports and ordered component destinations. This
identity makes no quotient under port/resource permutation and never applies the
legacy independently-sorted boundary/destination projection.
"""
from pathlib import Path
import hashlib,json
from .canon import canonical_sha256
from .core.l2 import j3,bridge_records
from .execution import TaskSpec
from .v05_chain import ChainExecutionResult

IDENTITY_VERSION='IG_ROOTED_L2_ABC_BOUNDARY_V1'
CLASSES=(3943,3963,3973)

class InputTables:
    def __init__(self,packet):
        if packet.get('schema_id')!='IG_STEP3_L2_TABLE_INPUT_V1' or tuple(packet['classes'])!=CLASSES:
            raise ValueError('STEP3_INPUT_SCOPE')
        for n in ('tri','cid','oc','ot','os','om','on','rk'):setattr(self,n,packet['arrays'][n])
        if len(self.tri)!=3*len(self.cid) or any(len(getattr(self,n))!=10*len(self.oc) for n in ('ot','os','om','on','rk')):
            raise ValueError('STEP3_TABLE_SHAPE')
        self.lookup={self.triple(i):i for i in range(len(self.cid))};self.rep={}
        for i,c in enumerate(self.cid):self.rep.setdefault(c,i)
        if [self.rep[c] for c in CLASSES]!=packet['representative_tids']:raise ValueError('STEP3_REPRESENTATIVE_BINDING')
    def triple(self,tid):return tuple(self.tri[tid*3:tid*3+3])
    @staticmethod
    def at(a,s,i):return a[s*10+i]

def event_id(event):return canonical_sha256({'schema':'IG_ROOTED_J3_EVENT_V1','event':event})

def evaluate(payload):
    a_events=payload['a_events'];b=payload['b'];c_events=payload['c_events'];states=[]
    left=[a for a in a_events if bridge_records(a,0,b,0)]
    right=[c for c in c_events if bridge_records(b,1,c,0)]
    for a in left:
        for c in right:
            record=(a[0]|b[0]|c[0],(a[1][1],a[1][2],b[1][2],c[1][1],c[1][2]),min(a[2],b[2],c[2]),(a[3],b[3],c[3]),int(a[4] and b[4] and c[4]))
            identity={'schema':IDENTITY_VERSION,'catalog_sha256':payload['catalog_sha256'],'classes':list(CLASSES),'record':record}
            states.append({'identity':identity,'state':{'record':record,'object_id':canonical_sha256(identity),
                'components':[event_id(a),event_id(b),event_id(c)]}})
    return {'states':states,'metrics':{'left_links':len(left),'right_links':len(right),'joined_paths':len(states)}}

def export_rows(generated):
    """Lossless typed export: resources, endpoints, formation and occurrences."""
    state=generated['state'];r=state['record'];oid=state['object_id']
    if len(r[1])!=5 or len(r[3])!=3 or len(state['components'])!=3:raise ValueError('STEP3_EXPORT_SHAPE')
    labels=('A1','A2','B2','C1','C2')
    yield {'type':'OBJECT','id':oid,'identity_version':IDENTITY_VERSION,'record':r,
           'ports':[{'id':oid+':'+label,'owner':oid,'slot':label,'P':pm[0],'M':pm[1]} for label,pm in zip(labels,r[1])],
           'components':[{'slot':slot,'event_id':event,'target_class':target} for slot,event,target in zip(('A','B','C'),state['components'],r[3])],
           'internal_bridges':[['A0','B0'],['B1','C0']],
           'multiplicity':generated['occurrence_count']}
    # This first registered case has one occurrence per object. Refuse rather than
    # collapse a future multiplicity without enumerating its occurrence bindings.
    if generated['occurrence_count']!=1:raise ValueError('STEP3_OCCURRENCE_EXPORT_REQUIRES_UNIQUE_CASE')
    yield {'type':'OCCURRENCE','id':generated['representative_occurrence_id'],'object_id':oid,
           'components':state['components'],'recipe':'L2_STEP10_CASE5'}

def handler(stage,runtime):
    p=stage['execution']['parameters']
    raw=Path(stage['input_artifacts']['l2_tables']).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=p['input_sha256']:raise ValueError('STEP3_INPUT_HASH')
    if p['classes']!=list(CLASSES):raise ValueError('STEP3_PARAMETER_SCOPE')
    packet=json.loads(raw);T=InputTables(packet)
    events=[sorted(j3(T,T.rep[c]),key=repr) for c in CLASSES]
    if [len(e) for e in events]!=[27,27,27]:raise ValueError('STEP3_J3_CARDINALITY')
    context=runtime.publish_bulk_shard('STEP3_FRESH_J3_EVENTS',{'schema':'IG_STEP3_J3_EVENTS_V1','classes':list(CLASSES),
       'events':[[{'event_id':event_id(e),'record':e} for e in es] for es in events],
       'input_sha256':p['input_sha256'],'upstream':'RETAINED_TABLES','execution':'FRESH_J3_GENERATION'})
    tasks=[]
    for i,b in enumerate(events[1]):
        payload={'a_events':events[0],'b':b,'c_events':events[2],'catalog_sha256':packet['catalog_sha256']}
        tasks.append(TaskSpec(f'B{i:03d}','STEP3_FRESH_L2_ABC',canonical_sha256(payload),payload))
    result=runtime.run_content_indexed_generation(phase_id='L2_ABC',tasks=tasks,
        evaluator_ref='infinity_grid.step3_l2_producer:evaluate',max_tasks=27,
        max_generated_occurrences=19683,max_task_occurrences=729,max_result_bytes=2097152,max_work_seconds=30)
    refs=[];batch=[];objects=occurrences=0
    for generated in runtime.iter_generated_states(phase_id='L2_ABC'):
        for row in export_rows(generated):
            objects+=row['type']=='OBJECT';occurrences+=row['type']=='OCCURRENCE';batch.append(row)
            if len(batch)==4000:
                refs.append(runtime.publish_bulk_shard(f'STEP3_EXPORT_{len(refs):03d}',{'schema':'IG_STEP3_TYPED_EXPORT_SHARD_V1','rows':batch}));batch=[]
    if batch:refs.append(runtime.publish_bulk_shard(f'STEP3_EXPORT_{len(refs):03d}',{'schema':'IG_STEP3_TYPED_EXPORT_SHARD_V1','rows':batch}))
    if objects!=19683 or occurrences!=19683:raise ValueError('STEP3_EXPORT_CARDINALITY')
    return ChainExecutionResult(result={'outcome':'PASS','generation':result.summary,'execution':result.execution_metadata,
        'export':{'objects':objects,'occurrences':occurrences,'port_incidences':objects*5,'component_links':objects*3,
                  'internal_bridges':objects*2,'shards':refs,'j3_context':context},
        'coverage':{'input_tables':'RETAINED','j3_and_abc':'FRESH_GENERATED','identity':IDENTITY_VERSION,
            'algebra_backend':'LEGACY_PURE_STDLIB_CORE_L2','storage_backend':'SQLITE_DELETE_FULL_AND_IMMUTABLE_SHARDS',
            'library_migration_claim':'No new algebra-library migration claimed; igraph/SymPy not invoked by this scope.',
            'scope':'ENGINEERING_VALIDATION_CASE5_NOT_OFFICIAL_MASTER_NOT_FRESH_L0'}})
