from pathlib import Path
import sys,json,hashlib,shutil,tempfile,resource
h=Path(__file__).resolve().parent;r=h.parent;s=r/'recover167/source';sys.path.insert(0,str(s))
resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
from infinity_grid import submission as sub,portable_registry as reg,project_checkpoint as pc,result_contracts as rc
from infinity_grid.checkpoint_stream import bounded_tree
assert shutil.disk_usage(r).free>=10*1024**3
pins={str(p.relative_to(r)):sub._sha(p.read_bytes()) for p in [h/'qualify.py',r/'recover167/source.zip',r/'recover167/project.zip',r/'recover167/evidence/native_workspace/CAPTURE.json']}
(h/'REGISTRATION.json').write_text(json.dumps({'scope':'Reconstructed closure fixture using real saved validation bytes; no native rerun or original capsule identity claim','pins':pins,'stop':'First unexplained mismatch','limits':'Existing defaults; externalization opt-in only'},indent=2))
root=h/'project';assert not root.exists();shutil.copytree(r/'recover167/project',root)
baseline=(r/'recover167/project.zip').read_bytes();files={}
# Reconstruct the same content-addressed closure shape. No original capsule digest claim.
for base,prefix in [(s,'source'),(r/'recover167/evidence/native_workspace/runtime','runtime')]:
 for p in sorted(base.rglob('*')):
  if p.is_file() and '__pycache__' not in p.parts:
   files[prefix+'/'+p.relative_to(base).as_posix()]=reg.put(root,p.read_bytes())
capsule={'schema_id':'IG_PROJECT_COMPLETION_CAPSULE_V1','files':files,'job_id':'P0.DEV165.CAPTURE.VALIDATION','fixture_only':True}
digest=reg.put(root,sub._json_bytes(capsule));reg.append(root,'closure-fixture',[],{'capsule':digest},'Qualification closure from real captured source and saved validation records; no scientific claim')
limits=rc.policy({'preservation':{'project_evidence_externalization':1}})
raw=sum(p.stat().st_size for p in root.rglob('*') if p.is_file());assert raw>limits['max_project_raw_bytes']
try:bounded_tree(root,limits['max_project_raw_bytes'])
except sub.SubmissionError as e:assert e.code=='CHECKPOINT_TREE_RAW_LIMIT'
else:raise RuntimeError('INLINE_REFUSAL_REQUIRED')
with tempfile.TemporaryDirectory() as tmp:
 c=pc.prepare(root,baseline,limits,tmp); project={'schema_id':pc.SCHEMA,**{k:c[k] for k in ['full_sha256','metadata_sha256','evidence_raw_bytes','evidence_file_count']}}
 full=pc.hydrate(c['metadata'],c['evidence_path'].read_bytes(),project,limits)
 assert full==reg.snapshot(root)
 negatives=0
 for field,value in [('metadata_sha256','0'*64),('full_sha256','0'*64),('evidence_file_count',c['evidence_file_count']+1)]:
  try:pc.hydrate(c['metadata'],c['evidence_path'].read_bytes(),dict(project,**{field:value}),limits)
  except sub.SubmissionError:negatives+=1
  else:raise RuntimeError('TAMPER_ACCEPTED')
 try:pc.hydrate(c['metadata'],c['evidence_path'].read_bytes(),project,dict(limits,project_evidence_externalization=0))
 except sub.SubmissionError:negatives+=1
 else:raise RuntimeError('OPT_IN_BYPASSED')
 result={'status':'PASS_RECONSTRUCTED_CLOSURE_FIXTURE','project_raw_bytes':raw,'metadata_raw_bytes':c['metadata_raw_bytes'],'evidence_raw_bytes':c['evidence_raw_bytes'],'evidence_file_count':c['evidence_file_count'],'full_snapshot_exact':True,'negative_checks':negatives,'limits':limits,'native_checkpoint_qualified':False,'original_capsule_recovered':False}
(h/'results/RESULT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
