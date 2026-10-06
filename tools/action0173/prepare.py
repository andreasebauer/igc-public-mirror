from pathlib import Path
import json,hashlib,zipfile
B=Path(__file__).resolve().parent;R=B.parent;V=R/'o7_additional_readiness0172'
sha=lambda b:hashlib.sha256(b).hexdigest();load=lambda p:json.loads(p.read_bytes())
readiness=load(V/'READINESS_VALIDATION.json')
for n,h in readiness['output_pins'].items():assert sha((V/n).read_bytes())==h
for n,x in load(V/'INPUT_PINS.json').items():assert sha((R/n).read_bytes())==x['sha256']
pb=load(R/'scope_reconcile0171/RECOVERED_PROFILE_BINDINGS.json')
files={'roots':V/'ROOT_SCOPE.json','objects':V/'COMPONENT_OBJECTS.json','occurrences':V/'COMPONENT_OCCURRENCES.json','owners':V/'OWNER_BINDINGS.json','contract':V/'EXPORT_CONTRACT.json','readiness':V/'READINESS_VALIDATION.json','input_pins':V/'INPUT_PINS.json','profiles':R/pb['profiles']['recovered_path'],'survivors':R/pb['survivors']['recovered_path'],'source_profile_bindings':R/'scope_reconcile0171/RECOVERED_PROFILE_BINDINGS.json','source_search':R/'scope_reconcile0171/SOURCE_SEARCH.json'}
hits=load(R/'scope_reconcile0171/SOURCE_SEARCH.json')['hits']
for h in {x['source_checkpoint_sha256'] for x in load(V/'ROOT_SCOPE.json')}:
 hit=next(x for x in hits if x['sha256']==h);files['checkpoint_'+h]=R/hit['recovered_path']
blobs={};items=[]
for n,p in files.items():
 b=p.read_bytes();h=sha(b);items.append({'name':n,'sha256':h,'bytes':len(b)});blobs[h]=b
root={'schema':'IG_ADDITIONAL_SAVED_O7_EXPORT_V1','scope':'24 additional saved whole HOM6 roots,205 source component occurrences,164 distinct exact contextual objects; external O6 twins explicit','content':items,'O7_bindings':load(R/'o7_export0169/EXPORT_BINDINGS.json'),'catalog_sha256':sha((R/'o7_integrate0170/CATALOG_0148.json').read_bytes())}
raw=json.dumps(root,sort_keys=True,separators=(',',':')).encode()
with zipfile.ZipFile(B/'SCIENTIFIC_EXPORT.zip','w',zipfile.ZIP_DEFLATED) as z:
 def put(n,b):
  i=zipfile.ZipInfo(n,(2026,10,6,0,0,0));i.compress_type=zipfile.ZIP_DEFLATED;z.writestr(i,b)
 put('ROOT.json',raw)
 for h,b in sorted(blobs.items()):put('content/'+h+'.blob',b)
meta={'archive_sha256':sha((B/'SCIENTIFIC_EXPORT.zip').read_bytes()),'root_sha256':sha(raw),'catalog_sha256':root['catalog_sha256']};(B/'EXPORT_BINDINGS.json').write_text(json.dumps(meta,indent=2)+'\n')
s=load(R/'o7_export0169/SPEC.json');s.update(job_id='MASTER.O7.ADDITIONAL.SAVED.ROOTS.COMPONENTS.EXPORT.READER.V1',project_source=str(B/'project'));s['question'].update(stage_id='MASTER:O7:ADDITIONAL:SAVED:ROOTS:COMPONENTS:EXPORT:READER:V1',description=root['scope']);s['execution']['parameters']=meta
paths=[('archive',B/'SCIENTIFIC_EXPORT.zip'),('catalog',R/'o7_integrate0170/CATALOG_0148.json')]+[(name+'_archive',R/folder/'SCIENTIFIC_EXPORT.zip') for name,folder in [('o7','o7_export0169'),('o6','o6_export0165'),('o5','o5_export0162'),('o4','o4_export0159'),('o3','o3_export0156')]]
s['inputs']=[{'logical_name':n,'path':str(p),'sha256':sha(p.read_bytes())} for n,p in paths]
s['output_contract']['result_checks']=[{'pointer':'/'+k,'equals':v} for k,v in dict(readiness['counts'],outcome='PASS',generation_calls=0,scientific_master_admission=False).items()]
(B/'SPEC.json').write_text(json.dumps(s,indent=2)+'\n');print(json.dumps(meta))
