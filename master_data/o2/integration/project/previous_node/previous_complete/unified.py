from pathlib import Path
import hashlib,json,tempfile,zipfile
from .previous.unified import IntegratedMasterReader
from .previous.recursive_reader import check
from .legacy_pairs.prefix_reader import PrefixReader
from .legacy_pairs.pair_reader import PairReader
from .legacy_pairs.pair_contract import PARENT_ROOT
from .bindings import CATALOG_SHA256,BRANCH_SHA256,ADMISSION_SHA256

class CompleteMasterReader:
 def __init__(self,catalog,branch,admission,parent_archive,pair_archive,*,previous_catalog,recursive_admission,predecessor_catalog,recursive_archive,canonical_directory=None,archive_directory=None):
  self.closed=False;self.previous=None;self.temp=None
  try:
   def load(p,h):
    raw=Path(p).read_bytes();check(hashlib.sha256(raw).hexdigest()==h,'PINNED_AUTHORITY_HASH');return json.loads(raw)
   c=load(catalog,CATALOG_SHA256);old=json.loads(Path(previous_catalog).read_bytes());branch=load(branch,BRANCH_SHA256);ad=load(admission,ADMISSION_SHA256)
   check(c['release_id']=='MASTER_DATA_V1_0141' and c['slices'][:-1]==old['slices'] and len(old['slices'])==140,'PREDECESSOR_ROUTES')
   check(c['slices'][-1]==branch['slices'][1] and c['slices'][0]==branch['slices'][0],'BRANCH_EXACT_MERGE')
   check(ad['decision']=='ACCEPTED_FOR_CANONICAL_REUSE_WITHIN_DECLARED_PAIR_SCOPE' and ad['scientific_root_sha256']==c['slices'][-1]['scientific_root_sha256'] and ad['dependency_root_sha256']==PARENT_ROOT,'PAIR_ADMISSION')
   self.previous=IntegratedMasterReader(previous_catalog,recursive_admission,predecessor_catalog,recursive_archive,canonical_directory=canonical_directory,archive_directory=archive_directory)
   self.temp=tempfile.TemporaryDirectory();base=Path(self.temp.name)
   for name,p,s in [('parent',parent_archive,c['slices'][0]),('pairs',pair_archive,c['slices'][-1])]:
    raw=Path(p).read_bytes();check(len(raw)==s['archive']['size_bytes'] and hashlib.sha256(raw).hexdigest()==s['archive']['sha256'],'SCIENTIFIC_ARCHIVE_HASH')
    dest=base/name;dest.mkdir()
    with zipfile.ZipFile(p) as z:
     names=z.namelist();check(len(names)==len(set(names))==s['counts']['scientific_files'],'SCIENTIFIC_ARCHIVE_INVENTORY')
     for info in z.infolist():
      n=Path(info.filename);check(not n.is_absolute() and '..' not in n.parts and (info.filename=='ROOT.json' or (len(n.parts)==2 and n.parts[0]=='content' and n.suffix=='.blob')) and not info.is_dir() and not info.flag_bits&1 and info.file_size<=4194304,'UNSAFE_ARCHIVE_MEMBER')
      q=dest/n;q.parent.mkdir(exist_ok=True);q.write_bytes(z.read(info))
   self.parent=PrefixReader(base/'parent',PARENT_ROOT);self.parent.verify();self.pairs=PairReader(base/'pairs',c['slices'][-1]['scientific_root_sha256'],self.parent);self.pairs.verify();self.catalog=c
  except BaseException:self.close();raise
 def ready(self):check(not self.closed,'READER_CLOSED')
 def lookup_pair(self,family,object_id):self.ready();return self.pairs.lookup(family,object_id)
 def lookup_pair_formation(self,family,formation_id):
  self.ready();check(family in self.pairs.formations and formation_id in self.pairs.formations[family],'PAIR_FORMATION_NOT_FOUND')
  row=self.pairs.formations[family][formation_id];out=self.pairs.lookup(family,row['object_id']);return next(x for x in out['occurrences'] if x['formation']['formation_id']==formation_id)
 def lookup_construction(self,*a,**k):self.ready();return self.previous.lookup_construction(*a,**k)
 def lookup_formation(self,*a,**k):self.ready();return self.previous.lookup_formation(*a,**k)
 def lineage(self,*a,**k):self.ready();return self.previous.lineage(*a,**k)
 def projection_preimages(self,*a,**k):self.ready();return self.previous.projection_preimages(*a,**k)
 def lookup_j3(self,*a,**k):self.ready();return self.previous.lookup_j3(*a,**k)
 def lookup_target(self,*a,**k):self.ready();return self.previous.lookup_target(*a,**k)
 def iter_j3(self,*a,**k):self.ready();return self.previous.iter_j3(*a,**k)
 def primitive(self,*a,**k):self.ready();return self.previous.primitive(*a,**k)
 def triple(self,*a,**k):self.ready();return self.previous.triple(*a,**k)
 def lookup_abc(self,*a,**k):self.ready();return self.previous.lookup_abc(*a,**k)
 def foundation_catalog_sha256(self):self.ready();return self.previous.foundation_catalog_sha256()
 def coverage_report(self):
  self.ready();return {'release_id':'MASTER_DATA_V1_0141','scientific_slices':141,'j3_carriers':300696,'recursive_constructions':199579,'pair_objects':1458,'pair_formations':1458,'generator_calls':0,'full_l0_to_g8_complete':False}
 def close(self):
  self.closed=True
  if self.previous:self.previous.close()
  if self.temp:self.temp.cleanup()
 def __enter__(self):self.ready();return self
 def __exit__(self,*exc):self.close()
