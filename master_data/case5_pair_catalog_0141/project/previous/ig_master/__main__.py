import argparse,json,sys
from .master_reader import MasterReader
p=argparse.ArgumentParser(description='Read admitted canonical J3 data; no generation or repair.')
p.add_argument('directory');p.add_argument('command',choices=['verify','integrity','j3','triple','primitive','abc']);p.add_argument('key',nargs='?')
a=p.parse_args()
with MasterReader(a.directory) as reader:
 if a.command=='verify':
  result=reader.verify(progress=lambda r:print('Verified '+r['scientific_root_sha256'],file=sys.stderr,flush=True))
 elif a.command=='integrity':result=reader.integrity_report()
 elif a.command=='abc':result=reader.lookup_abc(a.key,catalog_sha256=reader.foundation_catalog_sha256())
 else:
  if a.key is None:p.error('This command requires an integer key.')
  method={'j3':reader.lookup_j3,'triple':reader.triple,'primitive':reader.primitive}[a.command]
  result=method(int(a.key))
 print(json.dumps(result,indent=2,ensure_ascii=False))
