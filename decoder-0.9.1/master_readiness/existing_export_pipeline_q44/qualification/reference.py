"""Independent saved-page byte reader; no Decoder imports or scientific kernels."""
import hashlib,json
def read(z,prefix,candidate):
    def blob(ref):
        raw=z.read(prefix+'content/'+ref['sha256']+'.blob')
        assert len(raw)==int(ref['size_bytes']) and hashlib.sha256(raw).hexdigest()==ref['sha256']
        return raw
    counts={};h=hashlib.sha256();rowhashes={}
    for kind,collection in sorted(candidate['collections'].items()):
        count=0;prior=None
        def shards(ref):
            page=json.loads(blob(ref))
            for entry in page['entries']:
                if entry['entry_type']=='PAGE':yield from shards(entry['content_ref'])
                else:yield blob(entry['content_ref'])
        profile=json.loads(blob(collection['key_definition_ref']))
        assert profile['format']=='CANONICAL_STORAGE_JSONL_V1'
        for raw in shards(collection['root']):
            for line in raw.splitlines():
                row=json.loads(line);assert json.dumps(row,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()==line
                values=[]
                for path in profile['key_fields']:
                    v=row
                    for field in path:v=v[field]
                    values.append(v)
                key=json.dumps(values,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode().hex()
                assert prior is None or prior<key;prior=key
                h.update(len(line).to_bytes(8,'big'));h.update(line);count+=1
        assert count==candidate['counts'][kind];counts[kind]=count
    return counts,h.hexdigest()
