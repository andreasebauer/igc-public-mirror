"""Bounded, fail-closed reader of a pinned scientific archive, without generation."""
from pathlib import Path
import hashlib, zipfile
from project.reader import RecursiveReader, check, strict

def read_archive(archive, root_sha256, catalog_sha256, view_sha256):
    p=Path(archive)
    check(p.is_file() and not p.is_symlink() and p.stat().st_size<=100*1024*1024,'ARCHIVE_LIMIT')
    with zipfile.ZipFile(p) as z:
        check(len(z.namelist())==len(set(z.namelist()))==183,'ARCHIVE_MEMBER_COUNT')
        rootinfo=z.getinfo('ROOT.json');check(rootinfo.file_size<=1048576,'ROOT_LIMIT')
        raw=z.read('ROOT.json');check(hashlib.sha256(raw).hexdigest()==root_sha256,'SCIENTIFIC_ROOT_HASH');root=strict(raw)
        check(root['schema_id']=='IG_RECURSIVE_SCIENTIFIC_EXPORT_V1' and root['dataset_id']=='NODE_NATIVE_ROOTED_DEPTH1_DEPTH2_V1','SCIENTIFIC_PROFILE')
        check(root['scope']=={'depths':[1,2],'native_events':13,'depth1_constructions':1391,'depth2_constructions':198188,'selection':'EXHAUSTIVE_LAWFUL_ENDPOINTS_WITHIN_FROZEN_13_EVENT_ALPHABET'},'SCIENTIFIC_SCOPE')
        check(root['identity']=='ROOTED_CONSTRUCTION_AND_FORMATION_IDS_WITH_EXPLICIT_PROJECTION_PREIMAGES','SCIENTIFIC_IDENTITY')
        check(root['historical_selection_weights_qualified'] is False and root['historical_l13_reproduced'] is False,'HISTORICAL_AUTHORITY')
        members=root['scientific_members'];refs={x['sha256']:x for x in members}
        check(len(members)==len(refs)==182 and root['qualified_view_sha256']==view_sha256,'SCIENTIFIC_MEMBERS')
        check(set(z.namelist())=={'ROOT.json'}|{'content/'+h+'.blob' for h in refs},'SCIENTIFIC_CLOSURE')
        check(sum(x['size_bytes'] for x in members)==286694082,'SCIENTIFIC_PAYLOAD_BYTES')
        locations={}
        for h,ref in refs.items():
            check(len(h)==64 and all(c in '0123456789abcdef' for c in h),'MEMBER_HASH_FORMAT')
            info=z.getinfo('content/'+h+'.blob')
            check(0<info.file_size<=4194304 and info.file_size==ref['size_bytes'] and not info.is_dir() and not info.flag_bits&1,'SCIENTIFIC_OBJECT_LIMIT')
            b=z.read(info);check(hashlib.sha256(b).hexdigest()==h,'SCIENTIFIC_MEMBER_HASH');locations[h]=b
    view=strict(locations[view_sha256]);deps=root['dependencies']
    check(deps['admitted_j3_catalog_sha256']==view['catalog_sha256']==catalog_sha256,'ADMITTED_CATALOG_REF')
    check(deps['primitive_packet_sha256']==view['primitive_packet_sha256'] and deps['parent_packet_sha256']==view['parent_packet_sha256'],'PACKET_REFERENCE')
    expected={view_sha256,view['primitive_packet_sha256'],view['parent_packet_sha256']}|{r['sha256'] for s in view['sources'] for r in s['shards']}
    check(expected==set(locations),'QUALIFIED_VIEW_MEMBER_CLOSURE')
    reader=RecursiveReader(locations[view_sha256],view_sha256,locations)
    check(deps['foundation_root_sha256']==reader.packet['parent_root_sha256'],'FOUNDATION_REFERENCE')
    check(root['depth2_ordered_stream_sha256']==reader.report['full_depth2_stream_sha256'],'SCIENTIFIC_STREAM')
    return reader,root
