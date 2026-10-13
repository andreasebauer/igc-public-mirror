from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

from infinity_grid.core.canonical import canonical_bytes
import infinity_grid.g6_s7_depth2_completion as d2


def _ops31():
    ops=[(0,0)]
    for i in range(1,16):
        ops.extend([(i,100+i),(100+i,i)])
    assert len(ops)==31
    return tuple(ops)


def test_a22_context_normalization_exact_248_to_124_alias_coverage():
    refs=("A","B","C","D"); ops=_ops31()
    obj=d2._e2_context_normalization(refs,ops)
    assert obj["scientific_context_count"]==248
    assert obj["execution_context_count"]==124
    assert len(obj["scientific_to_execution"])==248
    assert len(obj["execution_coordinates"])==124
    assert all(len(r["scientific_alias_indices"])==2 for r in obj["execution_coordinates"])
    # Scientific index 1 is RIGHT(A,(0,0)); diagonal transpose maps to same exec as LEFT.
    assert obj["scientific_to_execution"][0]["execution_context_index"]==0
    assert obj["scientific_to_execution"][1]["execution_context_index"]==0
    # For a non-diagonal pair, RIGHT maps to the transposed LEFT operator.
    right=obj["scientific_to_execution"][3]  # A,(1,101),RIGHT
    assert right["position"]=="RIGHT"
    assert right["execution_operator"]==[101,1]
    assert right["mapping_rule"]=="RIGHT_TO_TRANSPOSED_LEFT"


class _Runtime:
    def __init__(self, root: Path): self.root=root
    def _phase_root(self, phase_id: str) -> Path: return self.root/phase_id


def _make_partition(root: Path, phase: str, rows):
    p=root/phase; p.mkdir(parents=True)
    db=p/"partition.sqlite3"; conn=sqlite3.connect(db)
    conn.execute("CREATE TABLE task_results(task_id TEXT PRIMARY KEY,payload_sha256 TEXT,signature_sha256 TEXT,class_token TEXT,outcome_count INTEGER,metrics_json TEXT,locator_json TEXT,committed_utc TEXT)")
    conn.execute("CREATE TABLE classes(class_token TEXT PRIMARY KEY,signature_sha256 TEXT,class_index INTEGER,representative_task_id TEXT,size INTEGER,representative_signature_bytes BLOB)")
    by={}
    for tid,token,sig in rows:
        raw=canonical_bytes(sig); sha=hashlib.sha256(raw).hexdigest(); by.setdefault(token,(sha,tid,raw,0)); x=by[token]; by[token]=(x[0],x[1],x[2],x[3]+1)
        conn.execute("INSERT INTO task_results VALUES(?,?,?,?,?,?,?,?)",(tid,"p",sha,token,1,"{}",None,"now"))
    for i,(token,(sha,rep,raw,size)) in enumerate(sorted(by.items())):
        conn.execute("INSERT INTO classes VALUES(?,?,?,?,?,?)",(token,sha,i,rep,size,raw))
    conn.commit(); conn.close()


def test_a22_split_event_has_exact_structural_separator_and_stable_group_ids(tmp_path):
    phase="S7D2_S1_P008_O000_PARENT_PREFIX"
    members=[
        {"state_token":"a","identity_sha256":"a"*64,"state":{}},
        {"state_token":"b","identity_sha256":"b"*64,"state":{}},
        {"state_token":"c","identity_sha256":"c"*64,"state":{}},
    ]
    rows=[]
    for tok,cls,sig in [("a","C0",["x",1]),("b","C1",["x",2]),("c","C1",["x",2])]:
        tid=f"S7D2-PARENT-PREFIX-s1-P008-O000-{tok}"; rows.append((tid,cls,sig))
    _make_partition(tmp_path,phase,rows); rt=_Runtime(tmp_path)
    memberships=d2._partition_memberships(rt,phase)
    norm=d2._e2_context_normalization(("A","B","C","D"),_ops31())
    out,events=d2._refine_parent_prefix_with_e2(
        [{"depth1_class_id":"D1","members":members}],memberships,panel="s1",
        prefix_context_count=8,outer_context_index=0,phase_id=phase,runtime=rt,
        context_normalization=norm,
    )
    assert [len(g["members"]) for g in out]==[1,2]
    assert len(events)==1
    ev=events[0]; assert len(ev["buckets"])==2
    sep=ev["exact_structural_separator"]; assert len(sep)==2
    assert sep[0]["representative_signature_canonical_utf8"] != sep[1]["representative_signature_canonical_utf8"]
    for row in sep:
        raw=row["representative_signature_canonical_utf8"].encode()
        assert hashlib.sha256(raw).hexdigest()==row["signature_sha256"]
    gid=d2._e2_group_id("s1","D1",members)
    assert gid==ev["parent_group_id"]==d2._e2_group_id("s1","D1",list(reversed(members)))


def test_a22_final_export_has_complete_memberships_survivors_and_thin_manifest(tmp_path):
    groups={
        "s1":[{"depth1_class_id":"D1","members":[
            {"state_token":"a","identity_sha256":"a"*64,"state":{}},
            {"state_token":"b","identity_sha256":"b"*64,"state":{}},
        ]}],
        "higher":[{"depth1_class_id":"H1","members":[
            {"state_token":"h","identity_sha256":"c"*64,"state":{}},
        ]}],
    }
    norm=d2._e2_context_normalization(("A","B","C","D"),_ops31())
    ref=d2._write_e2_final(tmp_path,groups,[],norm,accepted_source_sha256="d"*64,scientific_design_sha256="e"*64,reuse_payload_sha256="f"*64,question_sha256="1"*64)
    m=json.loads((tmp_path/"G6_S7_DEPTH2_E2_MEMBERSHIPS.json").read_text())
    s=json.loads((tmp_path/"G6_S7_DEPTH2_E2_SURVIVORS.json").read_text())
    man=json.loads((tmp_path/"G6_S7_DEPTH2_E2_MANIFEST.json").read_text())
    assert m["member_count"]==3 and len(m["rows"])==3
    assert s["survivor_class_count"]==1 and s["survivors"][0]["class_size"]==2
    assert man["recursive_zip_nesting"] is False
    assert {x["logical_name"] for x in man["files"]}=={"memberships","split_witnesses","survivors","context_normalization"}
    assert ref["member_count"]==3


def test_a22_progress_checkpoint_is_atomic_current_state_not_nested_bundle(tmp_path):
    groups={"s1":[{"depth1_class_id":"D","members":[{"state_token":"a","identity_sha256":"a"*64,"state":{}}]}],"higher":[]}
    obj=d2._write_e2_progress(tmp_path,groups,[],cursor={"inner_prefix_context_count":8,"completed_outer_execution_context_index":7},normalization_sha256="1"*64,accepted_source_sha256="2"*64,scientific_design_sha256="3"*64,reuse_payload_sha256="4"*64)
    p=tmp_path/"G6_S7_DEPTH2_E2_PROGRESS.json"
    assert p.is_file() and not list(tmp_path.glob("*.zip"))
    loaded=json.loads(p.read_text()); assert loaded==obj
    assert loaded["groups"][0]["members"][0]["state_token"]=="a"
