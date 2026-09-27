from __future__ import annotations
import pytest

from infinity_grid.canon import canonical_sha256
from infinity_grid.v05_chain import (
    CHAIN_SCHEMA_V1 as CHAIN_SCHEMA, ScientificChainController,
    ScientificChainValidationError, seal_chain_registration, validate_chain_registration,
)
from infinity_grid.v05_execution_authority import ExecutionAuthorityError

SHA0="0"*64; SHA1="1"*64

def stage(sid, series, key, outcomes, transitions, *, auto=True):
    return {
        "stage_id":sid,"series":series,"stage_kind":"PREPARED_EXECUTOR",
        "question_ref":f"question:{sid}","question_sha256":SHA1,"depends_on":[],
        "execution":{"executor_key":key,"parameters":{}},
        "result_contract":{"artifact_logical_name":"result.json","outcome_pointer":"/status","allowed_outcomes":outcomes},
        "transitions":transitions,"auto_run":auto,"promotion_effect":"NONE",
    }

def reg(stages, *, mirror=False):
    seen=[]
    for s in stages:
        s["depends_on"]=[seen[-1]] if seen and s["series"]!="REVIEW" else []
        seen.append(s["stage_id"])
    return seal_chain_registration({
        "schema_id":CHAIN_SCHEMA,"chain_id":"G6-CHAIN-TEST",
        "subject":{"hierarchy":"G","level":6,"parent_ref":"G5_GRADUATED"},
        "release_line":"v0.5 / 0.50","mode":"CONTROLLED_S_THEN_ADAPTIVE_R",
        "parent_authority":{"authority_ref":"g5-closeout","science_sha256":SHA0,"verification_sha256":SHA1,"status":"GRADUATED"},
        "stages":stages,"budgets":{"max_stage_executions":32,"max_chain_wall_seconds":60,"default_workers":4},
        "durability":{"fsync_each_transition":True,"stage_commits":"APPEND_ONLY","external_mirror_required":mirror},
        "authority_policy":{"automatic_promotion":False,"require_verified_parent":True,"novelty_policy":"REVIEW_REQUIRED",
            "changed_assumptions_policy":"REVIEW_REQUIRED","unregistered_outcome_policy":"REVIEW_REQUIRED",
            "r_science_policy":"ADAPTIVE_ONLY_WITHIN_PREREGISTERED_BRANCHES"},
    })

def _assert_legacy_executor_is_inspect_only(tmp_path):
    c=ScientificChainController(tmp_path, allow_legacy_new_registrations=True, engineering_only=True)
    with pytest.raises(ExecutionAuthorityError, match="LEGACY_EXECUTOR_DISABLED"):
        c.register_executor("legacy", lambda *_: None)

def test_s_chain_historical_v1_registration_is_inspect_only(tmp_path):
    s0=stage("G6:S0","S","s0",["PASS","REVIEW_REQUIRED"],{
        "PASS":{"action":"NEXT","next_stage":"G6:S1","reason":"PASS"},
        "REVIEW_REQUIRED":{"action":"REVIEW","next_stage":None,"reason":"SCIENCE_REVIEW"}})
    s1=stage("G6:S1","S","s1",["PASS"],{"PASS":{"action":"END","next_stage":None,"reason":"S_DONE"}})
    x=reg([s0,s1]); validate_chain_registration(x); assert x["stages"][0]["transitions"]["PASS"]["next_stage"]=="G6:S1"
    _assert_legacy_executor_is_inspect_only(tmp_path)

def test_unregistered_outcome_policy_remains_frozen_in_historical_registration(tmp_path):
    s0=stage("G6:S0","S","s0",["PASS"],{"PASS":{"action":"END","next_stage":None,"reason":"DONE"}})
    x=reg([s0]); validate_chain_registration(x); assert x["authority_policy"]["unregistered_outcome_policy"]=="REVIEW_REQUIRED"
    _assert_legacy_executor_is_inspect_only(tmp_path)

def test_r_branch_is_preregistered_and_pause_boundary_is_preserved(tmp_path):
    r0=stage("G6:R0","R","r0",["STRUCTURE","NO_STRUCTURE"],{
        "STRUCTURE":{"action":"NEXT","next_stage":"G6:R1","reason":"FOLLOW_REGISTERED_STRUCTURE_BRANCH"},
        "NO_STRUCTURE":{"action":"END","next_stage":None,"reason":"EARLY_CLOSE"}},auto=True)
    r1=stage("G6:R1","R","r1",["PASS"],{"PASS":{"action":"END","next_stage":None,"reason":"DONE"}},auto=False)
    x=reg([r0,r1]); validate_chain_registration(x); assert x["stages"][1]["auto_run"] is False
    _assert_legacy_executor_is_inspect_only(tmp_path)

def test_external_mirror_requirement_remains_frozen_in_historical_registration(tmp_path):
    s0=stage("G6:S0","S","s0",["PASS"],{"PASS":{"action":"END","next_stage":None,"reason":"DONE"}})
    x=reg([s0],mirror=True); validate_chain_registration(x); assert x["durability"]["external_mirror_required"] is True
    _assert_legacy_executor_is_inspect_only(tmp_path)

def test_r_policy_cannot_be_weakened():
    s0=stage("G6:S0","S","s0",["PASS"],{"PASS":{"action":"END","next_stage":None,"reason":"DONE"}})
    x=reg([s0]); x["authority_policy"]["r_science_policy"]="INVENT_AS_YOU_GO"
    x["registration_sha256"]=canonical_sha256({k:v for k,v in x.items() if k!="registration_sha256"})
    with pytest.raises(ScientificChainValidationError): validate_chain_registration(x)

def test_one_stage_budget_and_review_transition_remain_frozen(tmp_path):
    s0=stage("G6:S0","S","s0",["PASS"],{"PASS":{"action":"REVIEW","next_stage":None,"reason":"SCIENTIFIC_REVIEW"}})
    x=reg([s0],mirror=True); x["budgets"]["max_stage_executions"]=1
    x["registration_sha256"]=canonical_sha256({k:v for k,v in x.items() if k!="registration_sha256"})
    validate_chain_registration(x); assert x["stages"][0]["transitions"]["PASS"]["action"]=="REVIEW"
    _assert_legacy_executor_is_inspect_only(tmp_path)
