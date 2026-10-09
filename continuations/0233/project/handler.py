"""Native qualification of the scoped master152 exact-parent reader."""
from infinity_grid.v05_chain import ChainExecutionResult
from .reader import qualify
def handler(stage,runtime):
    result=qualify(stage['input_artifacts'],stage['execution']['parameters']['bindings'])
    runtime.publish_json('exact_parent_reader_qualification',result)
    return ChainExecutionResult(result=result)
