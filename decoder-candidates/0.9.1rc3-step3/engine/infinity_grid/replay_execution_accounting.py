"""Assertion-level observed execution, separate from catalogue declarations.

Imported checkpoints without these attestations stay unverified restored
observations; their original declarations never become new fresh evidence.
"""
from collections import Counter

SCHEMA = 'IG_ASSERTION_EXECUTION_ACCOUNTING_V1'


def attach_execution_accounting(result, node, catalogue):
    mappings = {m['assertion_id']: m for m in catalogue['historical_assertion_mappings']}
    ids = node.get('assertion_mapping_ids', [])
    actual = result['execution_class']
    bounded_fresh = actual == 'FRESH_RECOMPUTE' and node['layer'] == 'NODE_IN'
    result['assertion_execution_accounting'] = [{
        'schema_id': SCHEMA, 'assertion_id': i,
        'requested_execution_class': mappings[i]['execution_class'],
        'observed_execution_class': actual,
        'input_generation': 'HISTORICAL_PINNED_INPUTS',
        'fresh_scope': 'BOUNDED_43008_COMPOSITION_CHECKS_ON_RETAINED_CENSUS'
                       if bounded_fresh else 'NONE',
        'counts_toward_full_fresh_generation': False,
    } for i in ids]
    result['counts_toward_full_fresh_generation'] = False
    return result


def observed_coverage(manifest, runner):
    observations = {}
    completed = set(runner.state['completed_node_ids'])
    for n in completed:
        import json
        checkpoint = json.loads((runner.state_dir / 'checkpoints' /
            (runner.state['accepted_checkpoint_sha256_by_node'][n] + '.json')).read_text())
        result = checkpoint.get('acceptance', {}).get('node_result', {})
        for row in result.get('assertion_execution_accounting', []):
            observations[row['assertion_id']] = dict(row)
    rows = []
    for m in manifest.get('historical_assertion_mappings', []):
        row = observations.get(m['assertion_id'])
        if row is None:
            restored = any(n in completed for n in m['canonical_obligation_ids'])
            row = {'schema_id':SCHEMA, 'assertion_id':m['assertion_id'],
                   'requested_execution_class':m['execution_class'],
                   'observed_execution_class':'UNATTESTED_RESTORED' if restored else 'NOT_EXECUTED',
                   'input_generation':'UNATTESTED' if restored else 'NONE',
                   'fresh_scope':'NONE', 'counts_toward_full_fresh_generation':False}
        rows.append(row)
    return {'schema_id':SCHEMA, 'assertions':rows,
            'observed_class_counts':dict(sorted(Counter(r['observed_execution_class'] for r in rows).items())),
            'full_fresh_generation_assertion_count':sum(r['counts_toward_full_fresh_generation'] for r in rows)}
