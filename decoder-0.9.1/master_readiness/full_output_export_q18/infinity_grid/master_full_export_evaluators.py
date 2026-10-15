"""Thin registered full-output producers; semantics remain Decoder services."""
from .v05_kernel_services import current_kernel_view


def owner_export_evaluator(payload):
    return current_kernel_view().call('COMPLETE_OWNER_EXPORT',payload['left'],payload['right'],
        payload['operator'],payload['binding'])


def marker_export_evaluator(payload):
    return current_kernel_view().call('COMPLETE_MARKER_EXPORT',payload['left'],payload['right'],
        payload['operator'],payload['binding'])


def saved_dag_export_evaluator(payload):
    return current_kernel_view().call('SAVED_DAG_FULL_EXPORT',payload['root'],payload['input_binding'],
        payload['role'],payload['binding'])
