"""Versioned execution-lifetime policy shared by registered runtimes."""
LEGACY = "LEGACY_RUNTIME_BUDGET_V1"
NO_DEADLINE = "NO_AUTOMATIC_RUNTIME_DEADLINE_V1"
SUPPORTED = {LEGACY, NO_DEADLINE}

def name(resources):
    value = (resources or {}).get("execution_policy", LEGACY)
    if value not in SUPPORTED:
        raise ValueError("EXECUTION_POLICY_NOT_SUPPORTED:" + str(value))
    return value

def automatic_deadlines(resources):
    return name(resources) == LEGACY
