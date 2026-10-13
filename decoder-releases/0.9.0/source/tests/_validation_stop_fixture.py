"""Intentional negatives: selected only by bounded runner regression controls."""
def test_failure():
    raise AssertionError('REGISTERED_NEGATIVE_CONTROL')

def test_later_must_not_run():
    raise AssertionError('LATER_SELECTOR_EXECUTED')
