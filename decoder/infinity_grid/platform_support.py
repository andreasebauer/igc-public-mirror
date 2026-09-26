"""Admission boundary for the native Decoder runtime."""
import os
import sys


def require_supported_platform():
    if sys.platform != 'linux' or not hasattr(os, 'fork'):
        raise RuntimeError(
            'DECODER_PLATFORM_UNSUPPORTED: use Linux with fork, fcntl locks, '
            'and a local filesystem supporting atomic rename and hard links. '
            'Windows and macOS execution are not supported.')
    if sys.version_info < (3, 11):
        raise RuntimeError('DECODER_PYTHON_UNSUPPORTED: Python 3.11 or newer is required; '
                           'the qualification profile specifies the tested interpreter.')
