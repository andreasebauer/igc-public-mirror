"""Explicit transport-copy verification. No network or unattended saving.

The operator supplies downloaded bytes and actual Drive identifiers. A V2
receipt verifies reconstructed object bytes, never direct raw-object readback.
"""
from pathlib import Path
import argparse
import hashlib
import json
import lzma
import os
import re
import tempfile
import time

from .canon import canonical_sha256, write_json_atomic
from . import submission as sub

SCHEMA = 'IG_DECODER_TRANSPORT_MANIFEST_V1'
RECEIPT = 'IG_DECODER_TRANSPORT_SAVE_RECEIPT_V2'
BLOCK = 1024 * 1024
MAX_PART_BYTES = 48_000_000
MAX_MANIFEST_BYTES = 2 * BLOCK
SCOPE = 'EXPLICIT_CONNECTOR_OPERATOR_ATTESTATION_NOT_REMOTE_AUTHENTICATION'


def _id(value):
    return isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9_-]{10,200}', value) is not None


def _object(row):
    return (isinstance(row, dict) and set(row) == {'sha256', 'size_bytes'}
            and isinstance(row['sha256'], str) and re.fullmatch('[0-9a-f]{64}', row['sha256']) is not None
            and type(row['size_bytes']) is int and row['size_bytes'] >= 0)


def validate_manifest(row, *, bound=True):
    if (not isinstance(row, dict) or set(row) - {'schema_id', 'object', 'encoding', 'parts', 'original_object_drive_file_id'}
            or row.get('schema_id') != SCHEMA or not _object(row.get('object'))
            or row.get('encoding') not in {'RAW_CHUNKS', 'XZ_CHUNKS'}
            or not isinstance(row.get('parts'), list) or not 1 <= len(row['parts']) <= 4096):
        raise sub.SubmissionError('TRANSPORT_MANIFEST_INVALID')
    if 'original_object_drive_file_id' in row and not _id(row['original_object_drive_file_id']):
        raise sub.SubmissionError('TRANSPORT_MANIFEST_INVALID')
    for part in row['parts']:
        if (not isinstance(part, dict) or set(part) != {'sha256', 'size_bytes', 'drive_file_id'}
                or not _object({k: v for k, v in part.items() if k != 'drive_file_id'})
                or part['size_bytes'] > MAX_PART_BYTES
                or not (_id(part['drive_file_id']) or not bound and part['drive_file_id'] is None)):
            raise sub.SubmissionError('TRANSPORT_PART_INVALID')
    if row['encoding'] == 'RAW_CHUNKS' and sum(p['size_bytes'] for p in row['parts']) != row['object']['size_bytes']:
        raise sub.SubmissionError('TRANSPORT_SIZE_MISMATCH')
    return row


def read_manifest(path, *, bound=True):
    with Path(path).open('rb') as handle:
        raw = handle.read(MAX_MANIFEST_BYTES + 1)
    if len(raw) > MAX_MANIFEST_BYTES:
        raise sub.SubmissionError('TRANSPORT_MANIFEST_TOO_LARGE')
    try:
        row = json.loads(raw)
    except (ValueError, UnicodeError) as exc:
        raise sub.SubmissionError('TRANSPORT_MANIFEST_INVALID') from exc
    return raw, validate_manifest(row, bound=bound)


def _chunks(manifest, directory):
    for part in manifest['parts']:
        path = Path(directory) / part['sha256']
        if path.is_symlink() or not path.is_file():
            raise sub.SubmissionError('TRANSPORT_PART_MISSING_OR_UNSAFE', part['sha256'])
        digest = hashlib.sha256(); size = 0
        with path.open('rb') as handle:
            while data := handle.read(BLOCK):
                size += len(data); digest.update(data)
                if size > part['size_bytes']:
                    raise sub.SubmissionError('TRANSPORT_PART_MISMATCH', part['sha256'])
                yield data
        if size != part['size_bytes'] or digest.hexdigest() != part['sha256']:
            raise sub.SubmissionError('TRANSPORT_PART_MISMATCH', part['sha256'])


def verify(manifest, directory, *, sink=None):
    validate_manifest(manifest)
    expected = manifest['object']; size = 0; digest = hashlib.sha256()
    decoder = lzma.LZMADecompressor(format=lzma.FORMAT_XZ, memlimit=256 * BLOCK) if manifest['encoding'] == 'XZ_CHUNKS' else None

    def accept(data):
        nonlocal size
        size += len(data)
        if size > expected['size_bytes']:
            raise sub.SubmissionError('TRANSPORT_DECODED_SIZE_MISMATCH')
        digest.update(data)
        if sink is not None:
            sink.write(data)

    try:
        for data in _chunks(manifest, directory):
            if decoder is None:
                accept(data); continue
            if decoder.eof:
                raise sub.SubmissionError('TRANSPORT_TRAILING_DATA')
            while True:
                accept(decoder.decompress(data, max_length=min(BLOCK, expected['size_bytes'] - size + 1)))
                data = b''
                if decoder.eof:
                    if decoder.unused_data:
                        raise sub.SubmissionError('TRANSPORT_TRAILING_DATA')
                    break
                if decoder.needs_input:
                    break
        if decoder is not None and not decoder.eof:
            raise sub.SubmissionError('TRANSPORT_TRUNCATED')
    except lzma.LZMAError as exc:
        raise sub.SubmissionError('TRANSPORT_DECODE_FAILED', str(exc)) from exc
    if size != expected['size_bytes'] or digest.hexdigest() != expected['sha256']:
        raise sub.SubmissionError('TRANSPORT_OBJECT_MISMATCH')
    return {'sha256': digest.hexdigest(), 'size_bytes': size}


def make_receipt(obj, manifest_path, parts_directory, manifest_drive_id):
    if not _id(manifest_drive_id):
        raise sub.SubmissionError('DRIVE_FILE_ID_REQUIRED')
    raw, manifest = read_manifest(manifest_path)
    if manifest['object'] != {k: obj[k] for k in ('sha256', 'size_bytes')}:
        raise sub.SubmissionError('TRANSPORT_OBJECT_BINDING_MISMATCH')
    verify(manifest, parts_directory)
    identity_keys=('obligation_id','obligation_scope','role','logical_name')
    identity={key:obj[key] for key in identity_keys if key in obj}
    if identity and set(identity)!=set(identity_keys):
        raise sub.SubmissionError('TRANSPORT_OBLIGATION_IDENTITY_INCOMPLETE')
    receipt = {'schema_id': RECEIPT, 'provider': 'google_drive', **manifest['object'], **identity,
               'raw_readback_verified': False, 'object_bytes_verified': True,
               'transport': {'manifest_drive_file_id': manifest_drive_id,
                             'manifest_sha256': hashlib.sha256(raw).hexdigest(), 'manifest_size_bytes': len(raw),
                             'manifest': manifest, 'parts_readback_verified': True},
               'recorded_ns': time.time_ns(), 'scope': SCOPE}
    return dict(receipt, receipt_sha256=canonical_sha256(receipt))


def valid_receipt(row, obj):
    try:
        transport = row['transport']; manifest = validate_manifest(transport['manifest'])
        return (row['schema_id'] == RECEIPT and row['provider'] == 'google_drive'
                and row['sha256'] == obj['sha256'] and row['size_bytes'] == obj['size_bytes']
                and manifest['object'] == {k: obj[k] for k in ('sha256', 'size_bytes')}
                and row['raw_readback_verified'] is False and row['object_bytes_verified'] is True
                and transport['parts_readback_verified'] is True and _id(transport['manifest_drive_file_id'])
                and _object({'sha256': transport['manifest_sha256'], 'size_bytes': transport['manifest_size_bytes']})
                and 0 < transport['manifest_size_bytes'] <= MAX_MANIFEST_BYTES
                and canonical_sha256({k: v for k, v in row.items() if k != 'receipt_sha256'}) == row['receipt_sha256'])
    except (KeyError, TypeError, ValueError, sub.SubmissionError):
        return False


def pack(source, output, *, encoding='RAW_CHUNKS', chunk_bytes=16 * BLOCK):
    if encoding not in {'RAW_CHUNKS', 'XZ_CHUNKS'} or type(chunk_bytes) is not int or not 1 <= chunk_bytes <= MAX_PART_BYTES:
        raise sub.SubmissionError('TRANSPORT_PACK_OPTIONS')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    parts = []; buffer = bytearray(); digest = hashlib.sha256(); size = 0
    compressor = lzma.LZMACompressor(format=lzma.FORMAT_XZ, preset=6) if encoding == 'XZ_CHUNKS' else None

    def part(data):
        sha = hashlib.sha256(data).hexdigest()
        (output / sha).write_bytes(data)
        parts.append({'sha256': sha, 'size_bytes': len(data), 'drive_file_id': None})

    def feed(data):
        buffer.extend(data)
        while len(buffer) >= chunk_bytes:
            part(buffer[:chunk_bytes]); del buffer[:chunk_bytes]
    with Path(source).open('rb') as handle:
        while data := handle.read(BLOCK):
            digest.update(data); size += len(data)
            feed(compressor.compress(data) if compressor else data)
    if compressor:
        feed(compressor.flush())
    if buffer or not parts:
        part(buffer)
    manifest = {'schema_id': SCHEMA, 'object': {'sha256': digest.hexdigest(), 'size_bytes': size},
                'encoding': encoding, 'parts': parts}
    validate_manifest(manifest, bound=False)
    write_json_atomic(output / 'MANIFEST_DRAFT.json', manifest)
    return {'status': 'TRANSPORT_PREPARED_NOT_SAVED', 'manifest': manifest, 'directory': str(output.resolve())}


def bind(draft, mapping_path, output):
    _, manifest = read_manifest(draft, bound=False)
    mapping = sub._read(mapping_path)
    for part in manifest['parts']:
        part['drive_file_id'] = mapping.get(part['sha256'])
    validate_manifest(manifest)
    write_json_atomic(output, manifest)
    return {'status': 'TRANSPORT_BOUND_NOT_CONFIRMED', 'manifest': manifest, 'path': str(Path(output).resolve())}


def unpack(manifest_path, directory, output):
    _, manifest = read_manifest(manifest_path)
    output = Path(output)
    if output.exists():
        raise sub.SubmissionError('TRANSPORT_DESTINATION_EXISTS')
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix='.decoder-transport-', dir=output.parent)
    try:
        with os.fdopen(fd, 'wb') as sink:
            obj = verify(manifest, directory, sink=sink)
        # Exclusive creation prevents a concurrent writer being overwritten.
        os.link(temp, output)
        return {'status': 'TRANSPORT_OBJECT_VERIFIED', **obj, 'path': str(output.resolve())}
    finally:
        Path(temp).unlink(missing_ok=True)


def main(argv):
    parser = argparse.ArgumentParser(description='Bounded explicit Drive transport copies; no network operations.')
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('pack'); p.add_argument('source'); p.add_argument('output'); p.add_argument('--encoding', default='RAW_CHUNKS'); p.add_argument('--chunk-bytes', type=int, default=16 * BLOCK)
    p = commands.add_parser('bind'); p.add_argument('draft'); p.add_argument('mapping'); p.add_argument('output')
    p = commands.add_parser('unpack'); p.add_argument('manifest'); p.add_argument('parts'); p.add_argument('output')
    args = parser.parse_args(argv)
    try:
        if args.command == 'pack':
            result = pack(args.source, args.output, encoding=args.encoding, chunk_bytes=args.chunk_bytes)
        elif args.command == 'bind':
            result = bind(args.draft, args.mapping, args.output)
        else:
            result = unpack(args.manifest, args.parts, args.output)
        print(json.dumps(result, sort_keys=True, indent=2)); return 0
    except Exception as exc:
        from .invocation import refusal_details
        print(json.dumps(refusal_details(exc, 'transport ' + args.command), sort_keys=True, indent=2)); return 2
