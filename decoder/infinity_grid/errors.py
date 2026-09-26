from __future__ import annotations


class DecoderError(RuntimeError):
    """Base class for fail-closed Decoder operational errors."""


class ContractError(DecoderError):
    """A persisted-object contract could not be satisfied."""


class SchemaValidationError(ContractError, ValueError):
    """A document failed its declared strict schema."""


class UnknownSchemaError(ContractError):
    """No contract decision exists for a declared payload schema."""


class UnsupportedArtifactVersion(ContractError):
    """The artifact/envelope major version is not supported by this runtime."""


class ArtifactCorruptError(ContractError):
    """An artifact digest or dependency binding does not match its bytes/content."""


class MigrationError(ContractError):
    """A deterministic migration could not be completed."""


class StateConflictError(ContractError):
    """Multiple state sources disagree and cannot be reconciled without an explicit rule."""
