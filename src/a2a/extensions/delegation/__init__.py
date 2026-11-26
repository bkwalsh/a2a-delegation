"""A2A delegation extension.

Carries a verifiable, monotonically narrowing delegation chain (RFC 8693) on
every A2A request. See ``docs/extensions/delegation/v1.md``.
"""

from a2a.extensions.delegation.card import (
    DELEGATION_EXTENSION_URI,
    DelegationParams,
    DelegationParamsError,
    delegation_extension,
    get_delegation_params,
)


__all__ = [
    'DELEGATION_EXTENSION_URI',
    'DelegationParams',
    'DelegationParamsError',
    'delegation_extension',
    'get_delegation_params',
]
