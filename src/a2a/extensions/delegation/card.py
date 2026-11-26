"""Agent Card declaration of the delegation extension.

See ``docs/extensions/delegation/v1.md`` §3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from google.protobuf import struct_pb2
from google.protobuf.json_format import MessageToDict

from a2a.extensions.common import find_extension_by_uri
from a2a.types.a2a_pb2 import AgentCard, AgentExtension


DELEGATION_EXTENSION_URI = 'https://github.com/bkwalsh/a2a-delegation/blob/main/docs/extensions/delegation/v1.md'


class DelegationParamsError(ValueError):
    """Raised when delegation extension params are missing or malformed."""


@dataclass(frozen=True)
class DelegationParams:
    """Policy an agent declares for the delegation extension.

    Attributes:
        issuer: The only token issuer (``iss``) the agent accepts.
        token_endpoint: Where callers perform the RFC 8693 token exchange.
        jwks_uri: The issuer's JSON Web Key Set.
        audience: The exact ``aud`` value the agent accepts.
        max_chain_depth: Maximum number of ``act`` levels in a token.
        skill_scopes: Skill id to the scopes required to invoke that skill.
        trusted_actors: If set, every actor in the chain must be listed.
        dpop_required: Whether tokens must be DPoP-bound (RFC 9449).
    """

    issuer: str
    token_endpoint: str
    jwks_uri: str
    audience: str
    max_chain_depth: int
    skill_scopes: dict[str, frozenset[str]] = field(default_factory=dict)
    trusted_actors: frozenset[str] | None = None
    dpop_required: bool = False

    def __post_init__(self) -> None:
        """Validates the params."""
        for name in ('issuer', 'token_endpoint', 'jwks_uri', 'audience'):
            if not getattr(self, name):
                raise DelegationParamsError(f'{name} is required')
        if self.max_chain_depth < 0:
            raise DelegationParamsError('max_chain_depth must be >= 0')
        if not self.skill_scopes:
            raise DelegationParamsError('skill_scopes must not be empty')
        for skill_id, scopes in self.skill_scopes.items():
            if not scopes:
                raise DelegationParamsError(
                    f'skill {skill_id!r} must require at least one scope'
                )

    @property
    def all_scopes(self) -> frozenset[str]:
        """Every scope that grants access to at least one skill."""
        return frozenset().union(*self.skill_scopes.values())

    def to_dict(self) -> dict[str, Any]:
        """Serializes the params to the JSON shape used in the Agent Card."""
        result: dict[str, Any] = {
            'issuer': self.issuer,
            'token_endpoint': self.token_endpoint,
            'jwks_uri': self.jwks_uri,
            'audience': self.audience,
            'max_chain_depth': self.max_chain_depth,
            'skill_scopes': {
                skill: sorted(scopes)
                for skill, scopes in sorted(self.skill_scopes.items())
            },
            'dpop_required': self.dpop_required,
        }
        if self.trusted_actors is not None:
            result['trusted_actors'] = sorted(self.trusted_actors)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DelegationParams:
        """Parses params from the Agent Card JSON shape.

        Raises:
            DelegationParamsError: If a field is missing or has the wrong type.
        """
        try:
            depth = data['max_chain_depth']
            # protobuf Struct stores every number as a float.
            if isinstance(depth, bool) or not isinstance(depth, int | float):
                raise DelegationParamsError('max_chain_depth must be a number')
            if depth != int(depth):
                raise DelegationParamsError('max_chain_depth must be integral')
            skill_scopes = data['skill_scopes']
            if not isinstance(skill_scopes, dict):
                raise DelegationParamsError('skill_scopes must be an object')
            trusted = data.get('trusted_actors')
            return cls(
                issuer=_require_str(data, 'issuer'),
                token_endpoint=_require_str(data, 'token_endpoint'),
                jwks_uri=_require_str(data, 'jwks_uri'),
                audience=_require_str(data, 'audience'),
                max_chain_depth=int(depth),
                skill_scopes={
                    str(skill): frozenset(_str_list(scopes, 'skill_scopes'))
                    for skill, scopes in skill_scopes.items()
                },
                trusted_actors=(
                    None
                    if trusted is None
                    else frozenset(_str_list(trusted, 'trusted_actors'))
                ),
                dpop_required=bool(data.get('dpop_required', False)),
            )
        except KeyError as e:
            raise DelegationParamsError(f'{e.args[0]} is required') from e


def _require_str(data: dict[str, Any], key: str) -> str:
    value = data[key]
    if not isinstance(value, str):
        raise DelegationParamsError(f'{key} must be a string')
    return value


def _str_list(value: Any, name: str) -> list[str]:
    if not isinstance(value, list) or not all(
        isinstance(v, str) for v in value
    ):
        raise DelegationParamsError(f'{name} must be a list of strings')
    return value


def delegation_extension(
    params: DelegationParams, *, required: bool = True
) -> AgentExtension:
    """Builds the AgentExtension entry declaring delegation for an agent."""
    struct = struct_pb2.Struct()
    struct.update(params.to_dict())
    return AgentExtension(
        uri=DELEGATION_EXTENSION_URI,
        description=(
            'Requests must carry a delegation token (RFC 8693) whose chain '
            'and scopes this agent verifies before execution.'
        ),
        required=required,
        params=struct,
    )


def get_delegation_params(card: AgentCard) -> DelegationParams | None:
    """Returns the delegation params declared on a card, if any.

    Raises:
        DelegationParamsError: If the extension is declared but malformed.
    """
    ext = find_extension_by_uri(card, DELEGATION_EXTENSION_URI)
    if ext is None:
        return None
    return DelegationParams.from_dict(MessageToDict(ext.params))
