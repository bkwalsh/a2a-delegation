from typing import Any

import pytest

from a2a.extensions.delegation import (
    DELEGATION_EXTENSION_URI,
    DelegationParams,
    DelegationParamsError,
    delegation_extension,
    get_delegation_params,
)
from a2a.types.a2a_pb2 import AgentCapabilities, AgentCard, AgentExtension


def _params(**overrides: Any) -> DelegationParams:
    values: dict[str, Any] = {
        'issuer': 'https://sts.example.com',
        'token_endpoint': 'https://sts.example.com/token',
        'jwks_uri': 'https://sts.example.com/.well-known/jwks.json',
        'audience': 'agent:payments',
        'max_chain_depth': 2,
        'skill_scopes': {
            'refund': frozenset({'payments:refund'}),
            'balance': frozenset({'payments:read'}),
        },
    }
    values.update(overrides)
    return DelegationParams(**values)


def _card_with(*extensions: AgentExtension) -> AgentCard:
    return AgentCard(
        name='Payments',
        capabilities=AgentCapabilities(extensions=list(extensions)),
    )


def test_round_trip_through_agent_card() -> None:
    params = _params(
        trusted_actors=frozenset({'agent:orchestrator', 'agent:travel'}),
        dpop_required=True,
    )
    card = _card_with(delegation_extension(params))

    assert get_delegation_params(card) == params


def test_extension_is_required_by_default() -> None:
    ext = delegation_extension(_params())

    assert ext.uri == DELEGATION_EXTENSION_URI
    assert ext.required is True
    assert delegation_extension(_params(), required=False).required is False


def test_trusted_actors_omitted_when_unset() -> None:
    params = _params()

    assert 'trusted_actors' not in params.to_dict()
    assert get_delegation_params(_card_with(delegation_extension(params))) == (
        params
    )


def test_card_without_extension_returns_none() -> None:
    other = AgentExtension(uri='https://example.com/other')

    assert get_delegation_params(_card_with(other)) is None
    assert get_delegation_params(AgentCard(name='bare')) is None


def test_all_scopes_is_union_of_skill_scopes() -> None:
    assert _params().all_scopes == {'payments:refund', 'payments:read'}


@pytest.mark.parametrize(
    ('overrides', 'message'),
    [
        ({'issuer': ''}, 'issuer is required'),
        ({'audience': ''}, 'audience is required'),
        ({'max_chain_depth': -1}, 'max_chain_depth must be >= 0'),
        ({'skill_scopes': {}}, 'skill_scopes must not be empty'),
        (
            {'skill_scopes': {'refund': frozenset()}},
            "skill 'refund' must require at least one scope",
        ),
    ],
)
def test_invalid_params_rejected(
    overrides: dict[str, Any], message: str
) -> None:
    with pytest.raises(DelegationParamsError, match=message):
        _params(**overrides)


@pytest.mark.parametrize(
    ('mutate', 'message'),
    [
        (lambda d: d.pop('jwks_uri'), 'jwks_uri is required'),
        (
            lambda d: d.update(issuer=42),
            'issuer must be a string',
        ),
        (
            lambda d: d.update(max_chain_depth='2'),
            'max_chain_depth must be a number',
        ),
        (
            lambda d: d.update(max_chain_depth=True),
            'max_chain_depth must be a number',
        ),
        (
            lambda d: d.update(max_chain_depth=1.5),
            'max_chain_depth must be integral',
        ),
        (
            lambda d: d.update(skill_scopes=['refund']),
            'skill_scopes must be an object',
        ),
        (
            lambda d: d.update(skill_scopes={'refund': 'payments:refund'}),
            'skill_scopes must be a list of strings',
        ),
        (
            lambda d: d.update(trusted_actors=[1]),
            'trusted_actors must be a list of strings',
        ),
    ],
)
def test_malformed_card_params_rejected(mutate: Any, message: str) -> None:
    data = _params().to_dict()
    mutate(data)

    with pytest.raises(DelegationParamsError, match=message):
        DelegationParams.from_dict(data)


def test_float_depth_from_struct_is_accepted() -> None:
    data = _params().to_dict()
    data['max_chain_depth'] = 2.0

    assert DelegationParams.from_dict(data).max_chain_depth == 2
