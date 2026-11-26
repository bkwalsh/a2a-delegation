import base64

from typing import Any

import httpx
import pytest

from a2a.extensions.delegation.sts import (
    DelegationTokenService,
    TokenExchangeError,
    basic_auth_header,
    create_sts_app,
    parse_basic_auth,
)
from a2a.extensions.delegation.tokens import (
    TOKEN_EXCHANGE_GRANT,
    SigningKey,
    actor_chain,
    scopes_of,
    verify_token,
)

from .conftest import ISSUER, SECRETS, Clock


def _root(sts: DelegationTokenService, scopes: set[str] | None = None) -> str:
    return sts.issue_root_token(
        subject='user:bayard',
        audience='agent:orchestrator',
        scopes=scopes or {'travel:book', 'payments:refund', 'payments:read'},
    ).access_token


def _exchange(
    sts: DelegationTokenService,
    token: str,
    actor: str,
    audience: str,
    scope: str | None = None,
) -> str:
    return sts.exchange(
        actor_id=actor,
        actor_secret=SECRETS[actor],
        subject_token=token,
        audience=audience,
        scope=scope,
    ).access_token


def _claims(
    sts: DelegationTokenService, token: str, aud: str
) -> dict[str, Any]:
    return verify_token(
        token, sts.keys, issuer=ISSUER, audience=aud, now=sts._clock()
    )


def test_root_token_has_no_chain(sts: DelegationTokenService) -> None:
    claims = _claims(sts, _root(sts), 'agent:orchestrator')

    assert claims['sub'] == 'user:bayard'
    assert actor_chain(claims) == []


def test_two_hop_chain(sts: DelegationTokenService) -> None:
    t1 = _exchange(sts, _root(sts), 'agent:orchestrator', 'agent:travel')
    t2 = _exchange(sts, t1, 'agent:travel', 'agent:payments')

    claims = _claims(sts, t2, 'agent:payments')
    assert claims['sub'] == 'user:bayard'
    assert actor_chain(claims) == ['agent:travel', 'agent:orchestrator']
    # travel only accepts travel:book + payments:refund, so payments:read
    # was dropped at hop 1 and cannot come back at hop 2.
    assert scopes_of(claims) == {'payments:refund'}


def test_scopes_can_be_narrowed(sts: DelegationTokenService) -> None:
    t1 = _exchange(
        sts,
        _root(sts),
        'agent:orchestrator',
        'agent:payments',
        scope='payments:read',
    )

    assert scopes_of(_claims(sts, t1, 'agent:payments')) == {'payments:read'}


def test_scope_escalation_rejected(sts: DelegationTokenService) -> None:
    root = _root(sts, {'travel:book'})

    with pytest.raises(TokenExchangeError) as e:
        _exchange(
            sts,
            root,
            'agent:orchestrator',
            'agent:travel',
            scope='travel:book payments:refund',
        )
    assert e.value.error == 'invalid_scope'
    assert 'payments:refund' in e.value.description


def test_no_overlap_with_audience_scopes_rejected(
    sts: DelegationTokenService,
) -> None:
    with pytest.raises(TokenExchangeError) as e:
        _exchange(
            sts,
            _root(sts, {'travel:book'}),
            'agent:orchestrator',
            'agent:payments',
        )
    assert e.value.error == 'invalid_scope'


def test_stolen_token_cannot_be_exchanged_by_another_agent(
    sts: DelegationTokenService,
) -> None:
    # The root token was issued to the orchestrator; travel cannot use it.
    with pytest.raises(TokenExchangeError) as e:
        _exchange(sts, _root(sts), 'agent:travel', 'agent:payments')
    assert e.value.error == 'invalid_grant'
    assert 'not issued to the requesting agent' in e.value.description


def test_wrong_secret_rejected(sts: DelegationTokenService) -> None:
    with pytest.raises(TokenExchangeError) as e:
        sts.exchange(
            actor_id='agent:orchestrator',
            actor_secret='guess',
            subject_token=_root(sts),
            audience='agent:travel',
        )
    assert (e.value.error, e.value.status) == ('invalid_client', 401)


def test_unknown_actor_rejected(sts: DelegationTokenService) -> None:
    with pytest.raises(TokenExchangeError) as e:
        sts.exchange(
            actor_id='agent:nobody',
            actor_secret='x',
            subject_token=_root(sts),
            audience='agent:travel',
        )
    assert e.value.error == 'invalid_client'


def test_chain_depth_limit(sts: DelegationTokenService) -> None:
    t1 = _exchange(sts, _root(sts), 'agent:orchestrator', 'agent:travel')
    t2 = _exchange(sts, t1, 'agent:travel', 'agent:payments')

    with pytest.raises(TokenExchangeError, match='too deep'):
        _exchange(sts, t2, 'agent:payments', 'agent:ledger')


@pytest.mark.parametrize(
    ('actor', 'audience', 'message'),
    [
        ('agent:orchestrator', 'agent:orchestrator', 'itself'),
        ('agent:orchestrator', 'agent:unknown', 'unknown audience'),
    ],
)
def test_invalid_targets(
    sts: DelegationTokenService, actor: str, audience: str, message: str
) -> None:
    with pytest.raises(TokenExchangeError) as e:
        _exchange(sts, _root(sts), actor, audience)
    assert e.value.error == 'invalid_target'
    assert message in e.value.description


def test_cycle_rejected(sts: DelegationTokenService) -> None:
    t1 = _exchange(sts, _root(sts), 'agent:orchestrator', 'agent:travel')

    with pytest.raises(TokenExchangeError, match='only once'):
        _exchange(sts, t1, 'agent:travel', 'agent:orchestrator')


def test_unsupported_subject_token_type(sts: DelegationTokenService) -> None:
    with pytest.raises(TokenExchangeError) as e:
        sts.exchange(
            actor_id='agent:orchestrator',
            actor_secret=SECRETS['agent:orchestrator'],
            subject_token=_root(sts),
            subject_token_type='urn:ietf:params:oauth:token-type:id_token',
            audience='agent:travel',
        )
    assert e.value.error == 'invalid_request'


def test_expired_subject_cannot_be_exchanged(
    sts: DelegationTokenService, clock: Clock
) -> None:
    root = _root(sts)
    clock.now += 301

    with pytest.raises(TokenExchangeError) as e:
        _exchange(sts, root, 'agent:orchestrator', 'agent:travel')
    assert e.value.error == 'invalid_grant'


def test_exchanged_token_never_outlives_subject(
    sts: DelegationTokenService, clock: Clock
) -> None:
    root = _root(sts)
    root_exp = _claims(sts, root, 'agent:orchestrator')['exp']
    clock.now += 200

    issued = sts.exchange(
        actor_id='agent:orchestrator',
        actor_secret=SECRETS['agent:orchestrator'],
        subject_token=root,
        audience='agent:travel',
    )

    assert issued.expires_in == 100
    assert _claims(sts, issued.access_token, 'agent:travel')['exp'] == root_exp


def test_token_from_other_issuer_key_rejected(
    sts: DelegationTokenService,
) -> None:
    forged = DelegationTokenService(
        issuer=ISSUER,
        signing_key=SigningKey.generate(),
        agents=list(sts._agents.values()),
    )
    with pytest.raises(TokenExchangeError) as e:
        _exchange(sts, _root(forged), 'agent:orchestrator', 'agent:travel')
    assert e.value.error == 'invalid_grant'


def test_root_token_requires_accepted_scope(
    sts: DelegationTokenService,
) -> None:
    with pytest.raises(TokenExchangeError, match='no requested scope'):
        sts.issue_root_token(
            subject='user:bayard',
            audience='agent:payments',
            scopes={'travel:book'},
        )


def test_basic_auth_round_trips_ids_with_colons() -> None:
    header = basic_auth_header('agent:travel', 's:e c&ret')

    assert parse_basic_auth(header) == ('agent:travel', 's:e c&ret')


def test_parse_basic_auth() -> None:
    raw = 'Basic ' + base64.b64encode(b'agent%3Aa:s%3Aec').decode()

    assert parse_basic_auth(raw) == ('agent:a', 's:ec')
    assert parse_basic_auth(None) is None
    assert parse_basic_auth('Bearer x') is None
    assert parse_basic_auth('Basic !!!') is None
    assert (
        parse_basic_auth('Basic ' + base64.b64encode(b'nocolon').decode())
        is None
    )


# --- HTTP endpoint -----------------------------------------------------------


def _client(sts: DelegationTokenService) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_sts_app(sts)),
        base_url='https://sts.example.com',
    )


def _auth(agent: str, secret: str | None = None) -> dict[str, str]:
    return {'Authorization': basic_auth_header(agent, secret or SECRETS[agent])}


def _form(sts: DelegationTokenService, **overrides: str) -> dict[str, str]:
    form = {
        'grant_type': TOKEN_EXCHANGE_GRANT,
        'subject_token': _root(sts),
        'audience': 'agent:travel',
    }
    form.update(overrides)
    return {k: v for k, v in form.items() if v}


@pytest.mark.asyncio
async def test_http_exchange(sts: DelegationTokenService) -> None:
    async with _client(sts) as client:
        r = await client.post(
            '/token',
            data=_form(sts),
            headers=_auth('agent:orchestrator'),
        )

    assert r.status_code == 200
    assert r.headers['cache-control'] == 'no-store'
    body = r.json()
    assert body['token_type'] == 'Bearer'
    assert body['issued_token_type'].endswith('access_token')
    claims = _claims(sts, body['access_token'], 'agent:travel')
    assert actor_chain(claims) == ['agent:orchestrator']


@pytest.mark.asyncio
async def test_http_rejects_non_form_body(
    sts: DelegationTokenService,
) -> None:
    async with _client(sts) as client:
        r = await client.post(
            '/token', json=_form(sts), headers=_auth('agent:orchestrator')
        )

    assert (r.status_code, r.json()['error']) == (400, 'invalid_request')


@pytest.mark.asyncio
async def test_http_jwks(sts: DelegationTokenService) -> None:
    async with _client(sts) as client:
        r = await client.get('/.well-known/jwks.json')

    assert r.json() == sts.jwks()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('auth', 'overrides', 'status', 'error'),
    [
        (None, {}, 401, 'invalid_client'),
        (('agent:orchestrator', 'wrong'), {}, 401, 'invalid_client'),
        (
            ('agent:orchestrator', None),
            {'grant_type': 'client_credentials'},
            400,
            'unsupported_grant_type',
        ),
        (
            ('agent:orchestrator', None),
            {'audience': ''},
            400,
            'invalid_request',
        ),
        (
            ('agent:travel', None),
            {'audience': 'agent:payments'},
            400,
            'invalid_grant',
        ),
    ],
)
async def test_http_errors(
    sts: DelegationTokenService,
    auth: tuple[str, str | None] | None,
    overrides: dict[str, str],
    status: int,
    error: str,
) -> None:
    headers = _auth(*auth) if auth else {}
    async with _client(sts) as client:
        r = await client.post(
            '/token', data=_form(sts, **overrides), headers=headers
        )

    assert r.status_code == status
    assert r.json()['error'] == error
    if status == 401:
        assert r.headers['www-authenticate'].startswith('Basic')
