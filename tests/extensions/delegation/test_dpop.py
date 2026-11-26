import time

from typing import Any

import httpx
import jwt
import pytest

from a2a.extensions.common import HTTP_EXTENSION_HEADER
from a2a.extensions.delegation import DELEGATION_EXTENSION_URI, DelegationParams
from a2a.extensions.delegation.client import TokenExchangeClient
from a2a.extensions.delegation.dpop import (
    DPoPError,
    DPoPKey,
    DPoPSigner,
    ReplayCache,
    access_token_hash,
    normalize_htu,
    verify_proof,
)
from a2a.extensions.delegation.server import StaticKeySource
from a2a.extensions.delegation.sts import (
    DelegationTokenService,
    basic_auth_header,
    create_sts_app,
)
from a2a.extensions.delegation.tokens import (
    TOKEN_EXCHANGE_GRANT,
    verify_token,
)

from .agents import SkillExecutor, build_agent_app
from .conftest import ISSUER, SECRETS


URL = 'http://payments.test/'


@pytest.fixture
def key() -> DPoPKey:
    return DPoPKey.generate()


def _verify(proof: str, **kw: Any) -> str:
    return verify_proof(
        proof,
        method=kw.pop('method', 'POST'),
        url=kw.pop('url', URL),
        replay_cache=kw.pop('cache', ReplayCache()),
        **kw,
    ).jkt


# --- proofs ------------------------------------------------------------------


def test_valid_proof(key: DPoPKey) -> None:
    proof = key.proof('post', URL + '?q=1#f', access_token='tok')

    assert _verify(proof, access_token='tok', expected_jkt=key.thumbprint) == (
        key.thumbprint
    )
    claims = jwt.decode(proof, options={'verify_signature': False})
    assert claims['htm'] == 'POST'
    assert claims['htu'] == URL
    assert claims['ath'] == access_token_hash('tok')


@pytest.mark.parametrize(
    ('url', 'expected'),
    [
        ('https://a.test/x?y=1#z', 'https://a.test/x'),
        ('http://travel.demo', 'http://travel.demo/'),
        ('HTTP://Travel.Demo:80/', 'http://travel.demo/'),
        ('https://a.test:443/p', 'https://a.test/p'),
        ('https://a.test:8443/p', 'https://a.test:8443/p'),
    ],
)
def test_normalize_htu(url: str, expected: str) -> None:
    assert normalize_htu(url) == expected


def test_proof_for_url_without_trailing_slash_verifies(key: DPoPKey) -> None:
    # Clients often call 'http://host' while servers see 'http://host/'.
    proof = key.proof('POST', 'http://payments.test')

    assert _verify(proof, url='http://payments.test/') == key.thumbprint


@pytest.mark.parametrize(
    ('kwargs', 'message'),
    [
        ({'method': 'GET'}, 'htm'),
        ({'url': 'http://other.test/'}, 'htu'),
        ({'access_token': 'other'}, 'ath'),
        ({'expected_jkt': 'not-my-key'}, 'binding'),
        ({'now': time.time() + 120}, 'fresh'),
        ({'now': time.time() - 120}, 'fresh'),
    ],
)
def test_proof_mismatches_rejected(
    key: DPoPKey, kwargs: dict[str, Any], message: str
) -> None:
    proof = key.proof('POST', URL, access_token='tok')
    kwargs.setdefault('access_token', 'tok')

    with pytest.raises(DPoPError, match=message):
        _verify(proof, **kwargs)


def test_replayed_proof_rejected(key: DPoPKey) -> None:
    cache = ReplayCache()
    proof = key.proof('POST', URL)

    _verify(proof, cache=cache)
    with pytest.raises(DPoPError, match='already been used'):
        _verify(proof, cache=cache)


def test_failed_proof_does_not_burn_jti(key: DPoPKey) -> None:
    cache = ReplayCache()
    proof = key.proof('POST', URL)

    with pytest.raises(DPoPError):
        _verify(proof, cache=cache, method='GET')
    assert _verify(proof, cache=cache) == key.thumbprint


def _forge(
    headers: dict[str, Any], claims: dict[str, Any] | None = None
) -> str:
    signer = DPoPKey.generate()
    body = claims or {
        'jti': 'j',
        'htm': 'POST',
        'htu': URL,
        'iat': int(time.time()),
    }
    return jwt.encode(
        body, signer.private_key, algorithm='ES256', headers=headers
    )


@pytest.mark.parametrize(
    ('headers', 'message'),
    [
        ({'typ': 'JWT'}, 'typ'),
        ({'typ': 'dpop+jwt', 'jwk': {'kty': 'RSA'}}, 'public EC jwk'),
        ({'typ': 'dpop+jwt'}, 'public EC jwk'),
    ],
)
def test_malformed_proof_headers(
    key: DPoPKey, headers: dict[str, Any], message: str
) -> None:
    with pytest.raises(DPoPError, match=message):
        _verify(_forge(headers))


def test_private_key_in_jwk_rejected(key: DPoPKey) -> None:
    jwk = {**key.public_jwk, 'd': 'secret'}

    with pytest.raises(DPoPError, match='public EC jwk'):
        _verify(_forge({'typ': 'dpop+jwt', 'jwk': jwk}))


def test_signature_by_other_key_rejected(key: DPoPKey) -> None:
    # Header claims key A, but the proof is signed by key B.
    with pytest.raises(DPoPError, match='signature'):
        _verify(_forge({'typ': 'dpop+jwt', 'jwk': key.public_jwk}))


def test_garbage_and_wrong_alg_rejected(key: DPoPKey) -> None:
    with pytest.raises(DPoPError, match='malformed'):
        _verify('garbage')
    hs = jwt.encode(
        {'jti': 'j'},
        'k' * 32,
        algorithm='HS256',
        headers={'typ': 'dpop+jwt', 'jwk': key.public_jwk},
    )
    with pytest.raises(DPoPError, match='not allowed'):
        _verify(hs)


def test_non_numeric_iat_rejected(key: DPoPKey) -> None:
    proof = jwt.encode(
        {'jti': 'j', 'htm': 'POST', 'htu': URL, 'iat': True},
        key.private_key,
        algorithm='ES256',
        headers={'typ': 'dpop+jwt', 'jwk': key.public_jwk},
    )
    with pytest.raises(DPoPError, match='iat'):
        _verify(proof)


def test_replay_cache_evicts_expired_and_bounds_size() -> None:
    cache = ReplayCache(max_entries=2)

    assert cache.check_and_add('a', 10, now=0)
    assert cache.check_and_add('b', 10, now=0)
    assert cache.check_and_add('c', 10, now=0)  # evicts 'a' (size bound)
    assert cache.check_and_add('a', 10, now=0)
    assert cache.check_and_add('b', 30, now=20)  # 'b' had expired


@pytest.mark.asyncio
async def test_signer_adds_proof_only_for_dpop_requests(key: DPoPKey) -> None:
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(record), auth=DPoPSigner(key)
    ) as client:
        await client.post(URL, headers={'Authorization': 'DPoP tok'})
        await client.post(URL, headers={'Authorization': 'Bearer tok'})

    assert _verify(
        seen[0].headers['DPoP'], access_token='tok', expected_jkt=key.thumbprint
    )
    assert 'DPoP' not in seen[1].headers


# --- binding at the STS ------------------------------------------------------


def _sts_client(sts: DelegationTokenService) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_sts_app(sts)),
        base_url=ISSUER,
    )


def _root(sts: DelegationTokenService) -> str:
    return sts.issue_root_token(
        subject='user:bayard',
        audience='agent:orchestrator',
        scopes={'payments:refund', 'payments:read'},
    ).access_token


@pytest.mark.asyncio
async def test_exchange_with_proof_binds_token(
    sts: DelegationTokenService, key: DPoPKey
) -> None:
    exchanger = TokenExchangeClient(
        issuer=ISSUER,
        token_endpoint=f'{ISSUER}/token',
        client_id='agent:orchestrator',
        client_secret=SECRETS['agent:orchestrator'],
        http=_sts_client(sts),
        dpop_key=key,
    )

    token = await exchanger.exchange(_root(sts), 'agent:payments')

    assert token.token_type == 'DPoP'
    claims = verify_token(
        token.access_token, sts.keys, issuer=ISSUER, audience='agent:payments'
    )
    assert claims['cnf'] == {'jkt': key.thumbprint}


@pytest.mark.asyncio
async def test_exchange_rejects_bad_or_multiple_proofs(
    sts: DelegationTokenService, key: DPoPKey
) -> None:
    form = {
        'grant_type': TOKEN_EXCHANGE_GRANT,
        'subject_token': _root(sts),
        'audience': 'agent:payments',
    }
    auth = basic_auth_header(
        'agent:orchestrator', SECRETS['agent:orchestrator']
    )
    async with _sts_client(sts) as http:
        wrong_url = await http.post(
            '/token',
            data=form,
            headers={
                'Authorization': auth,
                'DPoP': key.proof('POST', 'https://elsewhere/token'),
            },
        )
        two = await http.post(
            '/token',
            data=form,
            headers=[
                ('Authorization', auth),
                ('DPoP', key.proof('POST', f'{ISSUER}/token')),
                ('DPoP', key.proof('POST', f'{ISSUER}/token')),
            ],
        )

    for r in (wrong_url, two):
        assert (r.status_code, r.json()['error']) == (400, 'invalid_dpop_proof')


@pytest.mark.asyncio
async def test_exchanger_requires_sts_to_bind(key: DPoPKey) -> None:
    def unbound(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                'access_token': 't',
                'token_type': 'Bearer',
                'expires_in': 60,
            },
        )

    exchanger = TokenExchangeClient(
        issuer=ISSUER,
        token_endpoint=f'{ISSUER}/token',
        client_id='agent:orchestrator',
        client_secret='s',
        http=httpx.AsyncClient(transport=httpx.MockTransport(unbound)),
        dpop_key=key,
    )
    with pytest.raises(Exception, match='invalid_token_type'):
        await exchanger.exchange('subject', 'agent:payments')


# --- enforcement at the receiving agent --------------------------------------


def _params(*, dpop_required: bool) -> DelegationParams:
    return DelegationParams(
        issuer=ISSUER,
        token_endpoint=f'{ISSUER}/token',
        jwks_uri=f'{ISSUER}/.well-known/jwks.json',
        audience='agent:payments',
        max_chain_depth=1,
        skill_scopes={'refund': frozenset({'payments:refund'})},
        dpop_required=dpop_required,
    )


def _bound_token(sts: DelegationTokenService, jkt: str | None) -> str:
    return sts.exchange(
        actor_id='agent:orchestrator',
        actor_secret=SECRETS['agent:orchestrator'],
        subject_token=_root(sts),
        audience='agent:payments',
        dpop_jkt=jkt,
    ).access_token


async def _call(
    sts: DelegationTokenService,
    headers: list[tuple[str, str]],
    *,
    dpop_required: bool = True,
) -> tuple[httpx.Response, SkillExecutor]:
    app, executor = build_agent_app(
        _params(dpop_required=dpop_required),
        StaticKeySource(sts.keys),
        URL.rstrip('/'),
    )
    assert isinstance(executor, SkillExecutor)
    body = {
        'jsonrpc': '2.0',
        'id': 1,
        'method': 'SendMessage',
        'params': {
            'message': {
                'messageId': 'm',
                'role': 'ROLE_USER',
                'parts': [{'text': 'refund'}],
            }
        },
    }
    base = [
        ('A2A-Version', '1.0'),
        (HTTP_EXTENSION_HEADER, DELEGATION_EXTENSION_URI),
    ]
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url=URL
    ) as http:
        return await http.post('/', json=body, headers=base + headers), executor


@pytest.mark.asyncio
async def test_bound_token_with_valid_proof_is_accepted(
    sts: DelegationTokenService, key: DPoPKey
) -> None:
    token = _bound_token(sts, key.thumbprint)

    r, executor = await _call(
        sts,
        [
            ('Authorization', f'DPoP {token}'),
            ('DPoP', key.proof('POST', URL, access_token=token)),
        ],
    )

    assert r.status_code == 200, r.text
    assert len(executor.principals) == 1


@pytest.mark.asyncio
async def test_stolen_bound_token_with_attackers_own_proof(
    sts: DelegationTokenService, key: DPoPKey
) -> None:
    token = _bound_token(sts, key.thumbprint)
    attacker = DPoPKey.generate()

    r, executor = await _call(
        sts,
        [
            ('Authorization', f'DPoP {token}'),
            ('DPoP', attacker.proof('POST', URL, access_token=token)),
        ],
    )

    assert (r.status_code, r.json()['error']) == (401, 'invalid_dpop_proof')
    assert 'binding' in r.json()['error_description']
    assert r.headers['www-authenticate'].startswith('DPoP ')
    assert executor.principals == []


@pytest.mark.asyncio
@pytest.mark.parametrize('dpop_required', [True, False])
async def test_bound_token_sent_as_bearer_rejected(
    sts: DelegationTokenService, key: DPoPKey, dpop_required: bool
) -> None:
    token = _bound_token(sts, key.thumbprint)

    r, _ = await _call(
        sts,
        [('Authorization', f'Bearer {token}')],
        dpop_required=dpop_required,
    )

    assert r.json()['error'] == 'invalid_dpop_proof'


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('scheme', 'proofs', 'dpop_required', 'message'),
    [
        ('Bearer', 0, True, 'not DPoP-bound'),
        ('DPoP', 0, False, 'unbound token'),
    ],
)
async def test_unbound_token_rules(
    sts: DelegationTokenService,
    scheme: str,
    proofs: int,
    dpop_required: bool,
    message: str,
) -> None:
    token = _bound_token(sts, None)

    r, _ = await _call(
        sts,
        [('Authorization', f'{scheme} {token}')],
        dpop_required=dpop_required,
    )

    assert r.json()['error'] == 'invalid_dpop_proof'
    assert message in r.json()['error_description']


@pytest.mark.asyncio
async def test_missing_proof_rejected(
    sts: DelegationTokenService, key: DPoPKey
) -> None:
    token = _bound_token(sts, key.thumbprint)

    r, _ = await _call(sts, [('Authorization', f'DPoP {token}')])

    assert 'exactly one' in r.json()['error_description']


def test_invalid_ec_point_in_jwk_rejected() -> None:
    bad = {'kty': 'EC', 'crv': 'P-256', 'x': 'AA', 'y': 'AA'}

    with pytest.raises(DPoPError, match='public EC jwk'):
        _verify(_forge({'typ': 'dpop+jwt', 'jwk': bad}))
