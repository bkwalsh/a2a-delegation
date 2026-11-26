"""Production hardening: shared replay store, private_key_jwt, proxies."""

import time

from typing import Any

import httpx
import jwt
import pytest

from a2a.extensions.common import HTTP_EXTENSION_HEADER
from a2a.extensions.delegation import DELEGATION_EXTENSION_URI, DelegationParams
from a2a.extensions.delegation.client import TokenExchangeClient
from a2a.extensions.delegation.dpop import (
    DPoPKey,
    RedisReplayCache,
    ReplayCache,
)
from a2a.extensions.delegation.server import (
    DelegationAuthBackend,
    StaticKeySource,
)
from a2a.extensions.delegation.sts import (
    CLIENT_ASSERTION_TYPE,
    DelegationTokenService,
    RegisteredAgent,
    TokenExchangeError,
    basic_auth_header,
    create_sts_app,
)
from a2a.extensions.delegation.tokens import (
    TOKEN_EXCHANGE_GRANT,
    SigningKey,
    actor_chain,
    verify_token,
)

from .agents import build_agent_app
from .conftest import AGENTS, ISSUER, SECRETS, Clock


TOKEN_URL = f'{ISSUER}/token'


class FakeRedis:
    """Implements SET NX EX like redis.asyncio, with a controllable clock."""

    def __init__(self, clock: Clock) -> None:
        self._data: dict[str, float] = {}
        self._clock = clock
        self.calls: list[tuple[str, int]] = []

    async def set(
        self, name: str, value: str, *, nx: bool, ex: int
    ) -> bool | None:
        assert nx
        self.calls.append((name, ex))
        expiry = self._data.get(name)
        if expiry is not None and expiry > self._clock():
            return None
        self._data[name] = self._clock() + ex
        return True


# --- shared replay store -----------------------------------------------------


@pytest.mark.asyncio
async def test_redis_replay_cache_semantics(clock: Clock) -> None:
    redis = FakeRedis(clock)
    cache = RedisReplayCache(redis, prefix='t:')

    assert await cache.check_and_add('j', clock() + 60.2, clock())
    assert not await cache.check_and_add('j', clock() + 60, clock())
    assert redis.calls[0] == ('t:j', 61)  # TTL rounds up, key is prefixed
    clock.now += 62
    assert await cache.check_and_add('j', clock() + 60, clock())
    # A ttl never drops below one second.
    await cache.check_and_add('k', clock() - 5, clock())
    assert redis.calls[-1] == ('t:k', 1)


def _params() -> DelegationParams:
    return DelegationParams(
        issuer=ISSUER,
        token_endpoint=TOKEN_URL,
        jwks_uri=f'{ISSUER}/.well-known/jwks.json',
        audience='agent:payments',
        max_chain_depth=1,
        skill_scopes={'refund': frozenset({'payments:refund'})},
        dpop_required=True,
    )


def _bound_token(sts: DelegationTokenService, key: DPoPKey) -> str:
    root = sts.issue_root_token(
        subject='user:bayard',
        audience='agent:orchestrator',
        scopes={'payments:refund'},
    ).access_token
    return sts.exchange(
        actor_id='agent:orchestrator',
        actor_secret=SECRETS['agent:orchestrator'],
        subject_token=root,
        audience='agent:payments',
        dpop_jkt=key.thumbprint,
    ).access_token


def _rpc() -> dict[str, Any]:
    return {
        'jsonrpc': '2.0',
        'id': 1,
        'method': 'SendMessage',
        'params': {
            'message': {
                'messageId': f'm{time.time_ns()}',
                'role': 'ROLE_USER',
                'parts': [{'text': 'refund'}],
            }
        },
    }


async def _post(app: Any, base: str, token: str, proof: str) -> httpx.Response:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url=base
    ) as http:
        return await http.post(
            '/',
            json=_rpc(),
            headers={
                'A2A-Version': '1.0',
                HTTP_EXTENSION_HEADER: DELEGATION_EXTENSION_URI,
                'Authorization': f'DPoP {token}',
                'DPoP': proof,
            },
        )


@pytest.mark.asyncio
@pytest.mark.parametrize('shared', [False, True])
async def test_replay_across_replicas(
    sts: DelegationTokenService, clock: Clock, shared: bool
) -> None:
    """Two replicas of payments behind one load balancer."""
    key = DPoPKey.generate()
    token = _bound_token(sts, key)
    proof = key.proof('POST', 'http://payments.test/', access_token=token)
    store = RedisReplayCache(FakeRedis(clock))

    def replica() -> Any:
        app, _ = build_agent_app(
            _params(),
            StaticKeySource(sts.keys),
            'http://payments.test',
            replay_cache=store if shared else ReplayCache(),
        )
        return app

    first = await _post(replica(), 'http://payments.test', token, proof)
    replay = await _post(replica(), 'http://payments.test', token, proof)

    assert first.status_code == 200
    if shared:
        assert replay.json()['error'] == 'invalid_dpop_proof'
    else:
        # Per-process caches cannot see each other: the replay succeeds.
        assert replay.status_code == 200


# --- proxies -----------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize('configured', [False, True])
async def test_dpop_behind_a_proxy(
    sts: DelegationTokenService, configured: bool
) -> None:
    """The client signs the public URL; the app sees the internal one."""
    key = DPoPKey.generate()
    token = _bound_token(sts, key)
    proof = key.proof(
        'POST', 'https://payments.example.com/', access_token=token
    )
    app, _ = build_agent_app(
        _params(),
        StaticKeySource(sts.keys),
        'https://payments.example.com',
        public_base_url='https://payments.example.com/' if configured else None,
    )

    r = await _post(app, 'http://10.0.0.7:8080', token, proof)

    if configured:
        assert r.status_code == 200, r.text
    else:
        assert 'htu' in r.json()['error_description']


def test_public_base_url_is_used_for_htu(sts: DelegationTokenService) -> None:
    backend = DelegationAuthBackend(
        _params(),
        StaticKeySource(sts.keys),
        public_base_url='https://payments.example.com/',
    )

    class Conn:
        url = httpx.URL('http://10.0.0.7:8080/rpc?x=1')

    assert backend._public_url(Conn()) == 'https://payments.example.com/rpc'  # type: ignore[arg-type]


# --- private_key_jwt client authentication ------------------------------------


@pytest.fixture
def travel_key() -> SigningKey:
    return SigningKey.generate()


@pytest.fixture
def keyed_sts(
    signing_key: SigningKey, clock: Clock, travel_key: SigningKey
) -> DelegationTokenService:
    agents = [a for a in AGENTS if a.agent_id != 'agent:travel']
    agents.append(
        RegisteredAgent.with_key(
            'agent:travel',
            travel_key.public_jwk(),
            {'travel:book', 'payments:refund'},
        )
    )
    return DelegationTokenService(
        issuer=ISSUER,
        signing_key=signing_key,
        agents=agents,
        max_chain_depth=2,
        clock=clock,
    )


def _travel_subject(sts: DelegationTokenService) -> str:
    root = sts.issue_root_token(
        subject='user:bayard',
        audience='agent:orchestrator',
        scopes={'payments:refund'},
    ).access_token
    return sts.exchange(
        actor_id='agent:orchestrator',
        actor_secret=SECRETS['agent:orchestrator'],
        subject_token=root,
        audience='agent:travel',
    ).access_token


def _assertion(key: SigningKey, clock: Clock, **overrides: Any) -> str:
    now = int(clock())
    claims: dict[str, Any] = {
        'iss': 'agent:travel',
        'sub': 'agent:travel',
        'aud': TOKEN_URL,
        'iat': now,
        'exp': now + 60,
        'jti': f'a{time.time_ns()}',
    }
    claims.update(overrides)
    return key.sign({k: v for k, v in claims.items() if v is not None})


def _sts_http(sts: DelegationTokenService) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(
            app=create_sts_app(sts, token_url=TOKEN_URL)
        ),
        base_url='http://sts.internal:9000',
    )


@pytest.mark.asyncio
async def test_exchange_with_private_key_jwt(
    keyed_sts: DelegationTokenService,
    travel_key: SigningKey,
    clock: Clock,
) -> None:
    exchanger = TokenExchangeClient(
        issuer=ISSUER,
        token_endpoint=TOKEN_URL,
        client_id='agent:travel',
        client_key=travel_key,
        http=_sts_http(keyed_sts),
        clock=clock,
    )

    token = await exchanger.exchange(
        _travel_subject(keyed_sts), 'agent:payments'
    )

    claims = verify_token(
        token.access_token,
        keyed_sts.keys,
        issuer=ISSUER,
        audience='agent:payments',
    )
    assert actor_chain(claims) == ['agent:travel', 'agent:orchestrator']


async def _post_assertion(
    sts: DelegationTokenService,
    assertion: str,
    *,
    extra: dict[str, str] | None = None,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    form = {
        'grant_type': TOKEN_EXCHANGE_GRANT,
        'subject_token': _travel_subject(sts),
        'audience': 'agent:payments',
        'client_assertion_type': CLIENT_ASSERTION_TYPE,
        'client_assertion': assertion,
        **(extra or {}),
    }
    async with _sts_http(sts) as http:
        return await http.post('/token', data=form, headers=headers or {})


@pytest.mark.asyncio
async def test_assertion_cannot_be_replayed(
    keyed_sts: DelegationTokenService,
    travel_key: SigningKey,
    clock: Clock,
) -> None:
    async with _sts_http(keyed_sts) as http:
        app_assertion = _assertion(travel_key, clock)
        forms = [
            {
                'grant_type': TOKEN_EXCHANGE_GRANT,
                'subject_token': _travel_subject(keyed_sts),
                'audience': 'agent:payments',
                'client_assertion_type': CLIENT_ASSERTION_TYPE,
                'client_assertion': app_assertion,
            }
            for _ in range(2)
        ]
        first = await http.post('/token', data=forms[0])
        second = await http.post('/token', data=forms[1])

    assert first.status_code == 200, first.text
    assert second.status_code == 401
    assert 'already been used' in second.json()['error_description']


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('overrides', 'message'),
    [
        ({'aud': 'https://elsewhere/token'}, 'invalid client assertion'),
        ({'sub': 'agent:orchestrator'}, 'sub must equal iss'),
        ({'iss': 'agent:nobody'}, 'unknown client'),
        (
            {'iss': 'agent:orchestrator', 'sub': 'agent:orchestrator'},
            'no registered key',
        ),
        ({'exp': 1}, 'expired'),
        ({'exp': None}, 'invalid client assertion'),
        ({'exp': 'later'}, 'invalid client assertion'),
    ],
)
async def test_bad_assertions_rejected(
    keyed_sts: DelegationTokenService,
    travel_key: SigningKey,
    clock: Clock,
    overrides: dict[str, Any],
    message: str,
) -> None:
    r = await _post_assertion(
        keyed_sts, _assertion(travel_key, clock, **overrides)
    )

    assert (r.status_code, r.json()['error']) == (401, 'invalid_client')
    assert message in r.json()['error_description']


@pytest.mark.asyncio
async def test_assertion_lifetime_and_signing_key(
    keyed_sts: DelegationTokenService,
    travel_key: SigningKey,
    clock: Clock,
) -> None:
    now = int(clock())
    too_long = _assertion(travel_key, clock, exp=now + 3600)
    wrong_key = _assertion(SigningKey.generate(), clock)

    long_r = await _post_assertion(keyed_sts, too_long)
    key_r = await _post_assertion(keyed_sts, wrong_key)

    assert 'lifetime too long' in long_r.json()['error_description']
    assert 'invalid client assertion' in key_r.json()['error_description']


@pytest.mark.asyncio
async def test_assertion_request_shape_errors(
    keyed_sts: DelegationTokenService,
    travel_key: SigningKey,
    clock: Clock,
) -> None:
    mismatch = await _post_assertion(
        keyed_sts,
        _assertion(travel_key, clock),
        extra={'client_id': 'agent:orchestrator'},
    )
    both = await _post_assertion(
        keyed_sts,
        _assertion(travel_key, clock),
        headers={
            'Authorization': basic_auth_header(
                'agent:orchestrator', SECRETS['agent:orchestrator']
            )
        },
    )
    wrong_type = await _post_assertion(
        keyed_sts,
        _assertion(travel_key, clock),
        extra={'client_assertion_type': 'urn:example:other'},
    )
    garbage = await _post_assertion(keyed_sts, 'not-a-jwt')

    assert 'client_id does not match' in mismatch.json()['error_description']
    assert both.json()['error'] == 'invalid_request'
    assert wrong_type.json()['error'] == 'invalid_client'
    assert 'malformed' in garbage.json()['error_description']


def test_key_registered_agent_cannot_use_a_secret(
    keyed_sts: DelegationTokenService,
) -> None:
    with pytest.raises(TokenExchangeError) as e:
        keyed_sts.authenticate_secret('agent:travel', '')
    assert e.value.error == 'invalid_client'


def test_exchange_for_rejects_unregistered_actor_objects(
    keyed_sts: DelegationTokenService,
) -> None:
    impostor = RegisteredAgent.with_secret(
        'agent:travel', 'x', {'payments:refund'}
    )

    with pytest.raises(TokenExchangeError) as e:
        keyed_sts.exchange_for(
            impostor,
            subject_token=_travel_subject(keyed_sts),
            audience='agent:payments',
        )
    assert e.value.error == 'invalid_client'


def test_exchanger_needs_exactly_one_credential(travel_key: SigningKey) -> None:
    for kwargs in ({}, {'client_secret': 's', 'client_key': travel_key}):
        with pytest.raises(ValueError, match='exactly one'):
            TokenExchangeClient(
                issuer=ISSUER,
                token_endpoint=TOKEN_URL,
                client_id='agent:travel',
                http=httpx.AsyncClient(),
                **kwargs,
            )


@pytest.mark.asyncio
async def test_sts_dpop_uses_public_token_url(
    sts: DelegationTokenService,
) -> None:
    """A DPoP proof signed for the public token URL verifies behind a proxy."""
    key = DPoPKey.generate()
    exchanger = TokenExchangeClient(
        issuer=ISSUER,
        token_endpoint=TOKEN_URL,
        client_id='agent:orchestrator',
        client_secret=SECRETS['agent:orchestrator'],
        http=_sts_http(sts),
        dpop_key=key,
    )
    root = sts.issue_root_token(
        subject='user:bayard',
        audience='agent:orchestrator',
        scopes={'payments:refund'},
    ).access_token

    token = await exchanger.exchange(root, 'agent:payments')

    assert token.token_type == 'DPoP'
    assert jwt.decode(token.access_token, options={'verify_signature': False})[
        'cnf'
    ] == {'jkt': key.thumbprint}
