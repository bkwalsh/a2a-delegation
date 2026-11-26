import logging
import time

from typing import Any

import httpx
import pytest

from a2a.client.client import ClientCallContext, ClientConfig
from a2a.client.client_factory import ClientFactory
from a2a.extensions.common import HTTP_EXTENSION_HEADER
from a2a.extensions.delegation import DELEGATION_EXTENSION_URI, DelegationParams
from a2a.extensions.delegation.server import (
    DelegatedPrincipal,
    DelegationDenied,
    JwksKeySource,
    SkillNotAuthorizedError,
    StaticKeySource,
    authorize_skill,
    delegation_error_response,
    get_principal,
)
from a2a.extensions.delegation.sts import DelegationTokenService
from a2a.extensions.delegation.tokens import KeySet, SigningKey
from a2a.server.context import ServerCallContext
from a2a.types import (
    Message,
    Part,
    Role,
    SendMessageRequest,
    TaskState,
)
from a2a.utils import TransportProtocol
from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH

from .agents import SkillExecutor, agent_card, build_agent_app
from .conftest import ISSUER, SECRETS


URL = 'http://payments.test'

PARAMS = DelegationParams(
    issuer=ISSUER,
    token_endpoint=f'{ISSUER}/token',
    jwks_uri=f'{ISSUER}/.well-known/jwks.json',
    audience='agent:payments',
    max_chain_depth=1,
    skill_scopes={
        'refund': frozenset({'payments:refund'}),
        'balance': frozenset({'payments:read'}),
    },
    trusted_actors=frozenset({'agent:orchestrator', 'agent:travel'}),
)


@pytest.fixture
def app_and_executor(
    sts: DelegationTokenService,
) -> tuple[Any, SkillExecutor]:
    app, executor = build_agent_app(PARAMS, StaticKeySource(sts.keys), URL)
    assert isinstance(executor, SkillExecutor)
    return app, executor


@pytest.fixture
def http(app_and_executor: tuple[Any, SkillExecutor]) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app_and_executor[0]), base_url=URL
    )


def _payments_token(
    sts: DelegationTokenService, scopes: set[str] | None = None
) -> str:
    """user -> orchestrator -> payments (depth 1)."""
    root = sts.issue_root_token(
        subject='user:bayard',
        audience='agent:orchestrator',
        scopes=scopes or {'payments:refund', 'payments:read'},
    ).access_token
    return sts.exchange(
        actor_id='agent:orchestrator',
        actor_secret=SECRETS['agent:orchestrator'],
        subject_token=root,
        audience='agent:payments',
    ).access_token


def _headers(
    token: str | None, *, activate: bool = True, scheme: str = 'Bearer'
) -> dict[str, str]:
    headers = {'A2A-Version': '1.0'}
    if activate:
        headers[HTTP_EXTENSION_HEADER] = DELEGATION_EXTENSION_URI
    if token is not None:
        headers['Authorization'] = f'{scheme} {token}'
    return headers


def _rpc(text: str) -> dict[str, Any]:
    return {
        'jsonrpc': '2.0',
        'id': 1,
        'method': 'SendMessage',
        'params': {
            'message': {
                'messageId': 'm1',
                'role': 'ROLE_USER',
                'parts': [{'text': text}],
            }
        },
    }


async def _send(
    http: httpx.AsyncClient, headers: dict[str, str], text: str = 'refund'
) -> httpx.Response:
    async with http:
        return await http.post('/', json=_rpc(text), headers=headers)


def _task_state(response: httpx.Response) -> str:
    return response.json()['result']['task']['status']['state']


# --- allowed --------------------------------------------------------------


@pytest.mark.asyncio
async def test_valid_delegated_request_runs_skill(
    sts: DelegationTokenService,
    http: httpx.AsyncClient,
    app_and_executor: tuple[Any, SkillExecutor],
) -> None:
    r = await _send(http, _headers(_payments_token(sts)))

    assert r.status_code == 200, r.text
    assert _task_state(r) == 'TASK_STATE_COMPLETED'
    [principal] = app_and_executor[1].principals
    assert principal.subject == 'user:bayard'
    assert principal.actors == ('agent:orchestrator',)
    assert principal.audience == 'agent:payments'
    assert principal.display_name == 'user:bayard'
    assert principal.depth == 1


@pytest.mark.asyncio
async def test_sdk_client_with_service_parameters(
    sts: DelegationTokenService,
    app_and_executor: tuple[Any, SkillExecutor],
) -> None:
    client = ClientFactory(
        config=ClientConfig(
            httpx_client=httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app_and_executor[0]),
                base_url=URL,
            ),
            supported_protocol_bindings=[TransportProtocol.JSONRPC],
        )
    ).create(agent_card(PARAMS, URL))
    context = ClientCallContext(
        service_parameters=_headers(_payments_token(sts))
    )
    request = SendMessageRequest(
        message=Message(
            role=Role.ROLE_USER, message_id='m1', parts=[Part(text='balance')]
        )
    )

    events = [e async for e in client.send_message(request, context=context)]

    assert events[-1].task.status.state == TaskState.TASK_STATE_COMPLETED


@pytest.mark.asyncio
async def test_agent_card_is_public(http: httpx.AsyncClient) -> None:
    async with http:
        r = await http.get(AGENT_CARD_WELL_KNOWN_PATH)

    assert r.status_code == 200
    assert DELEGATION_EXTENSION_URI in r.text


# --- rejected before the executor runs -------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('headers', 'status', 'code'),
    [
        ({'activate': False}, 400, 'extension_not_activated'),
        ({'token': None}, 401, 'missing_token'),
        ({'scheme': 'Basic'}, 401, 'missing_token'),
        ({'token': 'not-a-jwt'}, 401, 'invalid_token'),
    ],
)
async def test_request_shape_rejections(
    sts: DelegationTokenService,
    http: httpx.AsyncClient,
    app_and_executor: tuple[Any, SkillExecutor],
    headers: dict[str, Any],
    status: int,
    code: str,
) -> None:
    token = headers.pop('token', _payments_token(sts))

    r = await _send(http, _headers(token, **headers))

    assert (r.status_code, r.json()['error']) == (status, code)
    assert app_and_executor[1].principals == []


@pytest.mark.asyncio
async def test_root_token_for_another_agent_is_wrong_audience(
    sts: DelegationTokenService,
    http: httpx.AsyncClient,
    app_and_executor: tuple[Any, SkillExecutor],
) -> None:
    # Confused deputy: replaying the orchestrator's own token at payments.
    root = sts.issue_root_token(
        subject='user:bayard',
        audience='agent:orchestrator',
        scopes={'payments:refund'},
    ).access_token

    r = await _send(http, _headers(root))

    assert (r.status_code, r.json()['error']) == (401, 'wrong_audience')
    assert r.headers['www-authenticate'] == (
        'Bearer error="invalid_token", error_description="wrong_audience"'
    )
    assert app_and_executor[1].principals == []


@pytest.mark.asyncio
async def test_chain_too_deep(
    sts: DelegationTokenService, http: httpx.AsyncClient
) -> None:
    root = sts.issue_root_token(
        subject='user:bayard',
        audience='agent:orchestrator',
        scopes={'payments:refund'},
    ).access_token
    t1 = sts.exchange(
        actor_id='agent:orchestrator',
        actor_secret=SECRETS['agent:orchestrator'],
        subject_token=root,
        audience='agent:travel',
    ).access_token
    t2 = sts.exchange(
        actor_id='agent:travel',
        actor_secret=SECRETS['agent:travel'],
        subject_token=t1,
        audience='agent:payments',
    ).access_token

    r = await _send(http, _headers(t2))

    assert (r.status_code, r.json()['error']) == (403, 'chain_too_deep')
    assert 'insufficient_scope' in r.headers['www-authenticate']


@pytest.mark.asyncio
async def test_untrusted_actor(
    sts: DelegationTokenService, http: httpx.AsyncClient
) -> None:
    # The ledger agent is registered with the STS but not trusted here.
    root = sts.issue_root_token(
        subject='user:bayard',
        audience='agent:ledger',
        scopes={'payments:refund'},
    ).access_token
    token = sts.exchange(
        actor_id='agent:ledger',
        actor_secret=SECRETS['agent:ledger'],
        subject_token=root,
        audience='agent:payments',
    ).access_token

    r = await _send(http, _headers(token))

    assert (r.status_code, r.json()['error']) == (403, 'untrusted_actor')


@pytest.mark.asyncio
async def test_token_without_any_agent_scope(
    signing_key: SigningKey, http: httpx.AsyncClient
) -> None:
    now = int(time.time())
    token = signing_key.sign(
        {
            'iss': ISSUER,
            'sub': 'user:bayard',
            'aud': 'agent:payments',
            'scope': 'travel:book',
            'iat': now,
            'exp': now + 60,
            'jti': 'j',
        }
    )

    r = await _send(http, _headers(token))

    assert (r.status_code, r.json()['error']) == (403, 'insufficient_scope')


@pytest.mark.asyncio
async def test_token_signed_by_unknown_key(http: httpx.AsyncClient) -> None:
    now = int(time.time())
    forged = SigningKey.generate().sign(
        {
            'iss': ISSUER,
            'sub': 'user:bayard',
            'aud': 'agent:payments',
            'scope': 'payments:refund',
            'iat': now,
            'exp': now + 60,
            'jti': 'j',
        }
    )

    r = await _send(http, _headers(forged))

    assert (r.status_code, r.json()['error']) == (401, 'invalid_token')


# --- per-skill authorization in the executor -------------------------------


@pytest.mark.asyncio
async def test_skill_outside_token_scope_is_rejected(
    sts: DelegationTokenService,
    http: httpx.AsyncClient,
    app_and_executor: tuple[Any, SkillExecutor],
) -> None:
    # The token can read balances but the "model" asks for a refund.
    token = _payments_token(sts, scopes={'payments:read'})

    r = await _send(http, _headers(token), text='refund')

    assert r.status_code == 200
    assert _task_state(r) == 'TASK_STATE_REJECTED'
    assert 'payments:refund' in r.text
    assert app_and_executor[1].principals == []


def _principal(scopes: set[str]) -> DelegatedPrincipal:
    return DelegatedPrincipal(
        subject='user:bayard',
        actors=('agent:orchestrator',),
        scopes=frozenset(scopes),
        audience='agent:payments',
        token_id='j',
        token='t',
        claims={},
    )


def test_authorize_skill_rules() -> None:
    context = ServerCallContext(
        state={'delegation_principal': _principal({'payments:read'})}
    )

    assert authorize_skill(context, PARAMS, 'balance').subject == 'user:bayard'
    with pytest.raises(SkillNotAuthorizedError, match='payments:refund'):
        authorize_skill(context, PARAMS, 'refund')
    with pytest.raises(SkillNotAuthorizedError, match='undeclared'):
        authorize_skill(context, PARAMS, 'wire_transfer')
    with pytest.raises(SkillNotAuthorizedError):
        authorize_skill(ServerCallContext(), PARAMS, 'balance')
    assert get_principal(ServerCallContext(state={'x': 1})) is None


def test_principal_helpers() -> None:
    principal = _principal({'a', 'b'})

    assert principal.is_authenticated
    assert principal.identity == 'user:bayard'
    assert principal.has_scopes({'a'})
    assert not principal.has_scopes({'a', 'c'})


def test_error_response_for_unexpected_exception() -> None:
    response = delegation_error_response(None, ValueError('boom'))  # type: ignore[arg-type]

    assert response.status_code == 401
    assert DelegationDenied('weird', 'x').status == 401


# --- audit -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_audit_records_never_contain_tokens(
    sts: DelegationTokenService,
    http: httpx.AsyncClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    token = _payments_token(sts)
    caplog.set_level(logging.INFO, logger='a2a.extensions.delegation.audit')

    await _send(http, _headers(token))
    allow = caplog.records[-1]

    assert allow.delegation['decision'] == 'allow'  # type: ignore[attr-defined]
    assert allow.delegation['actors'] == ['agent:orchestrator']  # type: ignore[attr-defined]
    assert allow.delegation['sub'] == 'user:bayard'  # type: ignore[attr-defined]
    assert all(token not in r.getMessage() for r in caplog.records)
    assert all(token not in str(r.__dict__) for r in caplog.records)


# --- JWKS key source -------------------------------------------------------


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


@pytest.mark.asyncio
async def test_jwks_key_source_caches_and_rotates() -> None:
    old, new = SigningKey.generate(), SigningKey.generate()
    published = {'keys': [old.public_jwk()]}
    fetches = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal fetches
        fetches += 1
        return httpx.Response(200, json=published)

    clock = _Clock()
    source = JwksKeySource(
        'https://sts.test/jwks',
        httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        ttl_seconds=300,
        min_refresh_seconds=30,
        clock=clock,
    )

    keys = await source.get_keys()
    await source.get_keys()
    assert fetches == 1
    assert keys.get(old.kid) is not None

    # Rotation: a refresh within the rate limit is ignored...
    published = {'keys': [old.public_jwk(), new.public_jwk()]}
    clock.now = 10
    assert (await source.get_keys(refresh=True)).get(new.kid) is None
    # ...but honoured after it.
    clock.now = 31
    assert (await source.get_keys(refresh=True)).get(new.kid) is not None
    assert fetches == 2

    # TTL expiry refetches without being asked.
    clock.now = 400
    await source.get_keys()
    assert fetches == 3


@pytest.mark.asyncio
async def test_unknown_kid_triggers_one_refetch(
    sts: DelegationTokenService, signing_key: SigningKey
) -> None:
    class RotatingSource:
        def __init__(self) -> None:
            self.refreshes = 0

        async def get_keys(self, *, refresh: bool = False) -> KeySet:
            if refresh:
                self.refreshes += 1
                return KeySet.from_signing_keys(signing_key)
            return KeySet({})

    source = RotatingSource()
    app, _ = build_agent_app(PARAMS, source, URL)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url=URL
    ) as http:
        r = await http.post(
            '/', json=_rpc('refund'), headers=_headers(_payments_token(sts))
        )

    assert r.status_code == 200, r.text
    assert source.refreshes == 1
