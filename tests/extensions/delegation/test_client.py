from typing import Any

import httpx
import pytest

from a2a.client.client import ClientCallContext, ClientConfig
from a2a.client.client_factory import ClientFactory
from a2a.client.interceptors import BeforeArgs
from a2a.extensions.common import HTTP_EXTENSION_HEADER
from a2a.extensions.delegation import DELEGATION_EXTENSION_URI, DelegationParams
from a2a.extensions.delegation.client import (
    SUBJECT_TOKEN_STATE_KEY,
    DelegationForwardingError,
    DelegationInterceptor,
    TokenExchangeClient,
    forward_context,
)
from a2a.extensions.delegation.server import (
    StaticKeySource,
    get_principal,
)
from a2a.extensions.delegation.sts import (
    DelegationTokenService,
    create_sts_app,
)
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks.task_updater import TaskUpdater
from a2a.types import (
    AgentCard,
    Message,
    Part,
    Role,
    SendMessageRequest,
    Task,
    TaskState,
    TaskStatus,
)
from a2a.utils import TransportProtocol

from .agents import SkillExecutor, agent_card, build_agent_app
from .conftest import ISSUER, SECRETS, Clock


TOKEN_ENDPOINT = f'{ISSUER}/token'
ORCH_URL = 'http://orchestrator.test'
PAY_URL = 'http://payments.test'


def _params(audience: str, skills: dict[str, set[str]]) -> DelegationParams:
    return DelegationParams(
        issuer=ISSUER,
        token_endpoint=TOKEN_ENDPOINT,
        jwks_uri=f'{ISSUER}/.well-known/jwks.json',
        audience=audience,
        max_chain_depth=2,
        skill_scopes={k: frozenset(v) for k, v in skills.items()},
    )


PAY_PARAMS = _params(
    'agent:payments',
    {'refund': {'payments:refund'}, 'balance': {'payments:read'}},
)
ORCH_PARAMS = _params(
    'agent:orchestrator',
    {'plan': {'payments:refund', 'payments:read', 'travel:book'}},
)


def _exchanger(
    http: httpx.AsyncClient, clock: Clock | None = None, **kw: Any
) -> TokenExchangeClient:
    return TokenExchangeClient(
        issuer=kw.get('issuer', ISSUER),
        token_endpoint=kw.get('token_endpoint', TOKEN_ENDPOINT),
        client_id=kw.get('client_id', 'agent:orchestrator'),
        client_secret=kw.get('client_secret', SECRETS['agent:orchestrator']),
        http=http,
        **({'clock': clock} if clock else {}),
    )


def _root(sts: DelegationTokenService, scopes: set[str] | None = None) -> str:
    return sts.issue_root_token(
        subject='user:bayard',
        audience='agent:orchestrator',
        scopes=scopes or {'payments:refund', 'payments:read', 'travel:book'},
    ).access_token


class CountingSts:
    """Wraps the STS app to count token requests."""

    def __init__(self, sts: DelegationTokenService) -> None:
        self.app = create_sts_app(sts)
        self.calls = 0

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope['type'] == 'http' and scope['path'] == '/token':
            self.calls += 1
        await self.app(scope, receive, send)


# --- TokenExchangeClient ---------------------------------------------------


@pytest.mark.asyncio
async def test_exchange_and_cache(
    sts: DelegationTokenService, clock: Clock
) -> None:
    counting = CountingSts(sts)
    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=counting))
    exchanger = _exchanger(http, clock)
    root = _root(sts)

    first = await exchanger.exchange(root, 'agent:payments')
    second = await exchanger.exchange(root, 'agent:payments')
    narrowed = await exchanger.exchange(root, 'agent:payments', 'payments:read')

    assert first == second
    assert narrowed.scope == 'payments:read'
    assert counting.calls == 2

    # Refreshes once the cached token is within the refresh margin.
    clock.now += 295
    await exchanger.exchange(root, 'agent:payments')
    assert counting.calls == 3


@pytest.mark.asyncio
async def test_exchange_error_is_surfaced(sts: DelegationTokenService) -> None:
    http = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_sts_app(sts))
    )
    exchanger = _exchanger(http, client_secret='wrong')

    with pytest.raises(DelegationForwardingError) as e:
        await exchanger.exchange(_root(sts), 'agent:payments')
    assert e.value.error == 'invalid_client'


@pytest.mark.asyncio
async def test_non_json_error_body() -> None:
    http = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(502, text='x'))
    )

    with pytest.raises(DelegationForwardingError) as e:
        await _exchanger(http).exchange('t', 'agent:payments')
    assert e.value.error == 'http_502'


# --- DelegationInterceptor -------------------------------------------------


def _args(card: AgentCard, context: ClientCallContext | None) -> BeforeArgs:
    return BeforeArgs(
        input=None, method='send_message', agent_card=card, context=context
    )


@pytest.mark.asyncio
async def test_interceptor_ignores_agents_without_extension() -> None:
    interceptor = DelegationInterceptor(_exchanger(httpx.AsyncClient()))
    args = _args(AgentCard(name='plain'), None)

    await interceptor.before(args)
    await interceptor.after(None)  # type: ignore[arg-type]

    assert args.context is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'override',
    [
        {'issuer': 'https://evil.example.com'},
        {'token_endpoint': 'https://evil.example.com/token'},
    ],
)
async def test_interceptor_refuses_foreign_issuer(
    override: dict[str, str],
) -> None:
    # A malicious card must not redirect our credentials and the user's token.
    requests: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(500)

    http = httpx.AsyncClient(transport=httpx.MockTransport(record))
    evil = DelegationParams(**{**PAY_PARAMS.__dict__, **override})
    interceptor = DelegationInterceptor(_exchanger(http))

    with pytest.raises(DelegationForwardingError) as e:
        await interceptor.before(
            _args(agent_card(evil, PAY_URL), forward_context('subject'))
        )
    assert e.value.error == 'untrusted_issuer'
    assert requests == []


@pytest.mark.asyncio
async def test_interceptor_requires_subject_token() -> None:
    interceptor = DelegationInterceptor(_exchanger(httpx.AsyncClient()))

    with pytest.raises(DelegationForwardingError) as e:
        await interceptor.before(
            _args(agent_card(PAY_PARAMS, PAY_URL), ClientCallContext())
        )
    assert e.value.error == 'missing_subject_token'


@pytest.mark.asyncio
async def test_interceptor_sets_headers_and_merges_extensions(
    sts: DelegationTokenService,
) -> None:
    http = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_sts_app(sts))
    )
    interceptor = DelegationInterceptor(_exchanger(http))
    context = forward_context(_root(sts))
    context.service_parameters = {HTTP_EXTENSION_HEADER: 'https://x.test/ext'}
    args = _args(agent_card(PAY_PARAMS, PAY_URL), context)

    await interceptor.before(args)

    assert args.context is not None
    params = args.context.service_parameters or {}
    assert params['Authorization'].startswith('Bearer ')
    assert set(params[HTTP_EXTENSION_HEADER].split(',')) == {
        'https://x.test/ext',
        DELEGATION_EXTENSION_URI,
    }


def test_forward_context_from_principal_and_scope() -> None:
    class P:
        token = 'tok'

    context = forward_context(P(), scope='payments:read')  # type: ignore[arg-type]

    assert context.state == {
        SUBJECT_TOKEN_STATE_KEY: 'tok',
        'delegation_scope': 'payments:read',
    }


# --- two hops, end to end --------------------------------------------------


class ForwardingExecutor(AgentExecutor):
    """An orchestrator that delegates the user's request to payments."""

    def __init__(self, http: httpx.AsyncClient, scope: str | None) -> None:
        self._scope = scope
        self._client = ClientFactory(
            config=ClientConfig(
                httpx_client=http,
                supported_protocol_bindings=[TransportProtocol.JSONRPC],
            )
        ).create(
            agent_card(PAY_PARAMS, PAY_URL),
            interceptors=[DelegationInterceptor(_exchanger(http))],
        )

    async def execute(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        task_id, context_id = context.task_id or '', context.context_id or ''
        await event_queue.enqueue_event(
            Task(
                id=task_id,
                context_id=context_id,
                status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED),
            )
        )
        updater = TaskUpdater(event_queue, task_id, context_id)
        principal = get_principal(context.call_context)
        assert principal is not None
        request = SendMessageRequest(
            message=Message(
                role=Role.ROLE_USER,
                message_id=f'{task_id}-fwd',
                parts=[Part(text=context.get_user_input())],
            )
        )
        events = [
            e
            async for e in self._client.send_message(
                request, context=forward_context(principal, scope=self._scope)
            )
        ]
        reply = events[-1].task.status
        await updater.complete(
            updater.new_agent_message(
                [
                    Part(
                        text=f'{TaskState.Name(reply.state)}: {reply.message.parts[0].text}'
                    )
                ]
            )
        )

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        """Not needed."""


async def _two_hops(
    sts: DelegationTokenService, text: str, scope: str | None = None
) -> tuple[str, SkillExecutor]:
    keys = StaticKeySource(sts.keys)
    pay_app, pay_exec = build_agent_app(PAY_PARAMS, keys, PAY_URL)
    http = httpx.AsyncClient(
        mounts={
            f'{ISSUER}/': httpx.ASGITransport(app=create_sts_app(sts)),
            f'{PAY_URL}/': httpx.ASGITransport(app=pay_app),
        }
    )
    orch_app, _ = build_agent_app(
        ORCH_PARAMS, keys, ORCH_URL, ForwardingExecutor(http, scope)
    )
    user = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=orch_app), base_url=ORCH_URL
    )
    async with user:
        r = await user.post(
            '/',
            json={
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
            },
            headers={
                'A2A-Version': '1.0',
                HTTP_EXTENSION_HEADER: DELEGATION_EXTENSION_URI,
                'Authorization': f'Bearer {_root(sts)}',
            },
        )
    assert r.status_code == 200, r.text
    status = r.json()['result']['task']['status']
    assert isinstance(pay_exec, SkillExecutor)
    return status['message']['parts'][0]['text'], pay_exec


@pytest.mark.asyncio
async def test_two_hop_delegation(sts: DelegationTokenService) -> None:
    reply, payments = await _two_hops(sts, 'refund')

    assert reply.startswith('TASK_STATE_COMPLETED')
    assert (
        'refund for user:bayard via agent:payments <- agent:orchestrator'
        in (reply)
    )
    [principal] = payments.principals
    assert principal.subject == 'user:bayard'
    assert principal.actors == ('agent:orchestrator',)


@pytest.mark.asyncio
async def test_orchestrator_can_narrow_before_forwarding(
    sts: DelegationTokenService,
) -> None:
    # The orchestrator forwards read-only authority, so a refund is refused
    # by payments even though the user's root token allowed refunds.
    reply, payments = await _two_hops(sts, 'refund', scope='payments:read')

    assert reply.startswith('TASK_STATE_REJECTED')
    assert 'payments:refund' in reply
    assert payments.principals == []
