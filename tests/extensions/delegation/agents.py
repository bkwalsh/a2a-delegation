"""Test agents protected by the delegation extension."""

from __future__ import annotations

from typing import TYPE_CHECKING

from a2a.extensions.delegation import DelegationParams, delegation_extension
from a2a.extensions.delegation.server import (
    DelegatedPrincipal,
    DelegationContextBuilder,
    KeySource,
    SkillNotAuthorizedError,
    authorize_skill,
    delegation_middleware,
)
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from a2a.server.tasks.task_updater import TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    Part,
    Task,
    TaskState,
    TaskStatus,
)
from a2a.utils.constants import TransportProtocol
from starlette.applications import Starlette


if TYPE_CHECKING:
    from a2a.server.events import EventQueue


class SkillExecutor(AgentExecutor):
    """Picks a skill from the message text, then authorizes it in code.

    The text stands in for a model's decision: whatever it asks for, the
    skill only runs if the verified token holds the skill's scopes.
    """

    def __init__(self, params: DelegationParams) -> None:
        self.params = params
        self.principals: list[DelegatedPrincipal] = []

    async def execute(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        """Runs the requested skill if the token authorizes it."""
        task_id, context_id = context.task_id or '', context.context_id or ''
        if context.current_task is None:
            await event_queue.enqueue_event(
                Task(
                    id=task_id,
                    context_id=context_id,
                    status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED),
                    history=[context.message] if context.message else [],
                )
            )
        updater = TaskUpdater(event_queue, task_id, context_id)
        skill_id = context.get_user_input().strip()
        try:
            principal = authorize_skill(
                context.call_context, self.params, skill_id
            )
        except SkillNotAuthorizedError as e:
            await updater.reject(updater.new_agent_message([Part(text=str(e))]))
            return
        self.principals.append(principal)
        chain = ' <- '.join([principal.audience, *principal.actors])
        await updater.complete(
            updater.new_agent_message(
                [Part(text=f'{skill_id} for {principal.subject} via {chain}')]
            )
        )

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        """Cancellation is not needed by these agents."""


def agent_card(params: DelegationParams, url: str) -> AgentCard:
    """An Agent Card declaring delegation and one skill per scope entry."""
    return AgentCard(
        name=params.audience,
        description='Delegation test agent',
        version='1.0.0',
        supported_interfaces=[
            AgentInterface(
                protocol_binding=TransportProtocol.JSONRPC,
                url=url,
                protocol_version='1.0',
            )
        ],
        capabilities=AgentCapabilities(
            extensions=[delegation_extension(params)]
        ),
        default_input_modes=['text/plain'],
        default_output_modes=['text/plain'],
        skills=[
            AgentSkill(id=skill, name=skill, description=skill, tags=[skill])
            for skill in sorted(params.skill_scopes)
        ],
    )


def build_agent_app(
    params: DelegationParams,
    keys: KeySource,
    url: str,
    executor: AgentExecutor | None = None,
) -> tuple[Starlette, SkillExecutor | AgentExecutor]:
    """A JSON-RPC agent app protected by the delegation middleware."""
    card = agent_card(params, url)
    executor = executor or SkillExecutor(params)
    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
        agent_card=card,
    )
    app = Starlette(
        routes=[
            *create_agent_card_routes(agent_card=card),
            *create_jsonrpc_routes(
                request_handler=handler,
                rpc_url='/',
                context_builder=DelegationContextBuilder(),
            ),
        ],
        middleware=[delegation_middleware(params, keys)],
    )
    return app, executor
