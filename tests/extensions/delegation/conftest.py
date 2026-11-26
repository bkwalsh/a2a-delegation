import pytest

from a2a.extensions.delegation.sts import (
    DelegationTokenService,
    RegisteredAgent,
)
from a2a.extensions.delegation.tokens import SigningKey


ISSUER = 'https://sts.example.com'

AGENTS = [
    RegisteredAgent.with_secret(
        'agent:orchestrator',
        'orchestrator-secret',
        {'travel:book', 'payments:refund', 'payments:read'},
    ),
    RegisteredAgent.with_secret(
        'agent:travel', 'travel-secret', {'travel:book', 'payments:refund'}
    ),
    RegisteredAgent.with_secret(
        'agent:payments',
        'payments-secret',
        {'payments:refund', 'payments:read'},
    ),
    RegisteredAgent.with_secret(
        'agent:ledger', 'ledger-secret', {'payments:refund', 'payments:read'}
    ),
]

SECRETS = {
    'agent:orchestrator': 'orchestrator-secret',
    'agent:travel': 'travel-secret',
    'agent:payments': 'payments-secret',
    'agent:ledger': 'ledger-secret',
}


class Clock:
    """A controllable clock for token lifetimes."""

    def __init__(self, now: float = 1_800_000_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def signing_key() -> SigningKey:
    return SigningKey.generate()


@pytest.fixture
def sts(signing_key: SigningKey, clock: Clock) -> DelegationTokenService:
    return DelegationTokenService(
        issuer=ISSUER,
        signing_key=signing_key,
        agents=AGENTS,
        max_chain_depth=2,
        ttl_seconds=300,
        clock=clock,
    )
