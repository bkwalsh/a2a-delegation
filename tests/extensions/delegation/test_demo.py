"""Runs every scenario in samples/delegation_demo.py as a test."""

from collections.abc import Awaitable, Callable

import pytest

from samples import delegation_demo as demo


@pytest.mark.asyncio
@pytest.mark.parametrize('scenario', demo.SCENARIOS, ids=lambda s: s.__name__)
async def test_scenario(
    scenario: Callable[[demo.Network], Awaitable[demo.Outcome]],
) -> None:
    net = demo.build_network()
    async with net.http:
        outcome = await scenario(net)

    assert outcome.ok, f'{outcome.name}: {outcome.observed}'


@pytest.mark.asyncio
async def test_legit_refund_reaches_payments_with_full_chain() -> None:
    net = demo.build_network()
    async with net.http:
        await demo.legit_refund(net)

    assert net.agents['agent:payments'].seen == [
        ('user:bayard', ('agent:travel', 'agent:orchestrator'))
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'scenario',
    [
        demo.user_never_granted_refunds,
        demo.prompt_injection,
        demo.confused_deputy_replay,
        demo.chain_too_deep,
        demo.untrusted_actor,
        demo.forged_token,
    ],
    ids=lambda s: s.__name__,
)
async def test_blocked_requests_never_run_a_payments_skill(
    scenario: Callable[[demo.Network], Awaitable[demo.Outcome]],
) -> None:
    net = demo.build_network()
    async with net.http:
        await scenario(net)

    assert net.agents['agent:payments'].seen == []


@pytest.mark.asyncio
async def test_unknown_host_is_not_routed() -> None:
    net = demo.build_network()
    async with net.http:
        r = await net.http.get('http://nowhere.demo/')

    assert r.status_code == 502


def test_main_prints_all_passing(capsys: pytest.CaptureFixture[str]) -> None:
    demo.main()

    out = capsys.readouterr().out
    assert out.count('[PASS]') == len(demo.SCENARIOS)
    assert '[FAIL]' not in out
