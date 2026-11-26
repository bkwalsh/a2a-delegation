"""Client-side forwarding for the delegation extension.

When an agent that received a delegated request calls another agent, it must
exchange the token it received for one addressed to the next agent (spec §5).
:class:`DelegationInterceptor` does this automatically for every call to an
agent whose card declares the extension.

Typical use inside an executor::

    principal = get_principal(context.call_context)
    await client.send_message(request, context=forward_context(principal))
"""

from __future__ import annotations

import asyncio
import hashlib
import time

from dataclasses import dataclass
from typing import TYPE_CHECKING

from a2a.client.client import ClientCallContext
from a2a.client.interceptors import ClientCallInterceptor
from a2a.client.service_parameters import (
    ServiceParametersFactory,
    with_a2a_extensions,
)
from a2a.extensions.delegation.card import (
    DELEGATION_EXTENSION_URI,
    get_delegation_params,
)
from a2a.extensions.delegation.tokens import (
    ACCESS_TOKEN_TYPE,
    TOKEN_EXCHANGE_GRANT,
)


if TYPE_CHECKING:
    from collections.abc import Callable

    import httpx

    from a2a.client.interceptors import AfterArgs, BeforeArgs
    from a2a.extensions.delegation.dpop import DPoPKey
    from a2a.extensions.delegation.server import DelegatedPrincipal
    from a2a.extensions.delegation.tokens import SigningKey


SUBJECT_TOKEN_STATE_KEY = 'delegation_subject_token'  # noqa: S105 - a dict key
"""ClientCallContext.state key holding the token to delegate from."""

SCOPE_STATE_KEY = 'delegation_scope'
"""Optional ClientCallContext.state key narrowing the forwarded scopes."""

_REFRESH_MARGIN_SECONDS = 10
_ASSERTION_TTL_SECONDS = 60


class DelegationForwardingError(Exception):
    """A delegated call could not be prepared.

    Attributes:
        error: The OAuth error from the token endpoint, or a local reason.
        description: Human-readable detail.
    """

    def __init__(self, error: str, description: str) -> None:
        super().__init__(f'{error}: {description}')
        self.error = error
        self.description = description


@dataclass(frozen=True)
class ExchangedToken:
    """A token obtained from the STS, with its absolute expiry."""

    access_token: str
    scope: str
    expires_at: float
    token_type: str = 'Bearer'  # noqa: S105 - OAuth token type


class TokenExchangeClient:
    """Performs RFC 8693 exchanges against this agent's own STS.

    The client is bound to a single trusted issuer and token endpoint. It
    never sends its credentials or a subject token anywhere a target's
    Agent Card points to; see :meth:`DelegationInterceptor.before`.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        issuer: str,
        token_endpoint: str,
        client_id: str,
        client_secret: str | None = None,
        client_key: SigningKey | None = None,
        http: httpx.AsyncClient,
        clock: Callable[[], float] = time.time,
        dpop_key: DPoPKey | None = None,
    ) -> None:
        """Creates the client.

        Authenticate with exactly one of ``client_secret`` (HTTP Basic) or
        ``client_key`` (``private_key_jwt``, RFC 7523; preferred, since no
        shared secret is sent and each assertion is single-use).

        If ``dpop_key`` is set, every exchange proves possession of it and the
        STS binds issued tokens to it (RFC 9449). The httpx client used for
        A2A calls must then sign requests with ``DPoPSigner(dpop_key)``.
        """
        if (client_secret is None) == (client_key is None):
            raise ValueError('pass exactly one of client_secret or client_key')
        self._client_key = client_key
        self.issuer = issuer
        self.dpop_key = dpop_key
        self.token_endpoint = token_endpoint
        self._client_id = client_id
        self._client_secret = client_secret
        self._http = http
        self._clock = clock
        self._cache: dict[tuple[str, str, str], ExchangedToken] = {}
        self._lock = asyncio.Lock()

    async def exchange(
        self, subject_token: str, audience: str, scope: str | None = None
    ) -> ExchangedToken:
        """Returns a token for ``audience``, reusing a cached one if fresh.

        Raises:
            DelegationForwardingError: If the STS refuses the exchange.
        """
        key = (
            hashlib.sha256(subject_token.encode()).hexdigest(),
            audience,
            scope or '',
        )
        async with self._lock:
            cached = self._cache.get(key)
            if cached and cached.expires_at - _REFRESH_MARGIN_SECONDS > (
                self._clock()
            ):
                return cached
            token = await self._request(subject_token, audience, scope)
            self._cache[key] = token
            return token

    async def _request(
        self, subject_token: str, audience: str, scope: str | None
    ) -> ExchangedToken:
        from a2a.extensions.delegation.sts import (  # noqa: PLC0415
            CLIENT_ASSERTION_TYPE,
            basic_auth_header,
        )
        from a2a.extensions.delegation.tokens import (  # noqa: PLC0415
            new_jti,
        )

        form = {
            'grant_type': TOKEN_EXCHANGE_GRANT,
            'subject_token': subject_token,
            'subject_token_type': ACCESS_TOKEN_TYPE,
            'audience': audience,
        }
        if scope:
            form['scope'] = scope
        headers: dict[str, str] = {}
        if self._client_key is not None:
            now = int(self._clock())
            form['client_id'] = self._client_id
            form['client_assertion_type'] = CLIENT_ASSERTION_TYPE
            form['client_assertion'] = self._client_key.sign(
                {
                    'iss': self._client_id,
                    'sub': self._client_id,
                    'aud': self.token_endpoint,
                    'iat': now,
                    'exp': now + _ASSERTION_TTL_SECONDS,
                    'jti': new_jti(),
                }
            )
        else:
            headers['Authorization'] = basic_auth_header(
                self._client_id, self._client_secret or ''
            )
        if self.dpop_key is not None:
            from a2a.extensions.delegation.dpop import (  # noqa: PLC0415
                DPOP_HEADER,
            )

            headers[DPOP_HEADER] = self.dpop_key.proof(
                'POST', self.token_endpoint, now=self._clock()
            )
        response = await self._http.post(
            self.token_endpoint, data=form, headers=headers
        )
        try:
            body = response.json()
        except ValueError:
            body = {}
        if response.status_code != 200:  # noqa: PLR2004
            raise DelegationForwardingError(
                str(body.get('error', f'http_{response.status_code}')),
                str(body.get('error_description', 'token exchange failed')),
            )
        token_type = str(body.get('token_type', 'Bearer'))
        if self.dpop_key is not None and token_type.lower() != 'dpop':
            raise DelegationForwardingError(
                'invalid_token_type', 'STS did not bind the token to our key'
            )
        return ExchangedToken(
            access_token=body['access_token'],
            scope=body.get('scope', ''),
            expires_at=self._clock() + float(body['expires_in']),
            token_type='DPoP' if token_type.lower() == 'dpop' else 'Bearer',
        )


class DelegationInterceptor(ClientCallInterceptor):
    """Attaches a freshly exchanged delegation token to outgoing calls.

    For each call to an agent whose card declares the delegation extension,
    it exchanges the subject token in ``context.state`` for a token addressed
    to that agent, then sets ``Authorization`` and activates the extension.
    Calls to agents without the extension pass through unchanged.
    """

    def __init__(self, exchanger: TokenExchangeClient) -> None:
        self._exchanger = exchanger

    async def before(self, args: BeforeArgs) -> None:
        """Exchanges and attaches the token, or raises."""
        params = get_delegation_params(args.agent_card)
        if params is None:
            return

        # Only ever talk to our own STS. A card naming another issuer or
        # token endpoint could otherwise harvest our client credentials and
        # the user's token.
        if (
            params.issuer != self._exchanger.issuer
            or params.token_endpoint != self._exchanger.token_endpoint
        ):
            raise DelegationForwardingError(
                'untrusted_issuer',
                f'{params.audience} trusts {params.issuer}, not '
                f'{self._exchanger.issuer}',
            )

        state = args.context.state if args.context else {}
        subject_token = state.get(SUBJECT_TOKEN_STATE_KEY)
        if not isinstance(subject_token, str) or not subject_token:
            raise DelegationForwardingError(
                'missing_subject_token',
                f'{params.audience} requires delegation but the call has no '
                'subject token; use forward_context()',
            )
        scope = state.get(SCOPE_STATE_KEY)
        token = await self._exchanger.exchange(
            subject_token,
            params.audience,
            scope if isinstance(scope, str) else None,
        )

        context = args.context or ClientCallContext()
        context.service_parameters = ServiceParametersFactory.create_from(
            context.service_parameters,
            [with_a2a_extensions([DELEGATION_EXTENSION_URI])],
        )
        context.service_parameters['Authorization'] = (
            f'{token.token_type} {token.access_token}'
        )
        args.context = context

    async def after(self, args: AfterArgs) -> None:
        """Nothing to do after the call."""


def forward_context(
    principal: DelegatedPrincipal | str, *, scope: str | None = None
) -> ClientCallContext:
    """Builds a call context that delegates from an incoming request.

    Args:
        principal: The verified principal of the request being handled (or
            a raw subject token, for the first hop after login).
        scope: Optionally narrow the forwarded scopes (space-separated).
    """
    token = principal if isinstance(principal, str) else principal.token
    state: dict[str, str] = {SUBJECT_TOKEN_STATE_KEY: token}
    if scope:
        state[SCOPE_STATE_KEY] = scope
    return ClientCallContext(state=state)
