"""Server-side enforcement of the delegation extension.

Implements the receiver processing rules in
``docs/extensions/delegation/v1.md`` §6 as Starlette authentication
middleware, so every check runs before the request reaches the A2A handler
or any agent executor.

Typical wiring::

    app = Starlette(
        routes=[...],
        middleware=[delegation_middleware(params, keys)],
    )
    routes = create_jsonrpc_routes(
        handler, rpc_url='/', context_builder=DelegationContextBuilder()
    )
"""

from __future__ import annotations

import logging
import time

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from a2a.extensions.common import (
    HTTP_EXTENSION_HEADER,
    get_requested_extensions,
)
from a2a.extensions.delegation.card import DELEGATION_EXTENSION_URI
from a2a.extensions.delegation.tokens import (
    DEFAULT_LEEWAY_SECONDS,
    DelegationTokenError,
    KeySet,
    UnknownKeyError,
    actor_chain,
    scopes_of,
    verify_token,
)
from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH


try:
    from starlette.authentication import (
        AuthCredentials,
        AuthenticationBackend,
        AuthenticationError,
        BaseUser,
    )
    from starlette.middleware import Middleware
    from starlette.middleware.authentication import AuthenticationMiddleware
    from starlette.responses import JSONResponse
except ImportError as e:
    raise ImportError(
        'Delegation server middleware requires Starlette. '
        "Install with: 'pip install a2a-sdk[http-server,delegation]'"
    ) from e

from a2a.server.routes.common import DefaultServerCallContextBuilder


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    import httpx

    from starlette.requests import HTTPConnection, Request
    from starlette.responses import Response

    from a2a.extensions.delegation.card import DelegationParams
    from a2a.server.context import ServerCallContext


logger = logging.getLogger(__name__)
audit_logger = logging.getLogger('a2a.extensions.delegation.audit')

PRINCIPAL_STATE_KEY = 'delegation_principal'
"""Key under which the verified principal is stored in call-context state."""

_HTTP_BAD_REQUEST = 400
_HTTP_UNAUTHORIZED = 401
_HTTP_FORBIDDEN = 403

_STATUS_BY_CODE = {
    'extension_not_activated': _HTTP_BAD_REQUEST,
    'missing_token': _HTTP_UNAUTHORIZED,
    'invalid_token': _HTTP_UNAUTHORIZED,
    'wrong_issuer': _HTTP_UNAUTHORIZED,
    'wrong_audience': _HTTP_UNAUTHORIZED,
    'expired': _HTTP_UNAUTHORIZED,
    'invalid_dpop_proof': _HTTP_UNAUTHORIZED,
    'chain_too_deep': _HTTP_FORBIDDEN,
    'untrusted_actor': _HTTP_FORBIDDEN,
    'insufficient_scope': _HTTP_FORBIDDEN,
}


class DelegationDenied(AuthenticationError):  # noqa: N818 - mirrors Starlette
    """A request failed a delegation check.

    Attributes:
        code: The spec §7 error code.
        detail: Human-readable detail (safe to return to the caller).
    """

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f'{code}: {detail}')
        self.code = code
        self.detail = detail

    @property
    def status(self) -> int:
        """The HTTP status for this error."""
        return _STATUS_BY_CODE.get(self.code, _HTTP_UNAUTHORIZED)


class SkillNotAuthorizedError(PermissionError):
    """The verified token lacks the scopes for a specific skill."""

    def __init__(self, skill_id: str, missing: frozenset[str]) -> None:
        super().__init__(
            f'insufficient_scope: skill {skill_id!r} requires '
            f'{" ".join(sorted(missing))}'
        )
        self.skill_id = skill_id
        self.missing = missing


class KeySource(Protocol):
    """Supplies the issuer's verification keys."""

    async def get_keys(self, *, refresh: bool = False) -> KeySet:
        """Returns the current keys, refetching when ``refresh`` is set."""


class StaticKeySource:
    """A fixed key set, for in-process issuers and tests."""

    def __init__(self, keys: KeySet) -> None:
        self._keys = keys

    async def get_keys(self, *, refresh: bool = False) -> KeySet:
        """Returns the fixed keys."""
        return self._keys


class JwksKeySource:
    """Fetches and caches a remote JWKS.

    Keys are cached for ``ttl_seconds``. A token with an unknown ``kid``
    triggers a refetch (key rotation), at most once per
    ``min_refresh_seconds`` so forged ``kid`` values cannot be used to flood
    the issuer.
    """

    def __init__(
        self,
        jwks_uri: str,
        client: httpx.AsyncClient,
        *,
        ttl_seconds: float = 300,
        min_refresh_seconds: float = 30,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._uri = jwks_uri
        self._client = client
        self._ttl = ttl_seconds
        self._min_refresh = min_refresh_seconds
        self._clock = clock
        self._keys: KeySet | None = None
        self._fetched_at = float('-inf')

    async def get_keys(self, *, refresh: bool = False) -> KeySet:
        """Returns cached keys, refetching when stale or asked to refresh."""
        age = self._clock() - self._fetched_at
        stale = self._keys is None or age >= self._ttl
        if stale or (refresh and age >= self._min_refresh):
            response = await self._client.get(self._uri)
            response.raise_for_status()
            self._keys = KeySet.from_jwks(response.json())
            self._fetched_at = self._clock()
        assert self._keys is not None  # noqa: S101 - set above
        return self._keys


@dataclass(frozen=True)
class DelegatedPrincipal(BaseUser):
    """The verified identity behind a delegated request.

    Attributes:
        subject: The end principal (``sub``); tasks are owned by this value.
        actors: The delegation chain, most recent agent first.
        scopes: Scopes the token grants.
        audience: This agent (``aud``).
        token_id: The token's ``jti``.
        token: The raw token, kept server-side for onward exchange (PR 4).
            Never place it in model-visible content.
        claims: All verified claims.
    """

    subject: str
    actors: tuple[str, ...]
    scopes: frozenset[str]
    audience: str
    token_id: str
    token: str
    claims: dict[str, Any]

    @property
    def is_authenticated(self) -> bool:
        """Delegated principals are always authenticated."""
        return True

    @property
    def display_name(self) -> str:
        """The end principal, so task ownership follows the user."""
        return self.subject

    @property
    def identity(self) -> str:
        """The end principal."""
        return self.subject

    @property
    def depth(self) -> int:
        """The number of delegation hops."""
        return len(self.actors)

    def has_scopes(self, required: Iterable[str]) -> bool:
        """Whether the token grants every scope in ``required``."""
        return frozenset(required) <= self.scopes


class DelegationAuthBackend(AuthenticationBackend):
    """Starlette backend enforcing spec §6 on every non-public request."""

    def __init__(
        self,
        params: DelegationParams,
        keys: KeySource,
        *,
        public_paths: Iterable[str] = (AGENT_CARD_WELL_KNOWN_PATH,),
        leeway: int = DEFAULT_LEEWAY_SECONDS,
    ) -> None:
        self._params = params
        self._keys = keys
        self._public_paths = frozenset(public_paths)
        self._leeway = leeway

    async def authenticate(
        self, conn: HTTPConnection
    ) -> tuple[AuthCredentials, DelegatedPrincipal] | None:
        """Verifies the request, or raises DelegationDenied."""
        if conn.url.path in self._public_paths:
            return None
        try:
            principal = await self._verify(conn)
        except DelegationDenied as e:
            _audit(conn, 'deny', e.code, None)
            raise
        _audit(conn, 'allow', 'ok', principal)
        return AuthCredentials(sorted(principal.scopes)), principal

    async def _verify(self, conn: HTTPConnection) -> DelegatedPrincipal:
        params = self._params

        # 1. The request activates the extension.
        requested = get_requested_extensions(
            conn.headers.getlist(HTTP_EXTENSION_HEADER)
        )
        if DELEGATION_EXTENSION_URI not in requested:
            raise DelegationDenied(
                'extension_not_activated',
                f'{HTTP_EXTENSION_HEADER} must include the delegation '
                'extension',
            )

        # 2. A bearer token is present.
        token = _bearer_token(conn.headers.get('authorization'))
        if token is None:
            raise DelegationDenied('missing_token', 'no bearer token')

        # 3-5. Signature, issuer, audience and lifetime.
        claims = await self._verify_token(token)

        # 6. Chain depth.
        actors = tuple(actor_chain(claims))
        if len(actors) > params.max_chain_depth:
            raise DelegationDenied(
                'chain_too_deep',
                f'depth {len(actors)} exceeds {params.max_chain_depth}',
            )

        # 7. Trusted actors.
        if params.trusted_actors is not None:
            untrusted = [a for a in actors if a not in params.trusted_actors]
            if untrusted:
                raise DelegationDenied(
                    'untrusted_actor', f'untrusted actor {untrusted[0]!r}'
                )

        # 9. The token can invoke at least one skill.
        scopes = scopes_of(claims)
        if not scopes & params.all_scopes:
            raise DelegationDenied(
                'insufficient_scope', 'token grants no scope for this agent'
            )

        return DelegatedPrincipal(
            subject=str(claims['sub']),
            actors=actors,
            scopes=scopes,
            audience=str(claims['aud']),
            token_id=str(claims['jti']),
            token=token,
            claims=claims,
        )

    async def _verify_token(self, token: str) -> dict[str, Any]:
        keys = await self._keys.get_keys()
        try:
            return self._verify_with(token, keys)
        except UnknownKeyError:
            pass
        except DelegationTokenError as e:
            raise DelegationDenied(e.code, e.detail) from e
        # The issuer may have rotated keys; refetch once and retry.
        keys = await self._keys.get_keys(refresh=True)
        try:
            return self._verify_with(token, keys)
        except DelegationTokenError as e:
            raise DelegationDenied(e.code, e.detail) from e

    def _verify_with(self, token: str, keys: KeySet) -> dict[str, Any]:
        return verify_token(
            token,
            keys,
            issuer=self._params.issuer,
            audience=self._params.audience,
            leeway=self._leeway,
        )


def _bearer_token(header: str | None) -> str | None:
    if not header:
        return None
    scheme, _, token = header.partition(' ')
    if scheme.lower() != 'bearer' or not token.strip():
        return None
    return token.strip()


def _audit(
    conn: HTTPConnection,
    decision: str,
    code: str,
    principal: DelegatedPrincipal | None,
) -> None:
    """Writes one structured audit record. Never logs tokens."""
    record: dict[str, Any] = {
        'decision': decision,
        'code': code,
        'path': conn.url.path,
        'method': conn.scope.get('method'),
    }
    if principal is not None:
        record.update(
            sub=principal.subject,
            actors=list(principal.actors),
            aud=principal.audience,
            scopes=sorted(principal.scopes),
            jti=principal.token_id,
        )
    level = logging.INFO if decision == 'allow' else logging.WARNING
    audit_logger.log(
        level, 'delegation %s: %s', decision, code, extra={'delegation': record}
    )


def delegation_error_response(conn: HTTPConnection, exc: Exception) -> Response:
    """Maps a DelegationDenied to the spec §7 HTTP response."""
    if not isinstance(exc, DelegationDenied):
        exc = DelegationDenied('invalid_token', 'authentication failed')
    headers: dict[str, str] = {}
    if exc.status == _HTTP_UNAUTHORIZED:
        headers['WWW-Authenticate'] = (
            f'Bearer error="invalid_token", error_description="{exc.code}"'
        )
    elif exc.status == _HTTP_FORBIDDEN:
        headers['WWW-Authenticate'] = (
            f'Bearer error="insufficient_scope", error_description="{exc.code}"'
        )
    return JSONResponse(
        {'error': exc.code, 'error_description': exc.detail},
        status_code=exc.status,
        headers=headers,
    )


def delegation_middleware(
    params: DelegationParams,
    keys: KeySource,
    *,
    public_paths: Iterable[str] = (AGENT_CARD_WELL_KNOWN_PATH,),
) -> Middleware:
    """Starlette middleware enforcing the delegation extension."""
    return Middleware(
        AuthenticationMiddleware,
        backend=DelegationAuthBackend(params, keys, public_paths=public_paths),
        on_error=delegation_error_response,
    )


class DelegationContextBuilder(DefaultServerCallContextBuilder):
    """Exposes the verified principal to request handlers and executors."""

    def build(self, request: Request) -> ServerCallContext:
        """Builds the default context plus the delegation principal."""
        context = super().build(request)
        user = request.scope.get('user')
        if isinstance(user, DelegatedPrincipal):
            context.state[PRINCIPAL_STATE_KEY] = user
        return context


def get_principal(context: ServerCallContext) -> DelegatedPrincipal | None:
    """Returns the verified delegation principal for a call, if any."""
    principal = context.state.get(PRINCIPAL_STATE_KEY)
    return principal if isinstance(principal, DelegatedPrincipal) else None


def authorize_skill(
    context: ServerCallContext, params: DelegationParams, skill_id: str
) -> DelegatedPrincipal:
    """Checks that the caller may invoke ``skill_id`` (spec §6, per skill).

    Call this in the executor before performing a skill. It is a
    deterministic check on the verified token; the model's choice of skill
    cannot widen what the token grants.

    Raises:
        SkillNotAuthorizedError: If there is no principal, the skill is not
            declared, or the token lacks any of its scopes.
    """
    principal = get_principal(context)
    required = params.skill_scopes.get(skill_id)
    if required is None:
        raise SkillNotAuthorizedError(skill_id, frozenset({'<undeclared>'}))
    if principal is None:
        raise SkillNotAuthorizedError(skill_id, required)
    missing = required - principal.scopes
    if missing:
        raise SkillNotAuthorizedError(skill_id, missing)
    return principal
