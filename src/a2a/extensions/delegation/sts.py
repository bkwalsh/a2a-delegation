"""Reference Security Token Service (STS) for the delegation extension.

Issues root tokens and performs RFC 8693 token exchange with the narrowing
rules in ``docs/extensions/delegation/v1.md`` §5.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import time

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qsl, quote_plus, unquote_plus

from a2a.extensions.delegation.tokens import (
    ACCESS_TOKEN_TYPE,
    TOKEN_EXCHANGE_GRANT,
    DelegationTokenError,
    KeySet,
    SigningKey,
    actor_chain,
    new_jti,
    scopes_of,
    verify_token,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from starlette.applications import Starlette
    from starlette.requests import Request


DEFAULT_TTL_SECONDS = 300


class TokenExchangeError(Exception):
    """An OAuth error returned by the token endpoint (RFC 6749 §5.2).

    Attributes:
        error: The OAuth error code, for example ``invalid_scope``.
        description: Human-readable detail.
        status: The HTTP status to return.
    """

    def __init__(self, error: str, description: str, status: int = 400):
        super().__init__(f'{error}: {description}')
        self.error = error
        self.description = description
        self.status = status


@dataclass(frozen=True)
class RegisteredAgent:
    """An agent known to the STS.

    Attributes:
        agent_id: The identifier used as ``aud`` and in ``act.sub``.
        secret_sha256: SHA-256 hex digest of the agent's client secret.
        accepted_scopes: Scopes this agent may receive as an audience.
    """

    agent_id: str
    secret_sha256: str
    accepted_scopes: frozenset[str]

    @classmethod
    def with_secret(
        cls, agent_id: str, secret: str, accepted_scopes: Iterable[str]
    ) -> RegisteredAgent:
        """Registers an agent from a plaintext secret (stored hashed)."""
        return cls(
            agent_id=agent_id,
            secret_sha256=hashlib.sha256(secret.encode()).hexdigest(),
            accepted_scopes=frozenset(accepted_scopes),
        )

    def check_secret(self, secret: str) -> bool:
        """Constant-time comparison of a presented secret."""
        digest = hashlib.sha256(secret.encode()).hexdigest()
        return hmac.compare_digest(digest, self.secret_sha256)


@dataclass(frozen=True)
class IssuedToken:
    """A token endpoint response (RFC 8693 §2.2.1)."""

    access_token: str
    expires_in: int
    scope: str
    token_type: str = 'Bearer'  # noqa: S105 - OAuth token type, not a secret
    issued_token_type: str = ACCESS_TOKEN_TYPE

    def to_dict(self) -> dict[str, Any]:
        """The JSON body of a successful token response."""
        return {
            'access_token': self.access_token,
            'issued_token_type': self.issued_token_type,
            'token_type': self.token_type,
            'expires_in': self.expires_in,
            'scope': self.scope,
        }


class DelegationTokenService:
    """Issues and exchanges delegation tokens."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        issuer: str,
        signing_key: SigningKey,
        agents: Iterable[RegisteredAgent],
        max_chain_depth: int = 3,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.issuer = issuer
        self._key = signing_key
        self._agents = {a.agent_id: a for a in agents}
        self._max_depth = max_chain_depth
        self._ttl = ttl_seconds
        self._clock = clock

    def now(self) -> float:
        """The service's current time."""
        return self._clock()

    @property
    def keys(self) -> KeySet:
        """The public keys that verify tokens from this STS."""
        return KeySet.from_signing_keys(self._key)

    def jwks(self) -> dict[str, Any]:
        """The JWKS document served at ``jwks_uri``."""
        return {'keys': [self._key.public_jwk()]}

    def issue_root_token(
        self,
        *,
        subject: str,
        audience: str,
        scopes: Iterable[str],
        dpop_jkt: str | None = None,
    ) -> IssuedToken:
        """Issues a depth-0 token after the principal has authenticated.

        How ``subject`` authenticated (for example an OIDC login) is out of
        scope; callers must only invoke this for an authenticated principal.
        """
        target = self._target(audience)
        granted = frozenset(scopes) & target.accepted_scopes
        if not granted:
            raise TokenExchangeError(
                'invalid_scope', f'no requested scope is accepted by {audience}'
            )
        now = int(self._clock())
        return self._issue(
            {'sub': subject, 'aud': audience},
            scopes=granted,
            now=now,
            exp=now + self._ttl,
            dpop_jkt=dpop_jkt,
        )

    def exchange(  # noqa: PLR0913
        self,
        *,
        actor_id: str,
        actor_secret: str,
        subject_token: str,
        audience: str,
        scope: str | None = None,
        subject_token_type: str = ACCESS_TOKEN_TYPE,
        dpop_jkt: str | None = None,
    ) -> IssuedToken:
        """Performs an RFC 8693 token exchange for ``actor_id``.

        If ``dpop_jkt`` is given (the thumbprint of a key the actor proved it
        holds), the issued token is bound to that key (RFC 9449 §6).

        Raises:
            TokenExchangeError: If any rule in spec §5 is violated.
        """
        actor = self._authenticate(actor_id, actor_secret)
        if subject_token_type != ACCESS_TOKEN_TYPE:
            raise TokenExchangeError(
                'invalid_request', 'unsupported subject_token_type'
            )
        target = self._target(audience)
        if target.agent_id == actor.agent_id:
            raise TokenExchangeError(
                'invalid_target', 'an agent cannot delegate to itself'
            )

        # Rule 2: only the audience of a token may re-delegate it.
        try:
            subject = verify_token(
                subject_token,
                self.keys,
                issuer=self.issuer,
                audience=actor.agent_id,
                leeway=0,
                now=self._clock(),
            )
        except DelegationTokenError as e:
            description = (
                'subject token was not issued to the requesting agent'
                if e.code == 'wrong_audience'
                else e.detail
            )
            raise TokenExchangeError('invalid_grant', description) from e

        chain = actor_chain(subject)
        if len(chain) + 1 > self._max_depth:
            raise TokenExchangeError(
                'invalid_grant', 'delegation chain too deep'
            )
        if target.agent_id in chain or actor.agent_id in chain:
            raise TokenExchangeError(
                'invalid_grant', 'an agent may appear only once in the chain'
            )

        # Rules 3 and 4: scopes never widen.
        held = scopes_of(subject)
        requested = frozenset(scope.split()) if scope else held
        if not requested <= held:
            extra = ' '.join(sorted(requested - held))
            raise TokenExchangeError(
                'invalid_scope', f'subject token does not hold: {extra}'
            )
        granted = requested & target.accepted_scopes
        if not granted:
            raise TokenExchangeError(
                'invalid_scope', f'no requested scope is accepted by {audience}'
            )

        act: dict[str, Any] = {'sub': actor.agent_id}
        if 'act' in subject:
            act['act'] = subject['act']
        now = int(self._clock())
        return self._issue(
            {'sub': subject['sub'], 'aud': target.agent_id, 'act': act},
            scopes=granted,
            now=now,
            exp=min(now + self._ttl, int(subject['exp'])),
            dpop_jkt=dpop_jkt,
        )

    def _issue(
        self,
        claims: dict[str, Any],
        *,
        scopes: frozenset[str],
        now: int,
        exp: int,
        dpop_jkt: str | None = None,
    ) -> IssuedToken:
        scope = ' '.join(sorted(scopes))
        body = {
            **claims,
            'iss': self.issuer,
            'scope': scope,
            'iat': now,
            'exp': exp,
            'jti': new_jti(),
        }
        if dpop_jkt:
            body['cnf'] = {'jkt': dpop_jkt}
        return IssuedToken(
            access_token=self._key.sign(body),
            expires_in=max(exp - now, 0),
            scope=scope,
            token_type='DPoP' if dpop_jkt else 'Bearer',
        )

    def _authenticate(self, agent_id: str, secret: str) -> RegisteredAgent:
        agent = self._agents.get(agent_id)
        if agent is None or not agent.check_secret(secret):
            raise TokenExchangeError(
                'invalid_client', 'client authentication failed', status=401
            )
        return agent

    def _target(self, audience: str) -> RegisteredAgent:
        target = self._agents.get(audience)
        if target is None:
            raise TokenExchangeError(
                'invalid_target', f'unknown audience {audience!r}'
            )
        return target


def basic_auth_header(client_id: str, secret: str) -> str:
    """Builds an OAuth client ``Authorization: Basic`` header value.

    Per RFC 6749 §2.3.1 the id and secret are form-urlencoded first, so ids
    containing ``:`` (such as ``agent:travel``) survive the encoding.
    """
    raw = f'{quote_plus(client_id)}:{quote_plus(secret)}'
    return 'Basic ' + base64.b64encode(raw.encode()).decode()


def parse_basic_auth(header: str | None) -> tuple[str, str] | None:
    """Parses an OAuth client ``Authorization: Basic`` header (RFC 6749 §2.3.1)."""
    if not header or not header.lower().startswith('basic '):
        return None
    try:
        decoded = base64.b64decode(header[6:].strip(), validate=True).decode()
    except (binascii.Error, UnicodeDecodeError):
        return None
    client_id, sep, secret = decoded.partition(':')
    if not sep:
        return None
    return unquote_plus(client_id), unquote_plus(secret)


def create_sts_app(
    service: DelegationTokenService,
    *,
    token_path: str = '/token',  # noqa: S107 - a URL path
    jwks_path: str = '/.well-known/jwks.json',
) -> Starlette:
    """Builds a Starlette app serving the token endpoint and JWKS.

    Requires the ``http-server`` extra.
    """
    from starlette.applications import Starlette  # noqa: PLC0415
    from starlette.responses import JSONResponse  # noqa: PLC0415
    from starlette.routing import Route  # noqa: PLC0415

    from a2a.extensions.delegation.dpop import ReplayCache  # noqa: PLC0415

    no_store = {'Cache-Control': 'no-store'}
    replay_cache = ReplayCache()

    def error(e: TokenExchangeError) -> JSONResponse:
        headers = dict(no_store)
        if e.status == 401:  # noqa: PLR2004
            headers['WWW-Authenticate'] = 'Basic realm="delegation-sts"'
        return JSONResponse(
            {'error': e.error, 'error_description': e.description},
            status_code=e.status,
            headers=headers,
        )

    async def exchange(request: Request) -> IssuedToken:
        credentials = parse_basic_auth(request.headers.get('authorization'))
        if credentials is None:
            raise TokenExchangeError(
                'invalid_client', 'client authentication required', 401
            )
        content_type = request.headers.get('content-type', '')
        if not content_type.startswith('application/x-www-form-urlencoded'):
            raise TokenExchangeError(
                'invalid_request', 'body must be form-urlencoded'
            )
        fields = dict(parse_qsl((await request.body()).decode()))
        if fields.get('grant_type') != TOKEN_EXCHANGE_GRANT:
            raise TokenExchangeError(
                'unsupported_grant_type', 'only token exchange is supported'
            )
        for name in ('subject_token', 'audience'):
            if not fields.get(name):
                raise TokenExchangeError(
                    'invalid_request', f'{name} is required'
                )
        dpop_jkt = None
        proofs = request.headers.getlist('dpop')
        if proofs:
            from a2a.extensions.delegation.dpop import (  # noqa: PLC0415
                DPoPError,
                verify_proof,
            )

            if len(proofs) != 1:
                raise TokenExchangeError(
                    'invalid_dpop_proof', 'exactly one DPoP proof is allowed'
                )
            try:
                dpop_jkt = verify_proof(
                    proofs[0],
                    method=request.method,
                    url=str(request.url),
                    replay_cache=replay_cache,
                    now=service.now(),
                ).jkt
            except DPoPError as e:
                raise TokenExchangeError('invalid_dpop_proof', str(e)) from e
        return service.exchange(
            actor_id=credentials[0],
            actor_secret=credentials[1],
            subject_token=fields['subject_token'],
            subject_token_type=fields.get(
                'subject_token_type', ACCESS_TOKEN_TYPE
            ),
            audience=fields['audience'],
            scope=fields.get('scope') or None,
            dpop_jkt=dpop_jkt,
        )

    async def token(request: Request) -> JSONResponse:
        try:
            issued = await exchange(request)
        except TokenExchangeError as e:
            return error(e)
        return JSONResponse(issued.to_dict(), headers=no_store)

    async def jwks(_: Request) -> JSONResponse:
        return JSONResponse(service.jwks())

    return Starlette(
        routes=[
            Route(token_path, token, methods=['POST']),
            Route(jwks_path, jwks, methods=['GET']),
        ]
    )
