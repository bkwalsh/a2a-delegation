"""Reference Security Token Service (STS) for the delegation extension.

Issues root tokens and performs RFC 8693 token exchange with the narrowing
rules in ``docs/extensions/delegation/v1.md`` §5.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import time

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qsl, quote_plus, unquote_plus

import jwt

from cryptography.hazmat.primitives.asymmetric import ec
from jwt.algorithms import ECAlgorithm

from a2a.extensions.delegation.tokens import (
    ACCESS_TOKEN_TYPE,
    ALGORITHM,
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

    from a2a.extensions.delegation.dpop import ReplayStore


DEFAULT_TTL_SECONDS = 300
MAX_ASSERTION_LIFETIME_SECONDS = 300
CLIENT_ASSERTION_TYPE = 'urn:ietf:params:oauth:client-assertion-type:jwt-bearer'


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

    An agent authenticates to the token endpoint with a client secret, or
    (preferred) with a ``private_key_jwt`` assertion signed by the key whose
    public half is registered here (RFC 7523 §2.2).

    Attributes:
        agent_id: The identifier used as ``aud`` and in ``act.sub``.
        accepted_scopes: Scopes this agent may receive as an audience.
        secret_sha256: SHA-256 hex digest of the agent's client secret.
        assertion_jwk: Public EC JWK that verifies the agent's assertions.
    """

    agent_id: str
    accepted_scopes: frozenset[str]
    secret_sha256: str | None = None
    assertion_jwk: dict[str, Any] | None = None

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

    @classmethod
    def with_key(
        cls,
        agent_id: str,
        public_jwk: dict[str, Any],
        accepted_scopes: Iterable[str],
    ) -> RegisteredAgent:
        """Registers an agent that authenticates with ``private_key_jwt``."""
        return cls(
            agent_id=agent_id,
            assertion_jwk=dict(public_jwk),
            accepted_scopes=frozenset(accepted_scopes),
        )

    def check_secret(self, secret: str) -> bool:
        """Constant-time comparison of a presented secret."""
        if self.secret_sha256 is None:
            return False
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

        Authenticates the actor with a client secret; see
        :meth:`exchange_for` for actors authenticated another way.

        Raises:
            TokenExchangeError: If authentication or any rule in spec §5
                fails.
        """
        return self.exchange_for(
            self.authenticate_secret(actor_id, actor_secret),
            subject_token=subject_token,
            audience=audience,
            scope=scope,
            subject_token_type=subject_token_type,
            dpop_jkt=dpop_jkt,
        )

    def exchange_for(  # noqa: PLR0913
        self,
        actor: RegisteredAgent,
        *,
        subject_token: str,
        audience: str,
        scope: str | None = None,
        subject_token_type: str = ACCESS_TOKEN_TYPE,
        dpop_jkt: str | None = None,
    ) -> IssuedToken:
        """Performs an RFC 8693 token exchange for an authenticated actor.

        ``actor`` must come from :meth:`authenticate_secret` or
        :meth:`authenticate_assertion`. If ``dpop_jkt`` is given (the
        thumbprint of a key the actor proved it holds), the issued token is
        bound to that key (RFC 9449 §6).

        Raises:
            TokenExchangeError: If any rule in spec §5 is violated.
        """
        if self._agents.get(actor.agent_id) is not actor:
            raise TokenExchangeError(
                'invalid_client', 'actor is not a registered agent', 401
            )
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

    def authenticate_secret(
        self, agent_id: str, secret: str
    ) -> RegisteredAgent:
        """Authenticates an agent by client secret.

        Raises:
            TokenExchangeError: ``invalid_client`` on any failure.
        """
        agent = self._agents.get(agent_id)
        if agent is None or not agent.check_secret(secret):
            raise TokenExchangeError(
                'invalid_client', 'client authentication failed', status=401
            )
        return agent

    async def authenticate_assertion(
        self,
        assertion: str,
        *,
        token_url: str,
        replay_cache: ReplayStore,
        client_id: str | None = None,
    ) -> RegisteredAgent:
        """Authenticates an agent by ``private_key_jwt`` (RFC 7523 §3).

        The assertion must be an ES256 JWT signed by the agent's registered
        key, with ``iss`` = ``sub`` = the agent id, ``aud`` = this token
        endpoint, a lifetime of at most five minutes, and a ``jti`` that has
        not been used before.

        Raises:
            TokenExchangeError: ``invalid_client`` on any failure.
        """

        def fail(detail: str) -> TokenExchangeError:
            return TokenExchangeError('invalid_client', detail, status=401)

        try:
            unverified = jwt.decode(
                assertion, options={'verify_signature': False}
            )
        except jwt.PyJWTError as e:
            raise fail('malformed client assertion') from e
        agent = self._agents.get(str(unverified.get('iss', '')))
        if agent is None or agent.assertion_jwk is None:
            raise fail('unknown client or no registered key')
        key = ECAlgorithm.from_jwk(json.dumps(agent.assertion_jwk))
        if not isinstance(key, ec.EllipticCurvePublicKey):
            raise fail('registered key must be a public EC key')
        now = self._clock()
        try:
            claims = jwt.decode(
                assertion,
                key,
                algorithms=[ALGORITHM],
                audience=token_url,
                options={
                    'require': ['iss', 'sub', 'aud', 'exp', 'iat', 'jti'],
                    'verify_exp': False,
                    'verify_iat': False,
                },
            )
        except jwt.PyJWTError as e:
            raise fail(f'invalid client assertion: {e}') from e
        if claims['sub'] != agent.agent_id:
            raise fail('assertion sub must equal iss')
        if client_id is not None and client_id != agent.agent_id:
            raise fail('client_id does not match the assertion')
        exp, iat = claims['exp'], claims['iat']
        if not all(
            isinstance(v, int | float) and not isinstance(v, bool)
            for v in (exp, iat)
        ):
            raise fail('invalid client assertion: exp and iat must be numbers')
        if exp <= now or iat > now + 5:
            raise fail('client assertion expired or not yet valid')
        if exp - iat > MAX_ASSERTION_LIFETIME_SECONDS:
            raise fail('client assertion lifetime too long')
        if not await replay_cache.check_and_add(
            f'assertion:{claims["jti"]}', exp, now
        ):
            raise fail('client assertion has already been used')
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
    token_url: str | None = None,
    replay_cache: ReplayStore | None = None,
) -> Starlette:
    """Builds a Starlette app serving the token endpoint and JWKS.

    Args:
        service: The token service.
        token_path: Route of the token endpoint.
        jwks_path: Route of the JWKS document.
        token_url: The public URL of the token endpoint. Client assertions
            must name it as ``aud`` and DPoP proofs as ``htu``. Set it
            whenever the STS runs behind a proxy; defaults to the URL the
            server sees.
        replay_cache: Store for DPoP proof and client assertion ids. Share it
            across replicas (see ``RedisReplayCache``); defaults to an
            in-memory cache for a single process.

    Requires the ``http-server`` extra.
    """
    from starlette.applications import Starlette  # noqa: PLC0415
    from starlette.responses import JSONResponse  # noqa: PLC0415
    from starlette.routing import Route  # noqa: PLC0415

    from a2a.extensions.delegation.dpop import (  # noqa: PLC0415
        DPoPError,
        ReplayCache,
        verify_proof,
    )

    no_store = {'Cache-Control': 'no-store'}
    replays: ReplayStore = replay_cache or ReplayCache()

    def error(e: TokenExchangeError) -> JSONResponse:
        headers = dict(no_store)
        if e.status == 401:  # noqa: PLR2004
            headers['WWW-Authenticate'] = 'Basic realm="delegation-sts"'
        return JSONResponse(
            {'error': e.error, 'error_description': e.description},
            status_code=e.status,
            headers=headers,
        )

    async def authenticate(
        request: Request, fields: dict[str, str], url: str
    ) -> RegisteredAgent:
        basic = parse_basic_auth(request.headers.get('authorization'))
        assertion_type = fields.get('client_assertion_type')
        if assertion_type is not None:
            if basic is not None:
                raise TokenExchangeError(
                    'invalid_request', 'use one client authentication method'
                )
            if assertion_type != CLIENT_ASSERTION_TYPE:
                raise TokenExchangeError(
                    'invalid_client', 'unsupported client_assertion_type', 401
                )
            return await service.authenticate_assertion(
                fields.get('client_assertion', ''),
                token_url=url,
                replay_cache=replays,
                client_id=fields.get('client_id'),
            )
        if basic is None:
            raise TokenExchangeError(
                'invalid_client', 'client authentication required', 401
            )
        return service.authenticate_secret(*basic)

    async def exchange(request: Request) -> IssuedToken:
        url = token_url or str(request.url)
        content_type = request.headers.get('content-type', '')
        if not content_type.startswith('application/x-www-form-urlencoded'):
            if parse_basic_auth(request.headers.get('authorization')) is None:
                raise TokenExchangeError(
                    'invalid_client', 'client authentication required', 401
                )
            raise TokenExchangeError(
                'invalid_request', 'body must be form-urlencoded'
            )
        fields = dict(parse_qsl((await request.body()).decode()))
        actor = await authenticate(request, fields, url)
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
            if len(proofs) != 1:
                raise TokenExchangeError(
                    'invalid_dpop_proof', 'exactly one DPoP proof is allowed'
                )
            try:
                proof = await verify_proof(
                    proofs[0],
                    method=request.method,
                    url=url,
                    replay_cache=replays,
                    now=service.now(),
                )
            except DPoPError as e:
                raise TokenExchangeError('invalid_dpop_proof', str(e)) from e
            dpop_jkt = proof.jkt
        return service.exchange_for(
            actor,
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
