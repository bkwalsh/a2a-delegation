"""Delegation tokens: signing keys, JWKS, verification and chain parsing.

See ``docs/extensions/delegation/v1.md`` §4.
"""

from __future__ import annotations

import base64
import hashlib
import json
import time
import uuid

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any


try:
    import jwt

    from cryptography.hazmat.primitives.asymmetric import ec
    from jwt.algorithms import ECAlgorithm
except ImportError as e:
    raise ImportError(
        'The A2A delegation extension requires PyJWT and cryptography. '
        "Install with: 'pip install a2a-sdk[delegation]'"
    ) from e

if TYPE_CHECKING:
    from collections.abc import Mapping


ALGORITHM = 'ES256'
"""The only signing algorithm delegation tokens may use."""

DEFAULT_LEEWAY_SECONDS = 60
"""Maximum clock skew tolerated when checking ``exp`` and ``iat``."""

ACCESS_TOKEN_TYPE = 'urn:ietf:params:oauth:token-type:access_token'  # noqa: S105
TOKEN_EXCHANGE_GRANT = 'urn:ietf:params:oauth:grant-type:token-exchange'  # noqa: S105


class DelegationTokenError(Exception):
    """A delegation token failed verification.

    Attributes:
        code: The error code from spec §7 (for example ``wrong_audience``).
    """

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f'{code}: {detail}')
        self.code = code
        self.detail = detail


class UnknownKeyError(DelegationTokenError):
    """The token's ``kid`` is not in the key set (keys may have rotated)."""

    def __init__(self) -> None:
        super().__init__('invalid_token', 'unknown signing key')


def b64url(data: bytes) -> str:
    """Base64url-encodes bytes without padding."""
    return base64.urlsafe_b64encode(data).rstrip(b'=').decode('ascii')


def jwk_thumbprint(jwk: Mapping[str, Any]) -> str:
    """Computes the RFC 7638 SHA-256 thumbprint of an EC public JWK."""
    members = {k: jwk[k] for k in ('crv', 'kty', 'x', 'y')}
    canonical = json.dumps(members, separators=(',', ':'), sort_keys=True)
    return b64url(hashlib.sha256(canonical.encode('utf-8')).digest())


def public_jwk(public_key: ec.EllipticCurvePublicKey) -> dict[str, Any]:
    """Exports an EC P-256 public key as a JWK."""
    jwk = ECAlgorithm.to_jwk(public_key, as_dict=True)
    return {k: jwk[k] for k in ('kty', 'crv', 'x', 'y')}


@dataclass(frozen=True)
class SigningKey:
    """An ES256 signing key with an RFC 7638 key id."""

    private_key: ec.EllipticCurvePrivateKey
    kid: str

    @classmethod
    def generate(cls) -> SigningKey:
        """Generates a new P-256 key."""
        key = ec.generate_private_key(ec.SECP256R1())
        return cls(
            private_key=key, kid=jwk_thumbprint(public_jwk(key.public_key()))
        )

    def public_jwk(self) -> dict[str, Any]:
        """The public half as a JWK, including ``kid``, ``alg`` and ``use``."""
        return {
            **public_jwk(self.private_key.public_key()),
            'kid': self.kid,
            'alg': ALGORITHM,
            'use': 'sig',
        }

    def sign(self, claims: Mapping[str, Any]) -> str:
        """Signs claims as a compact JWS."""
        return jwt.encode(
            dict(claims),
            self.private_key,
            algorithm=ALGORITHM,
            headers={'kid': self.kid, 'typ': 'JWT'},
        )


class KeySet:
    """Public verification keys, indexed by ``kid``."""

    def __init__(self, keys: Mapping[str, ec.EllipticCurvePublicKey]) -> None:
        self._keys = dict(keys)

    @classmethod
    def from_jwks(cls, jwks: Mapping[str, Any]) -> KeySet:
        """Builds a key set from a JWKS document, keeping only ES256 keys."""
        keys: dict[str, ec.EllipticCurvePublicKey] = {}
        for jwk in jwks.get('keys', []):
            if jwk.get('kty') != 'EC' or jwk.get('crv') != 'P-256':
                continue
            if 'kid' not in jwk:
                continue
            key = ECAlgorithm.from_jwk(json.dumps(jwk))
            if isinstance(key, ec.EllipticCurvePublicKey):
                keys[jwk['kid']] = key
        return cls(keys)

    @classmethod
    def from_signing_keys(cls, *keys: SigningKey) -> KeySet:
        """Builds a key set from local signing keys (for an in-process STS)."""
        return cls({k.kid: k.private_key.public_key() for k in keys})

    def get(self, kid: str) -> ec.EllipticCurvePublicKey | None:
        """Returns the key for ``kid``, if known."""
        return self._keys.get(kid)


def new_jti() -> str:
    """Returns a fresh unique token id."""
    return uuid.uuid4().hex


def verify_token(  # noqa: PLR0913
    token: str,
    keys: KeySet,
    *,
    issuer: str,
    audience: str,
    leeway: int = DEFAULT_LEEWAY_SECONDS,
    now: float | None = None,
) -> dict[str, Any]:
    """Verifies a delegation token and returns its claims.

    Checks the signature (ES256 only, key chosen by ``kid``), ``iss``, ``aud``,
    ``exp`` and ``iat``. Chain and scope policy are checked by the caller.
    ``now`` defaults to the current time; times are checked against it with
    ``leeway`` seconds of tolerance.

    Raises:
        DelegationTokenError: With a spec §7 code describing the failure.
    """
    try:
        header = jwt.get_unverified_header(token)
    except jwt.PyJWTError as e:
        raise DelegationTokenError('invalid_token', 'malformed token') from e
    if header.get('alg') != ALGORITHM:
        raise DelegationTokenError(
            'invalid_token', f'algorithm {header.get("alg")!r} not allowed'
        )
    key = keys.get(str(header.get('kid', '')))
    if key is None:
        raise UnknownKeyError
    try:
        claims = jwt.decode(
            token,
            key,
            algorithms=[ALGORITHM],
            issuer=issuer,
            audience=audience,
            options={
                'require': ['iss', 'sub', 'aud', 'exp', 'iat', 'jti'],
                # Times are checked below against an injectable clock.
                'verify_exp': False,
                'verify_iat': False,
                'verify_nbf': False,
            },
        )
    except jwt.InvalidIssuerError as e:
        raise DelegationTokenError('wrong_issuer', str(e)) from e
    except jwt.InvalidAudienceError as e:
        raise DelegationTokenError('wrong_audience', str(e)) from e
    except jwt.PyJWTError as e:
        raise DelegationTokenError('invalid_token', str(e)) from e
    _check_times(claims, time.time() if now is None else now, leeway)
    if not isinstance(claims.get('aud'), str):
        raise DelegationTokenError(
            'invalid_token', 'aud must be a single string'
        )
    actor_chain(claims)  # validates the shape of `act`
    return claims


def _check_times(claims: Mapping[str, Any], now: float, leeway: int) -> None:
    for name in ('exp', 'iat'):
        value = claims[name]
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise DelegationTokenError(
                'invalid_token', f'{name} must be a number'
            )
    if claims['exp'] <= now - leeway:
        raise DelegationTokenError('expired', 'token has expired')
    if claims['iat'] > now + leeway:
        raise DelegationTokenError('expired', 'token issued in the future')


def actor_chain(claims: Mapping[str, Any]) -> list[str]:
    """Returns the actors in a token's ``act`` chain, most recent first.

    Raises:
        DelegationTokenError: If ``act`` is malformed.
    """
    chain: list[str] = []
    act = claims.get('act')
    while act is not None:
        if not isinstance(act, dict) or not isinstance(act.get('sub'), str):
            raise DelegationTokenError('invalid_token', 'malformed act claim')
        chain.append(act['sub'])
        act = act.get('act')
    return chain


def scopes_of(claims: Mapping[str, Any]) -> frozenset[str]:
    """Returns the token's scopes as a set."""
    scope = claims.get('scope', '')
    if not isinstance(scope, str):
        raise DelegationTokenError('invalid_token', 'scope must be a string')
    return frozenset(scope.split())
