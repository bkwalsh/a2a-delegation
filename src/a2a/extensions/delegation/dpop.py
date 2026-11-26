"""DPoP sender-constrained tokens (RFC 9449) for the delegation extension.

A DPoP-bound token carries ``cnf.jkt``, the thumbprint of its holder's key.
Every request using it must include a fresh proof JWT signed by that key and
bound to the request's method, URL and the token itself, so a stolen token is
useless without the private key. See ``docs/extensions/delegation/v1.md`` §8.
"""

from __future__ import annotations

import hashlib
import json
import time

from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from a2a.extensions.delegation.tokens import (
    ALGORITHM,
    b64url,
    jwk_thumbprint,
    new_jti,
    public_jwk,
)


try:
    import httpx
    import jwt

    from cryptography.hazmat.primitives.asymmetric import ec
    from jwt.algorithms import ECAlgorithm
except ImportError as e:
    raise ImportError(
        'DPoP support requires PyJWT, cryptography and httpx. '
        "Install with: 'pip install a2a-sdk[delegation]'"
    ) from e

if TYPE_CHECKING:
    from collections.abc import Callable, Generator


DPOP_HEADER = 'DPoP'
PROOF_TYPE = 'dpop+jwt'
DEFAULT_MAX_AGE_SECONDS = 60


class DPoPError(Exception):
    """A DPoP proof is missing, invalid, replayed or not bound to the token."""


def access_token_hash(token: str) -> str:
    """The ``ath`` claim: base64url(SHA-256(access token))."""
    return b64url(hashlib.sha256(token.encode('ascii')).digest())


_DEFAULT_PORTS = {'http': 80, 'https': 443}


def normalize_htu(url: str | httpx.URL) -> str:
    """The ``htu`` value for a URL (RFC 9449 §4.3).

    Applies RFC 3986 syntax-based normalization: lower-case scheme and host,
    default port omitted, empty path as ``/``, and no query or fragment.
    """
    u = httpx.URL(str(url))
    scheme = u.scheme.lower()
    port = '' if u.port in (None, _DEFAULT_PORTS.get(scheme)) else f':{u.port}'
    return f'{scheme}://{u.host.lower()}{port}{u.path or "/"}'


@dataclass(frozen=True)
class DPoPKey:
    """A holder's ES256 key for creating DPoP proofs."""

    private_key: ec.EllipticCurvePrivateKey

    @classmethod
    def generate(cls) -> DPoPKey:
        """Generates a new P-256 key."""
        return cls(ec.generate_private_key(ec.SECP256R1()))

    @property
    def public_jwk(self) -> dict[str, Any]:
        """The public key as a JWK."""
        return public_jwk(self.private_key.public_key())

    @property
    def thumbprint(self) -> str:
        """The RFC 7638 thumbprint, used as ``cnf.jkt``."""
        return jwk_thumbprint(self.public_jwk)

    def proof(
        self,
        method: str,
        url: str | httpx.URL,
        *,
        access_token: str | None = None,
        now: float | None = None,
    ) -> str:
        """Creates a proof for one request."""
        claims: dict[str, Any] = {
            'jti': new_jti(),
            'htm': method.upper(),
            'htu': normalize_htu(url),
            'iat': int(time.time() if now is None else now),
        }
        if access_token is not None:
            claims['ath'] = access_token_hash(access_token)
        return jwt.encode(
            claims,
            self.private_key,
            algorithm=ALGORITHM,
            headers={'typ': PROOF_TYPE, 'jwk': self.public_jwk},
        )


class ReplayCache:
    """Remembers proof ``jti`` values until they can no longer be fresh.

    In-memory and per process. A deployment with several replicas needs a
    shared store (for example Redis with TTLs) behind the same interface.
    """

    def __init__(self, max_entries: int = 100_000) -> None:
        self._seen: OrderedDict[str, float] = OrderedDict()
        self._max = max_entries

    def check_and_add(self, jti: str, expires_at: float, now: float) -> bool:
        """Records ``jti``; returns False if it was already seen."""
        while self._seen:
            oldest_jti, oldest_exp = next(iter(self._seen.items()))
            if oldest_exp > now and len(self._seen) < self._max:
                break
            del self._seen[oldest_jti]
        if jti in self._seen:
            return False
        self._seen[jti] = expires_at
        return True


@dataclass(frozen=True)
class VerifiedProof:
    """A proof that passed verification."""

    jkt: str
    jti: str


def _decode_proof(proof: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Checks the proof header and signature; returns (jwk, claims)."""
    try:
        header = jwt.get_unverified_header(proof)
    except jwt.PyJWTError as e:
        raise DPoPError('malformed proof') from e
    if header.get('typ') != PROOF_TYPE:
        raise DPoPError('typ must be dpop+jwt')
    if header.get('alg') != ALGORITHM:
        raise DPoPError(f'algorithm {header.get("alg")!r} not allowed')
    jwk = header.get('jwk')
    if not isinstance(jwk, dict) or jwk.get('kty') != 'EC' or 'd' in jwk:
        raise DPoPError('proof must carry a public EC jwk')
    try:
        key = ECAlgorithm.from_jwk(json.dumps(jwk))
    except (jwt.PyJWTError, ValueError) as e:
        raise DPoPError('proof must carry a public EC jwk') from e
    if not isinstance(key, ec.EllipticCurvePublicKey):
        raise DPoPError('proof must carry a public EC jwk')
    try:
        claims = jwt.decode(
            proof,
            key,
            algorithms=[ALGORITHM],
            options={'require': ['jti', 'htm', 'htu', 'iat']},
        )
    except jwt.PyJWTError as e:
        raise DPoPError('invalid proof signature') from e
    return jwk, claims


def verify_proof(  # noqa: PLR0913
    proof: str,
    *,
    method: str,
    url: str | httpx.URL,
    replay_cache: ReplayCache,
    access_token: str | None = None,
    expected_jkt: str | None = None,
    max_age: int = DEFAULT_MAX_AGE_SECONDS,
    now: float | None = None,
) -> VerifiedProof:
    """Verifies a DPoP proof (RFC 9449 §4.3).

    Raises:
        DPoPError: Describing the first check that failed.
    """
    now = time.time() if now is None else now
    jwk, claims = _decode_proof(proof)

    if claims['htm'] != method.upper():
        raise DPoPError('htm does not match the request method')
    if claims['htu'] != normalize_htu(url):
        raise DPoPError('htu does not match the request URL')
    iat = claims['iat']
    if not isinstance(iat, int | float) or isinstance(iat, bool):
        raise DPoPError('iat must be a number')
    if iat > now + 5 or iat < now - max_age:
        raise DPoPError('proof is not fresh')
    if access_token is not None and claims.get('ath') != access_token_hash(
        access_token
    ):
        raise DPoPError('ath does not match the access token')

    jkt = jwk_thumbprint(jwk)
    if expected_jkt is not None and jkt != expected_jkt:
        raise DPoPError('proof key does not match the token binding')
    # Checked last, so a proof that fails other checks cannot burn its jti.
    if not replay_cache.check_and_add(str(claims['jti']), iat + max_age, now):
        raise DPoPError('proof jti has already been used')
    return VerifiedProof(jkt=jkt, jti=str(claims['jti']))


class DPoPSigner(httpx.Auth):
    """httpx auth that adds a fresh DPoP proof to DPoP-authorized requests.

    Attach it to the httpx client an agent uses for A2A calls. Any request
    carrying ``Authorization: DPoP <token>`` gets a proof for its exact method
    and URL, so it works for every A2A transport binding.
    """

    def __init__(
        self, key: DPoPKey, clock: Callable[[], float] = time.time
    ) -> None:
        self.key = key
        self._clock = clock

    def auth_flow(
        self, request: httpx.Request
    ) -> Generator[httpx.Request, httpx.Response, None]:
        """Adds the DPoP header when the request uses a DPoP token."""
        scheme, _, token = request.headers.get('authorization', '').partition(
            ' '
        )
        if (
            scheme.lower() == 'dpop'
            and token
            and DPOP_HEADER not in (request.headers)
        ):
            request.headers[DPOP_HEADER] = self.key.proof(
                request.method,
                request.url,
                access_token=token,
                now=self._clock(),
            )
        yield request
