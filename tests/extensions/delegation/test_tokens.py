import base64
import hashlib
import time

from typing import Any

import jwt
import pytest

from a2a.extensions.delegation.tokens import (
    DelegationTokenError,
    KeySet,
    SigningKey,
    actor_chain,
    jwk_thumbprint,
    scopes_of,
    verify_token,
)


ISSUER = 'https://sts.example.com'


def _claims(**overrides: Any) -> dict[str, Any]:
    now = int(time.time())
    claims: dict[str, Any] = {
        'iss': ISSUER,
        'sub': 'user:bayard',
        'aud': 'agent:payments',
        'scope': 'payments:read payments:refund',
        'iat': now,
        'exp': now + 300,
        'jti': 'abc',
    }
    claims.update(overrides)
    return {k: v for k, v in claims.items() if v is not None}


@pytest.fixture
def key() -> SigningKey:
    return SigningKey.generate()


def _verify(token: str, key: SigningKey, **kw: Any) -> dict[str, Any]:
    return verify_token(
        token,
        KeySet.from_signing_keys(key),
        issuer=kw.get('issuer', ISSUER),
        audience=kw.get('audience', 'agent:payments'),
    )


def test_valid_token_round_trips(key: SigningKey) -> None:
    claims = _verify(key.sign(_claims()), key)

    assert claims['sub'] == 'user:bayard'
    assert scopes_of(claims) == {'payments:read', 'payments:refund'}


def test_jwks_round_trip_and_rfc7638_kid(key: SigningKey) -> None:
    jwks = {'keys': [key.public_jwk()]}
    keys = KeySet.from_jwks(jwks)

    assert keys.get(key.kid) is not None
    assert key.kid == jwk_thumbprint(key.public_jwk())
    assert verify_token(
        key.sign(_claims()), keys, issuer=ISSUER, audience='agent:payments'
    )


def test_rfc7638_thumbprint_uses_canonical_members() -> None:
    # EC key from RFC 7517 Appendix A.1. RFC 7638 §3.2: hash the required
    # members (crv, kty, x, y) in lexicographic order, no whitespace; other
    # members such as kid/use/alg must not affect the result.
    x = 'MKBCTNIcKUSDii11ySs3526iDZ8AiTo7Tu6KPAqv7D4'
    y = '4Etl6SRW2YiLUrN5vfvVHuhp7x8PxltmWWlbbM4IFyM'
    canonical = f'{{"crv":"P-256","kty":"EC","x":"{x}","y":"{y}"}}'
    expected = (
        base64.urlsafe_b64encode(hashlib.sha256(canonical.encode()).digest())
        .rstrip(b'=')
        .decode()
    )
    jwk = {
        'y': y,
        'use': 'sig',
        'kid': '1',
        'x': x,
        'crv': 'P-256',
        'kty': 'EC',
    }

    assert jwk_thumbprint(jwk) == expected


def test_jwks_skips_unusable_keys(key: SigningKey) -> None:
    rsa_like = {'kty': 'RSA', 'kid': 'r', 'n': 'x', 'e': 'AQAB'}
    no_kid = {k: v for k, v in key.public_jwk().items() if k != 'kid'}

    keys = KeySet.from_jwks({'keys': [rsa_like, no_kid]})

    assert keys.get('r') is None
    assert keys.get(key.kid) is None


@pytest.mark.parametrize(
    ('overrides', 'code'),
    [
        ({'iss': 'https://evil.example.com'}, 'wrong_issuer'),
        ({'aud': 'agent:travel'}, 'wrong_audience'),
        ({'exp': int(time.time()) - 120}, 'expired'),
        ({'iat': int(time.time()) + 600}, 'expired'),
        ({'jti': None}, 'invalid_token'),
        ({'aud': ['agent:payments', 'agent:other']}, 'invalid_token'),
        ({'act': {'nope': 1}}, 'invalid_token'),
    ],
)
def test_bad_claims_rejected(
    key: SigningKey, overrides: dict[str, Any], code: str
) -> None:
    with pytest.raises(DelegationTokenError) as e:
        _verify(key.sign(_claims(**overrides)), key)
    assert e.value.code == code


def test_small_clock_skew_tolerated(key: SigningKey) -> None:
    token = key.sign(_claims(exp=int(time.time()) - 30))

    assert _verify(token, key)['sub'] == 'user:bayard'


def test_token_from_unknown_key_rejected(key: SigningKey) -> None:
    other = SigningKey.generate()

    with pytest.raises(DelegationTokenError, match='unknown signing key'):
        _verify(other.sign(_claims()), key)


def test_forged_kid_with_wrong_key_rejected(key: SigningKey) -> None:
    attacker = SigningKey(
        private_key=SigningKey.generate().private_key, kid=key.kid
    )

    with pytest.raises(DelegationTokenError) as e:
        _verify(attacker.sign(_claims()), key)
    assert e.value.code == 'invalid_token'


@pytest.mark.parametrize('alg', ['HS256', 'none'])
def test_non_es256_algorithms_rejected(key: SigningKey, alg: str) -> None:
    secret = 'k' * 32 if alg == 'HS256' else None
    token = jwt.encode(
        _claims(), secret, algorithm=alg, headers={'kid': key.kid}
    )

    with pytest.raises(DelegationTokenError, match='not allowed'):
        _verify(token, key)


def test_garbage_rejected(key: SigningKey) -> None:
    with pytest.raises(DelegationTokenError, match='malformed'):
        _verify('not-a-jwt', key)


def test_actor_chain_most_recent_first() -> None:
    claims = {
        'act': {'sub': 'agent:travel', 'act': {'sub': 'agent:orchestrator'}}
    }

    assert actor_chain(claims) == ['agent:travel', 'agent:orchestrator']
    assert actor_chain({}) == []


def test_scope_must_be_string() -> None:
    with pytest.raises(DelegationTokenError):
        scopes_of({'scope': ['a']})
