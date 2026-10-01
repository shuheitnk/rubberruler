"""Seeds for every random draw, derived from the master seed."""

import hashlib
import json

import numpy as np


def derive_seed(domain: str, seed: int, *keys: str | int) -> int:
    """Derive the seed for one use of randomness from the master seed ``seed``.

    The value is in [0, 2**32 - 1]. It depends only on the arguments, so it is the
    same in every process, for any number of workers, and in any order of execution.
    Integers in ``keys`` must be Python ints (convert NumPy integers with ``int()``).
    """
    payload = json.dumps(
        [domain, seed, *keys], ensure_ascii=False, separators=(",", ":")
    )
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "little")


def generator(domain: str, seed: int, *keys: str | int) -> np.random.Generator:
    """Return ``numpy.random.default_rng(derive_seed(domain, seed, *keys))``."""
    return np.random.default_rng(derive_seed(domain, seed, *keys))
