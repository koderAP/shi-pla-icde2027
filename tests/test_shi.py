"""Byte-identity and lookup tests for the canonical PLA index."""

from __future__ import annotations

import random
import unittest

from forgetful_index.index import SHIIndex
from forgetful_index.pla import build_segments, is_anchor


def _lognormal(n: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    keys = {int(rng.lognormvariate(0, 1) * 1_000_000) for _ in range(n * 3)}
    out = sorted(keys)
    if len(out) < n:
        raise RuntimeError("not enough distinct keys")
    return out[:n]


class PLATests(unittest.TestCase):
    def test_prediction_error_within_epsilon(self) -> None:
        keys = _lognormal(4_000, 1)
        index = SHIIndex.build(keys, epsilon=16, secret=7, modulus=128)
        self.assertLessEqual(index.max_prediction_error(), 16 + 1e-6)
        for rank, key in enumerate(keys):
            self.assertEqual(index.lookup(key), rank)

    def test_lookup_every_key(self) -> None:
        keys = _lognormal(3_000, 2)
        index = SHIIndex.build(keys, epsilon=8, secret=3, modulus=64, max_levels=6)
        for rank, key in enumerate(keys):
            self.assertEqual(index.lookup(key), rank)
        self.assertIsNone(index.lookup(keys[0] - 1))
        self.assertIsNone(index.lookup(keys[-1] + 1))

    def test_anchors_are_segment_starts(self) -> None:
        keys = list(range(2_000))
        secret, modulus = 11, 32
        segments = build_segments(keys, epsilon=10_000, secret=secret, modulus=modulus)
        starts = {seg.start_key for seg in segments}
        for key in keys:
            if is_anchor(key, secret, modulus):
                self.assertIn(key, starts)

    def test_delete_matches_rebuild(self) -> None:
        keys = _lognormal(2_500, 3)
        index = SHIIndex.build(keys, epsilon=16, secret=9, modulus=256)
        rng = random.Random(0)
        for _ in range(40):
            key = rng.choice(index.keys)
            expected = [k for k in index.keys if k != key]
            fresh = SHIIndex.build(expected, epsilon=16, secret=9, modulus=256)
            index.delete(key)
            self.assertEqual(index.repr_bytes(), fresh.repr_bytes())
            self.assertEqual(index.keys, expected)

    def test_insert_matches_rebuild(self) -> None:
        keys = _lognormal(1_500, 4)
        index = SHIIndex.build(keys[:1000], epsilon=16, secret=4, modulus=128)
        for key in keys[1000:1100]:
            fresh_keys = sorted(index.keys + [key])
            fresh = SHIIndex.build(fresh_keys, epsilon=16, secret=4, modulus=128)
            index.insert(key)
            self.assertEqual(index.repr_bytes(), fresh.repr_bytes())

    def test_operation_sequence_is_history_independent(self) -> None:
        universe = _lognormal(800, 5)
        rng = random.Random(1)
        present: set[int] = set(rng.sample(universe, 200))
        index = SHIIndex.build(sorted(present), epsilon=8, secret=1, modulus=64, max_levels=4)
        for _ in range(300):
            if present and rng.random() < 0.5:
                key = rng.choice(tuple(present))
                present.remove(key)
                index.delete(key)
            else:
                missing = [k for k in universe if k not in present]
                key = rng.choice(missing)
                present.add(key)
                index.insert(key)
            fresh = SHIIndex.build(
                sorted(present), epsilon=8, secret=1, modulus=64, max_levels=4
            )
            self.assertEqual(index.repr_bytes(), fresh.repr_bytes())

    def test_convex_cascade_is_cut_by_anchors(self) -> None:
        keys = [i * i for i in range(1, 4001)]
        epsilon = 16
        secret = 5
        bare = SHIIndex.build(keys, epsilon=epsilon, secret=secret, modulus=None)
        anchored = SHIIndex.build(keys, epsilon=epsilon, secret=secret, modulus=256)
        # Delete a key near the front so the unanchored cascade runs to the end.
        # The first key sits in the first segment, so an unanchored delete
        # shifts every later boundary on i^2 (the convex lower bound).
        target = keys[0]
        bare_refit = _refit_count(bare, target)
        anchored_refit = _refit_count(anchored, target)
        self.assertEqual(bare_refit, len(bare.segments))
        self.assertGreater(bare_refit, anchored_refit)
        self.assertLessEqual(anchored_refit, 8)

    def test_no_anchors_convex_still_byte_identical(self) -> None:
        keys = [i * i for i in range(1, 801)]
        index = SHIIndex.build(keys, epsilon=16, secret=0, modulus=None)
        key = keys[25]
        fresh = SHIIndex.build([k for k in keys if k != key], epsilon=16, secret=0, modulus=None)
        index.delete(key)
        self.assertEqual(index.repr_bytes(), fresh.repr_bytes())


def _refit_count(index: SHIIndex, key: int) -> int:
    clone = SHIIndex.build(
        list(index.keys),
        epsilon=index.epsilon,
        secret=index.secret,
        modulus=index.modulus,
        max_levels=index.max_levels,
    )
    return clone.delete(key)


if __name__ == "__main__":
    unittest.main()
