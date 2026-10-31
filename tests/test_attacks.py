"""T1 retention attacks, T2 diff attacks, and the DP relaxation."""

from __future__ import annotations

import random
import unittest

import numpy as np

from forgetful_index.attacks import (
    TombstoneIndex,
    candidate_rebuild_identifies,
    residual_jump,
    segment_start_exposed,
    snapshot_diff_result,
)
from forgetful_index.data import convex_keys, lognormal_keys
from forgetful_index.dp import attack_advantage, bin_of, diff_bin, histogram, release_counts
from forgetful_index.index import SHIIndex
from forgetful_index.pla import absolute_suffix_changed, build_segments


class AttackTests(unittest.TestCase):
    def test_tombstone_retains_every_deleted_key(self) -> None:
        keys = list(range(500))
        index = TombstoneIndex(keys, epsilon=16)
        deleted = [10, 20, 30, 400]
        for key in deleted:
            index.delete(key)
        self.assertEqual(index.recover(), deleted)
        blob = index.repr_bytes()
        canonical = SHIIndex.build([k for k in keys if k not in deleted], epsilon=16)
        for key in deleted:
            self.assertIn(key.to_bytes(8, "little", signed=True), blob)
            self.assertNotIn(key, [seg.start_key for seg in canonical.segments])

    def test_residual_recovers_dense_integers(self) -> None:
        keys = list(range(2_000))
        rng = random.Random(0)
        hits = 0
        trials = 25
        for key in rng.sample(keys[1:-1], trials):
            result = residual_jump(keys, key, epsilon=16)
            self.assertTrue(result["localized"], msg=result)
            hits += int(result["exact"])
        self.assertEqual(hits, trials)

    def test_residual_localizes_every_convex_gap(self) -> None:
        keys = convex_keys(2_000)
        rng = random.Random(1)
        for key in rng.sample(keys[2:-2], 15):
            result = residual_jump(keys, key, epsilon=16)
            self.assertTrue(result["localized"], msg=result)

    def test_segment_start_exposure_is_exactly_the_stored_key(self) -> None:
        keys = lognormal_keys(1_000, seed=1)
        segments = build_segments(keys, 16, secret=0, modulus=None)
        self.assertTrue(segment_start_exposed(segments, segments[0].start_key))
        interior = keys[1]
        if all(seg.start_key != interior for seg in segments):
            self.assertFalse(segment_start_exposed(segments, interior))

    def test_snapshot_diff_beats_chance_and_rebuild_is_exact(self) -> None:
        keys = lognormal_keys(1_500, seed=2)
        epsilon, secret, modulus = 16, 3, 128
        index = SHIIndex.build(keys, epsilon=epsilon, secret=secret, modulus=modulus)
        rng = random.Random(1)
        top1 = []
        trials = 20
        for key in rng.sample(keys[10:-10], trials):
            before = list(index.segments)
            fresh = SHIIndex.build(
                [k for k in keys if k != key], epsilon=epsilon, secret=secret, modulus=modulus
            )
            diff = snapshot_diff_result(before, fresh.segments, key, keys, rng, n_candidates=200)
            self.assertTrue(diff["localized"], msg=key)
            top1.append(diff["top1"])
            changed = absolute_suffix_changed(before, fresh.segments)
            self.assertGreater(changed, 0)
        # Chance is 1/200. The diff range should hold the key and rule out most candidates.
        self.assertGreater(sum(top1) / trials, 2 / 200)

        survivors = [k for k in keys if k != keys[100]]
        decoys = keys[200:230]
        self.assertTrue(
            candidate_rebuild_identifies(survivors, keys[100], decoys, epsilon, secret, modulus)
        )

    def test_dp_noise_hides_the_bin_that_exact_release_reveals(self) -> None:
        keys = lognormal_keys(4_000, seed=3)
        lo, hi = keys[0], keys[-1]
        n_bins = 32
        rng = random.Random(0)
        np_rng = np.random.default_rng(0)
        quiet_hits = 0
        noisy_hits = 0
        trials = 40
        for key in rng.sample(range(1, len(keys) - 1), trials):
            deleted = keys[key]
            before = histogram(keys, n_bins, lo, hi)
            after_keys = keys[:key] + keys[key + 1 :]
            after = histogram(after_keys, n_bins, lo, hi)
            self.assertEqual(diff_bin(before, after), bin_of(deleted, n_bins, lo, hi))
            quiet = release_counts(before, epsilon=100, rng=np_rng)
            quiet_after = release_counts(after, epsilon=100, rng=np_rng)
            noisy = release_counts(before, epsilon=0.05, rng=np_rng)
            noisy_after = release_counts(after, epsilon=0.05, rng=np_rng)
            truth = bin_of(deleted, n_bins, lo, hi)
            quiet_hits += int(diff_bin(quiet, quiet_after) == truth)
            noisy_hits += int(diff_bin(noisy, noisy_after) == truth)
        self.assertGreater(quiet_hits / trials, 0.7)
        self.assertLess(noisy_hits / trials, 0.3)
        self.assertGreater(attack_advantage(quiet_hits, trials, n_bins), 10)
        self.assertLess(attack_advantage(noisy_hits, trials, n_bins), 10)


if __name__ == "__main__":
    unittest.main()
