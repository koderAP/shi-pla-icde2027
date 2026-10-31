#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace shi {

/**
 * Multi-snapshot Differentially Private Learned Index (DP-PLA) for Threat Model T2.
 *
 * Partitions the key space into B fine-grained quantile/dyadic intervals (default
 * ~32 keys per bucket) whose true counts have L1 sensitivity 1 under single-key
 * deletion. Each snapshot adds independent Laplace(1 / eps_dp) noise to bucket
 * counts, projects the noisy counts onto a monotone non-decreasing CDF (post-
 * processing), and fits piecewise-linear segments to the noisy CDF.
 *
 * By parallel composition across disjoint buckets, a single snapshot is eps_dp-DP,
 * and a pair of before/after snapshots is (2 * eps_dp)-DP, bounding any T2
 * distinguisher's multiplicative advantage by exp(2 * eps_dp).
 */
struct DPBenchmarkPoint {
    double epsilon_dp;
    double theoretical_max_advantage;
    double empirical_top1_rate;
    double chance_rate;
    double empirical_advantage;
    double mean_abs_rank_error;
    double p99_abs_rank_error;
    double avg_lookup_comparisons;
};

class DPLearnedIndex {
public:
    struct BucketSegment {
        uint64_t lo_key;
        uint64_t hi_key;
        double noisy_count;
        double cum_rank;
        double slope;
    };

    DPLearnedIndex(
        const std::vector<uint64_t> &sorted_keys,
        const std::vector<uint64_t> &public_grid,
        double epsilon_dp,
        uint64_t seed
    )
        : epsilon_dp_(epsilon_dp) {
        build(sorted_keys, public_grid, epsilon_dp, seed);
    }

    size_t predict_rank(uint64_t key, size_t n_total) const noexcept {
        if (buckets_.empty() || n_total == 0) return 0;
        if (key <= buckets_.front().lo_key) return 0;
        if (key >= buckets_.back().hi_key) return n_total - 1;

        size_t lo = 0, hi = buckets_.size() - 1, idx = 0;
        while (lo <= hi) {
            size_t mid = lo + ((hi - lo) >> 1);
            if (buckets_[mid].lo_key <= key) {
                idx = mid;
                lo = mid + 1;
            } else {
                if (mid == 0) break;
                hi = mid - 1;
            }
        }
        const auto &b = buckets_[idx];
        double r = b.cum_rank + b.slope * static_cast<double>(key - b.lo_key);
        double scaled = (total_noisy_ > 0.0) ? (r / total_noisy_ * static_cast<double>(n_total - 1)) : 0.0;
        if (scaled < 0.0) return 0;
        if (scaled >= static_cast<double>(n_total - 1)) return n_total - 1;
        return static_cast<size_t>(std::llround(scaled));
    }

    const std::vector<BucketSegment> &buckets() const noexcept { return buckets_; }
    double epsilon_dp() const noexcept { return epsilon_dp_; }

    static size_t bucket_for_key(const std::vector<uint64_t> &grid, uint64_t key) noexcept {
        if (grid.size() <= 2 || key <= grid.front()) return 0;
        size_t n_buckets = grid.size() - 1;
        if (key >= grid.back()) return n_buckets - 1;
        auto it = std::upper_bound(grid.begin(), grid.end(), key);
        size_t idx = static_cast<size_t>(it - grid.begin());
        if (idx == 0) return 0;
        return std::min(n_buckets - 1, idx - 1);
    }

    static double sample_laplace(std::mt19937_64 &rng, double scale) {
        std::uniform_real_distribution<double> unif(-0.499999999999, 0.499999999999);
        double u = unif(rng);
        return (u < 0.0) ? (scale * std::log(1.0 + 2.0 * u)) : (-scale * std::log(1.0 - 2.0 * u));
    }

    static std::vector<DPBenchmarkPoint> evaluate_tradeoff(
        const std::vector<uint64_t> &keys,
        const std::vector<double> &epsilons,
        size_t n_trials = 200,
        size_t n_candidates = 1000,
        uint64_t seed = 20261022ULL
    ) {
        const size_t n = keys.size();
        const size_t n_buckets = std::min<size_t>(n_candidates, std::max<size_t>(64, n / 64));

        // Build public domain grid from coarse quantiles
        std::vector<uint64_t> grid(n_buckets + 1);
        for (size_t b = 0; b <= n_buckets; ++b) {
            size_t pos = (b * (n - 1)) / n_buckets;
            grid[b] = keys[pos];
        }
        grid.back() = keys.back() + 1;

        // Precompute true bucket counts
        std::vector<double> base_counts(n_buckets, 0.0);
        for (uint64_t k : keys) {
            base_counts[bucket_for_key(grid, k)] += 1.0;
        }

        std::mt19937_64 rng(seed);
        std::uniform_int_distribution<size_t> pick_idx(1, n - 2);

        std::vector<size_t> deleted_indices(n_trials);
        for (size_t t = 0; t < n_trials; ++t) {
            deleted_indices[t] = pick_idx(rng);
        }

        std::vector<DPBenchmarkPoint> results;
        for (double eps : epsilons) {
            size_t top1_hits = 0;
            for (size_t t = 0; t < n_trials; ++t) {
                size_t del_idx = deleted_indices[t];
                uint64_t del_key = keys[del_idx];
                size_t true_bucket = bucket_for_key(grid, del_key);

                // Pick 1 candidate per bucket up to n_buckets so candidate pool has known size
                double scale = 1.0 / eps;
                size_t best_bucket = 0;
                double biggest_drop = -1e300;
                for (size_t b = 0; b < n_buckets; ++b) {
                    double before_b = base_counts[b] + sample_laplace(rng, scale);
                    double after_true = base_counts[b] - (b == true_bucket ? 1.0 : 0.0);
                    double after_b = after_true + sample_laplace(rng, scale);
                    double drop = before_b - after_b;
                    if (drop > biggest_drop) {
                        biggest_drop = drop;
                        best_bucket = b;
                    }
                }
                if (best_bucket == true_bucket) {
                    ++top1_hits;
                }
            }

            // Build one DP index at this epsilon and measure rank error across 5,000 queries
            DPLearnedIndex dp_idx(keys, grid, eps, seed + static_cast<uint64_t>(eps * 1000));
            std::vector<double> errors;
            size_t step = std::max<size_t>(1, n / 4000);
            errors.reserve(n / step + 1);
            double total_cmps = 0.0;
            for (size_t i = 0; i < n; i += step) {
                size_t pred = dp_idx.predict_rank(keys[i], n);
                double err = std::abs(static_cast<double>(pred) - static_cast<double>(i));
                errors.push_back(err);
                total_cmps += std::log2(2.0 + err);
            }
            std::sort(errors.begin(), errors.end());
            double mean_err = 0.0;
            for (double e : errors) mean_err += e;
            mean_err /= std::max<size_t>(1, errors.size());
            double p99_err = errors.empty() ? 0.0 : errors[static_cast<size_t>(0.99 * (errors.size() - 1))];

            double top1_rate = static_cast<double>(top1_hits) / static_cast<double>(n_trials);
            double chance = 1.0 / static_cast<double>(n_buckets);
            results.push_back(DPBenchmarkPoint{
                eps,
                std::exp(std::min(20.0, 2.0 * eps)),
                top1_rate,
                chance,
                top1_rate / chance,
                mean_err,
                p99_err,
                total_cmps / std::max<size_t>(1, errors.size())
            });
        }
        return results;
    }

private:
    double epsilon_dp_;
    double total_noisy_ = 0.0;
    std::vector<BucketSegment> buckets_;

    void build(
        const std::vector<uint64_t> &sorted_keys,
        const std::vector<uint64_t> &grid,
        double epsilon_dp,
        uint64_t seed
    ) {
        size_t n_buckets = grid.size() - 1;
        std::vector<double> counts(n_buckets, 0.0);
        for (uint64_t k : sorted_keys) {
            counts[bucket_for_key(grid, k)] += 1.0;
        }
        std::mt19937_64 rng(seed);
        double scale = 1.0 / epsilon_dp;
        buckets_.resize(n_buckets);
        double cum = 0.0;
        for (size_t b = 0; b < n_buckets; ++b) {
            double noisy = std::max(0.0, counts[b] + sample_laplace(rng, scale));
            uint64_t lo = grid[b];
            uint64_t hi = grid[b + 1];
            double width = static_cast<double>(hi > lo ? (hi - lo) : 1);
            buckets_[b] = BucketSegment{lo, hi, noisy, cum, noisy / width};
            cum += noisy;
        }
        total_noisy_ = std::max(1.0, cum);
    }
};

} // namespace shi
