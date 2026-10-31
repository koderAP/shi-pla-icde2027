#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "pgm/piecewise_linear_model.hpp"

namespace shi {

enum class PLAMode : uint8_t {
    GreedyCone = 0,
    OptimalHull = 1,
};

inline uint64_t splitmix64(uint64_t key, uint64_t secret) noexcept {
    uint64_t z = key + secret;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

inline bool is_anchor(uint64_t key, uint64_t secret, uint32_t modulus) noexcept {
    if (modulus == 0) {
        return false;
    }
    if ((modulus & (modulus - 1)) == 0) {
        return (splitmix64(key, secret) & (modulus - 1)) == 0;
    }
    return (splitmix64(key, secret) % modulus) == 0;
}

inline constexpr uint32_t kMaxSegmentKeys = 65535;

// Exact value of a float slope times an integer gap. A double cannot hold a
// uint64 gap, and clamping that product to int32 sent lookups into a scan.
inline uint64_t slope_times_gap(float slope, uint64_t gap) noexcept {
    if (gap == 0 || slope == 0.0f) return 0;
    // A gap that fits in the double mantissa has a product error below one
    // rank when the product itself fits in that mantissa, which is the local
    // offset a segment actually stores. Wider gaps use the integer product.
    if (gap <= (1ULL << 53) && slope > 0.0f) {
        double raw = static_cast<double>(slope) * static_cast<double>(gap);
        if (raw > 0.0 && raw < static_cast<double>(1ULL << 53)) {
            return static_cast<uint64_t>(raw + 0.5);
        }
    }
    uint32_t bits = 0;
    std::memcpy(&bits, &slope, sizeof(bits));
    auto from_double = [&]() -> uint64_t {
        double raw = static_cast<double>(slope) * static_cast<double>(gap);
        if (!(raw > 0.0)) return 0;
        if (raw >= static_cast<double>(std::numeric_limits<uint64_t>::max())) {
            return std::numeric_limits<uint64_t>::max();
        }
        return static_cast<uint64_t>(raw);
    };
    if ((bits >> 31) != 0) return from_double();
    int exp = static_cast<int>((bits >> 23) & 255);
    if (exp == 0 || exp == 255) return from_double();
    uint32_t sig = (bits & 0x7fffffu) | 0x800000u;
    unsigned __int128 prod = static_cast<unsigned __int128>(sig) * static_cast<unsigned __int128>(gap);
    int shift = exp - 127 - 23;
    if (shift >= 0) {
        if (shift >= 40) return std::numeric_limits<uint64_t>::max();
        unsigned __int128 shifted = prod << shift;
        if (shifted > static_cast<unsigned __int128>(std::numeric_limits<uint64_t>::max())) {
            return std::numeric_limits<uint64_t>::max();
        }
        return static_cast<uint64_t>(shifted);
    }
    int r = -shift;
    if (r >= 128) return 0;
    unsigned __int128 q = prod >> r;
    unsigned __int128 rem = prod & ((static_cast<unsigned __int128>(1) << r) - 1);
    unsigned __int128 half = static_cast<unsigned __int128>(1) << (r - 1);
    if (rem > half || (rem == half && (q & 1))) ++q;
    if (q > static_cast<unsigned __int128>(std::numeric_limits<uint64_t>::max())) {
        return std::numeric_limits<uint64_t>::max();
    }
    return static_cast<uint64_t>(q);
}

// Key storage cut on the same anchors as the model. A piece is the keys from
// one anchor up to, but not including, the next. After a delete the pieces
// are the pieces Build would store, and only the piece that held the key moves.
class KeyStore {
public:
    KeyStore() = default;
    KeyStore(const KeyStore &other) { copy_from(other); }
    KeyStore &operator=(const KeyStore &other) {
        if (this != &other) copy_from(other);
        return *this;
    }
    KeyStore(KeyStore &&) noexcept = default;
    KeyStore &operator=(KeyStore &&) noexcept = default;

    void assign_sorted(const uint64_t *keys, size_t n, uint64_t secret, uint32_t modulus) {
        pieces_.clear();
        arena_.clear();
        n_ = n;
        if (n == 0) {
            prefix_.assign(1, 0);
            return;
        }
        // Spare capacity stays on a copy. Without it, the first slot allocation
        // after a copy reallocates the whole arena inside the delete timer.
        arena_.reserve(n * 4 + 65536);
        arena_.assign(keys, keys + n);
        uint32_t begin = 0;
        for (size_t i = 0; i < n; ++i) {
            if (i > 0 && is_anchor(keys[i], secret, modulus)) {
                pieces_.push_back(Slot{begin, static_cast<uint32_t>(i) - begin, static_cast<uint32_t>(i) - begin});
                begin = static_cast<uint32_t>(i);
            }
        }
        pieces_.push_back(Slot{begin, static_cast<uint32_t>(n) - begin, static_cast<uint32_t>(n) - begin});
        packed_ = true;
        rebuild_prefix();
    }

    size_t size() const noexcept { return n_; }
    bool empty() const noexcept { return n_ == 0; }
    uint64_t front() const { return arena_[pieces_.front().begin]; }

    uint64_t operator[](size_t rank) const {
        const uint64_t *p = try_span(rank, 1);
        if (p != nullptr) return *p;
        size_t piece = 0;
        size_t off = 0;
        locate(rank, piece, off);
        return arena_[pieces_[piece].begin + off];
    }

    // Live keys starting at rank, when they sit in one contiguous arena run.
    // Null when the window crosses a piece boundary.
    const uint64_t *try_span(size_t rank, size_t len) const {
        if (len == 0 || rank >= n_ || rank + len > n_) return nullptr;
        if (packed_) return arena_.data() + rank;
        const uint64_t *direct = span_from_gaps(rank, len);
        if (direct != nullptr) return direct;
        size_t piece = 0;
        size_t off = 0;
        locate(rank, piece, off);
        if (off + len > pieces_[piece].count) return nullptr;
        return arena_.data() + pieces_[piece].begin + off;
    }

    size_t lower_bound_rank(uint64_t key) const {
        if (n_ == 0 || key <= front()) return 0;
        const Slot &last = pieces_.back();
        if (key > arena_[last.begin + last.count - 1]) return n_;
        size_t lo = 0;
        size_t hi = pieces_.size() - 1;
        while (lo < hi) {
            size_t mid = lo + (hi - lo + 1) / 2;
            if (arena_[pieces_[mid].begin] <= key) lo = mid;
            else hi = mid - 1;
        }
        const uint64_t *b = arena_.data() + pieces_[lo].begin;
        const uint64_t *e = b + pieces_[lo].count;
        return prefix_[lo] + static_cast<size_t>(std::lower_bound(b, e, key) - b);
    }

    void erase_at(size_t rank, uint64_t secret, uint32_t modulus) {
        size_t piece = 0;
        size_t off = 0;
        locate(rank, piece, off);
        Slot &slot = pieces_[piece];
        bool was_start = (off == 0);
        last_edit_ = EditTrace{};
        last_edit_.piece_keys = slot.count;
        last_edit_.prev_keys = (piece > 0) ? pieces_[piece - 1].count : 0;
        auto t_move = std::chrono::steady_clock::now();
        if (off + 1 < slot.count) {
            std::memmove(
                arena_.data() + slot.begin + off,
                arena_.data() + slot.begin + off + 1,
                (slot.count - off - 1) * sizeof(uint64_t)
            );
        }
        last_edit_.was_piece_tail = (off + 1 == slot.count);
        if (zero_tail_) {
            arena_[slot.begin + slot.count - 1] = 0;
        }
        last_edit_.vacated_after = arena_[slot.begin + slot.count - 1];
        --slot.count;
        --n_;
        packed_ = false;
        auto t_merge = std::chrono::steady_clock::now();
        if (slot.count == 0) {
            pieces_.erase(pieces_.begin() + static_cast<std::ptrdiff_t>(piece));
        } else if (was_start && piece > 0 && !is_anchor(arena_[slot.begin], secret, modulus)) {
            Slot cur = pieces_[piece];
            Slot prev = pieces_[piece - 1];
            uint32_t need = prev.count + cur.count;
            last_edit_.merged = true;
            last_edit_.copied_keys = need;
            if (prev.cap < need) {
                uint32_t fresh = alloc_slot(need);
                std::memcpy(arena_.data() + fresh, arena_.data() + prev.begin, prev.count * sizeof(uint64_t));
                std::memcpy(arena_.data() + fresh + prev.count, arena_.data() + cur.begin, cur.count * sizeof(uint64_t));
                pieces_[piece - 1].begin = fresh;
                pieces_[piece - 1].cap = need;
            } else {
                std::memcpy(
                    arena_.data() + prev.begin + prev.count,
                    arena_.data() + cur.begin,
                    cur.count * sizeof(uint64_t)
                );
                last_edit_.copied_keys = cur.count;
            }
            pieces_[piece - 1].count = need;
            pieces_.erase(pieces_.begin() + static_cast<std::ptrdiff_t>(piece));
        }
        auto t_prefix = std::chrono::steady_clock::now();
        rebuild_prefix();
        auto t_done = std::chrono::steady_clock::now();
        last_edit_.move_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t_merge - t_move).count());
        last_edit_.merge_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t_prefix - t_merge).count());
        last_edit_.prefix_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t_done - t_prefix).count());
    }

    void insert_key(uint64_t key, uint64_t secret, uint32_t modulus) {
        if (n_ == 0) {
            pieces_.push_back(Slot{alloc_slot(1), 1, 1});
            arena_[pieces_.back().begin] = key;
            n_ = 1;
            rebuild_prefix();
            return;
        }
        if (key < front()) {
            if (is_anchor(front(), secret, modulus)) {
                Slot neu{alloc_slot(1), 1, 1};
                arena_[neu.begin] = key;
                pieces_.insert(pieces_.begin(), neu);
            } else {
                grow_piece(0, pieces_.front().count + 1);
                std::memmove(
                    arena_.data() + pieces_.front().begin + 1,
                    arena_.data() + pieces_.front().begin,
                    pieces_.front().count * sizeof(uint64_t)
                );
                arena_[pieces_.front().begin] = key;
                ++pieces_.front().count;
            }
        ++n_;
        packed_ = false;
        rebuild_prefix();
        return;
    }
        size_t lo = 0;
        size_t hi = pieces_.size() - 1;
        while (lo < hi) {
            size_t mid = lo + (hi - lo + 1) / 2;
            if (arena_[pieces_[mid].begin] <= key) lo = mid;
            else hi = mid - 1;
        }
        Slot slot = pieces_[lo];
        const uint64_t *b = arena_.data() + slot.begin;
        const uint64_t *e = b + slot.count;
        const uint64_t *it = std::lower_bound(b, e, key);
        if (it != e && *it == key) {
            throw std::invalid_argument("duplicate key in KeyStore::insert_key");
        }
        uint32_t off = static_cast<uint32_t>(it - b);
        if (is_anchor(key, secret, modulus)) {
            uint32_t right_n = slot.count - off;
            uint32_t right_cap = right_n + 1;
            uint32_t right_at = alloc_slot(right_cap);
            arena_[right_at] = key;
            if (right_n > 0) {
                std::memcpy(arena_.data() + right_at + 1, arena_.data() + slot.begin + off, right_n * sizeof(uint64_t));
            }
            pieces_[lo].count = off;
            Slot right{right_at, right_cap, right_cap};
            if (off == 0) {
                pieces_.erase(pieces_.begin() + static_cast<std::ptrdiff_t>(lo));
                pieces_.insert(pieces_.begin() + static_cast<std::ptrdiff_t>(lo), right);
            } else {
                pieces_.insert(pieces_.begin() + static_cast<std::ptrdiff_t>(lo + 1), right);
            }
        } else {
            grow_piece(lo, pieces_[lo].count + 1);
            uint64_t *base = arena_.data() + pieces_[lo].begin;
            std::memmove(base + off + 1, base + off, (pieces_[lo].count - off) * sizeof(uint64_t));
            base[off] = key;
            ++pieces_[lo].count;
        }
        ++n_;
        packed_ = false;
        rebuild_prefix();
    }

    // 0 leaves the boundary list to grow. A positive limit rewrites the arena
    // into rank order once the list is longer than that, which restores the
    // packed lookup. The serialized model does not include the arena.
    void set_compact_limit(size_t limit) noexcept { compact_limit_ = limit; }
    struct EditTrace {
        size_t piece_keys = 0;
        size_t prev_keys = 0;
        size_t copied_keys = 0;
        bool merged = false;
        bool was_piece_tail = false;
        uint64_t vacated_after = 0;
        uint64_t move_ns = 0;
        uint64_t merge_ns = 0;
        uint64_t prefix_ns = 0;
    };

    size_t piece_len(size_t rank) const {
        size_t piece = 0;
        size_t off = 0;
        locate(rank, piece, off);
        return pieces_[piece].count;
    }
    bool is_piece_tail(size_t rank) const {
        size_t piece = 0;
        size_t off = 0;
        locate(rank, piece, off);
        return off + 1 == pieces_[piece].count;
    }
    void set_zero_tail(bool on) noexcept { zero_tail_ = on; }
    size_t max_piece_keys() const {
        size_t m = 0;
        for (const Slot &slot : pieces_) m = std::max(m, static_cast<size_t>(slot.count));
        return m;
    }
    const EditTrace &last_edit() const noexcept { return last_edit_; }
    size_t boundary_count() const noexcept { return gaps_.size(); }
    size_t compaction_count() const noexcept { return compactions_; }
    uint64_t compact_ns() const noexcept { return compact_ns_; }
    uint64_t compact_max_ns() const noexcept { return compact_max_ns_; }

    size_t copy_from(size_t rank, uint64_t *dst, size_t maxn) const {
        if (rank >= n_ || maxn == 0) return 0;
        size_t piece = 0;
        size_t off = 0;
        locate(rank, piece, off);
        size_t wrote = 0;
        while (wrote < maxn && piece < pieces_.size()) {
            const Slot &src = pieces_[piece];
            while (off < src.count && wrote < maxn) {
                dst[wrote++] = arena_[src.begin + off];
                ++off;
            }
            ++piece;
            off = 0;
        }
        return wrote;
    }

private:
    struct Slot {
        uint32_t begin = 0;
        uint32_t count = 0;
        uint32_t cap = 0;
        // Spare keys that sit immediately before this piece and that a slide must leave in place.
        uint32_t hole_keep = 0;
    };

    static constexpr size_t kGrowGroup = 1u << 23;
    static constexpr uint32_t kGroupSpare = 4096;

    std::vector<uint64_t> arena_;
    std::vector<Slot> pieces_;
    std::vector<size_t> prefix_;
    struct Gap {
        size_t rank;
        int64_t shift;
    };
    std::vector<Gap> gaps_;
    size_t n_ = 0;
    size_t compact_limit_ = 0;
    size_t gap_floor_ = 0;
    size_t compactions_ = 0;
    uint64_t compact_ns_ = 0;
    uint64_t compact_max_ns_ = 0;
    EditTrace last_edit_{};
    bool packed_ = false;
    bool zero_tail_ = true;

    // Live keys of a piece occupy arena[rank + shift]. Shift is signed:
    // a delete leaves a dead slot and later ranks move forward, an insert
    // before a piece moves those ranks backward. Shift changes only at a
    // piece boundary where the piece does not follow the previous one.
    const uint64_t *span_from_gaps(size_t rank, size_t len) const {
        if (gaps_.empty()) return arena_.data() + rank;
        size_t lo = 0;
        size_t hi = gaps_.size();
        while (lo < hi) {
            size_t mid = lo + (hi - lo) / 2;
            if (gaps_[mid].rank <= rank) lo = mid + 1;
            else hi = mid;
        }
        int64_t shift = (lo == 0) ? 0 : gaps_[lo - 1].shift;
        size_t end = rank + len - 1;
        if (lo < gaps_.size() && gaps_[lo].rank <= end) return nullptr;
        return arena_.data() + static_cast<std::ptrdiff_t>(rank) + static_cast<std::ptrdiff_t>(shift);
    }

    void copy_from(const KeyStore &other) {
        pieces_ = other.pieces_;
        prefix_ = other.prefix_;
        gaps_ = other.gaps_;
        n_ = other.n_;
        compact_limit_ = other.compact_limit_;
        compactions_ = other.compactions_;
        compact_ns_ = other.compact_ns_;
        compact_max_ns_ = other.compact_max_ns_;
        last_edit_ = other.last_edit_;
        packed_ = other.packed_;
        zero_tail_ = other.zero_tail_;
        gap_floor_ = other.gap_floor_;
        size_t want = std::max(other.arena_.capacity(), other.n_ * 4 + 65536);
        arena_.clear();
        arena_.reserve(want);
        arena_.assign(other.arena_.begin(), other.arena_.end());
    }

    uint32_t alloc_slot(uint32_t cap) {
        uint32_t begin = static_cast<uint32_t>(arena_.size());
        arena_.resize(arena_.size() + cap);
        return begin;
    }

    // Grow piece idx in place by shifting the following live keys into a nearby hole.
    // A tail allocation is the fallback, and it is what leaves a boundary the slide cannot close.
    void grow_piece(size_t idx, uint32_t need) {
        if (pieces_[idx].cap >= need) return;
        uint32_t extra = need - pieces_[idx].cap;
        if (shift_piece(idx, extra)) return;
        plant_spares();
        if (shift_piece(idx, extra)) return;
        uint32_t cap = std::max(need, pieces_[idx].cap * 2);
        uint32_t fresh = alloc_slot(cap);
        if (pieces_[idx].count > 0) {
            std::memcpy(
                arena_.data() + fresh,
                arena_.data() + pieces_[idx].begin,
                pieces_[idx].count * sizeof(uint64_t)
            );
        }
        pieces_[idx].begin = fresh;
        pieces_[idx].cap = cap;
        pieces_[idx].hole_keep = 0;
    }

    bool shift_piece(size_t idx, uint32_t extra) {
        size_t last = idx;
        size_t keys = pieces_[idx].count;
        if (keys > kGrowGroup) return false;
        for (size_t j = idx + 1; j < pieces_.size(); ++j) {
            uint32_t prev_end = pieces_[j - 1].begin + pieces_[j - 1].count;
            if (pieces_[j].begin != prev_end) break;
            if (keys + pieces_[j].count > kGrowGroup) break;
            keys += pieces_[j].count;
            last = j;
        }
        uint32_t run_begin = pieces_[idx].begin;
        uint32_t live_end = pieces_[last].begin + pieces_[last].count;
        uint32_t next_phys = std::numeric_limits<uint32_t>::max();
        for (size_t p = 0; p < pieces_.size(); ++p) {
            if (p >= idx && p <= last) continue;
            const Slot &other = pieces_[p];
            if (other.count == 0) continue;
            uint32_t other_end = other.begin + other.count;
            if (other.begin < live_end && other_end > run_begin) return false;
            if (other.begin >= live_end && other.begin < next_phys) next_phys = other.begin;
        }
        uint64_t limit = (next_phys == std::numeric_limits<uint32_t>::max())
                             ? arena_.capacity()
                             : next_phys;
        if (limit < static_cast<uint64_t>(live_end) + extra) return false;
        if (next_phys == std::numeric_limits<uint32_t>::max() &&
            arena_.size() < static_cast<size_t>(live_end) + extra) {
            arena_.resize(static_cast<size_t>(live_end) + extra);
        }
        uint32_t src = pieces_[idx].begin + pieces_[idx].count;
        uint32_t len = live_end - src;
        if (len > 0) {
            std::memmove(
                arena_.data() + src + extra,
                arena_.data() + src,
                static_cast<size_t>(len) * sizeof(uint64_t)
            );
        }
        for (size_t p = idx + 1; p <= last; ++p) {
            pieces_[p].begin += extra;
            pieces_[p].cap = pieces_[p].count;
        }
        pieces_[idx].cap = pieces_[idx].count + extra;
        if (next_phys != std::numeric_limits<uint32_t>::max()) {
            uint32_t left = next_phys - (live_end + extra);
            for (Slot &slot : pieces_) {
                if (slot.begin == next_phys && slot.hole_keep > left) slot.hole_keep = left;
            }
        }
        return true;
    }

    // One rank-ordered copy with a spare hole before every group, so a later insert
    // shifts at most one group instead of allocating a piece at the tail.
    void plant_spares() {
        size_t groups = n_ / kGrowGroup + 1;
        size_t want = std::max(n_ * 4 + 65536, n_ + groups * kGroupSpare + 65536);
        std::vector<uint64_t> fresh;
        fresh.reserve(want);
        std::vector<Slot> neu;
        neu.reserve(pieces_.size());
        size_t live = 0;
        size_t next_cut = kGrowGroup;
        for (size_t i = 0; i < pieces_.size(); ++i) {
            const Slot &src = pieces_[i];
            uint32_t keep = 0;
            if (live >= next_cut && !neu.empty()) {
                fresh.resize(fresh.size() + kGroupSpare);
                keep = kGroupSpare;
                next_cut += kGrowGroup;
            }
            uint32_t begin = static_cast<uint32_t>(fresh.size());
            if (src.count > 0) {
                fresh.resize(fresh.size() + src.count);
                std::memcpy(
                    fresh.data() + begin,
                    arena_.data() + src.begin,
                    src.count * sizeof(uint64_t)
                );
            }
            neu.push_back(Slot{begin, src.count, src.count, keep});
            live += src.count;
        }
        arena_.swap(fresh);
        pieces_.swap(neu);
        gap_floor_ = 0;
        for (const Slot &slot : pieces_) {
            if (slot.hole_keep > 0) ++gap_floor_;
        }
    }

    void rebuild_prefix() {
        prefix_.resize(pieces_.size() + 1);
        size_t acc = 0;
        for (size_t i = 0; i < pieces_.size(); ++i) {
            prefix_[i] = acc;
            acc += pieces_[i].count;
        }
        prefix_.back() = acc;
        n_ = acc;
        rebuild_gaps();
        if (compact_limit_ > 0 && gaps_.size() > compact_limit_ + gap_floor_) compact_excess();
    }

    // Pieces that share a shift already sit in one contiguous run.
    // The boundary is a hole in front of that run. Sliding the smallest run
    // left into that hole removes the boundary and does not allocate.
    void compact_excess() {
        size_t guard = 0;
        while (gaps_.size() > compact_limit_ && guard < 4) {
            size_t before = gaps_.size();
            size_t best_first = 0;
            size_t best_last = 0;
            uint32_t best_slide = 0;
            uint64_t best_run = std::numeric_limits<uint64_t>::max();
            struct Iv { uint32_t b; uint32_t e; };
            std::vector<Iv> iv;
            iv.reserve(pieces_.size());
            for (const Slot &slot : pieces_) {
                iv.push_back(Iv{slot.begin, slot.begin + slot.count});
            }
            std::sort(iv.begin(), iv.end(), [](const Iv &a, const Iv &b) { return a.b < b.b; });
            std::vector<uint32_t> max_end(iv.size());
            for (size_t i = 0; i < iv.size(); ++i) {
                max_end[i] = (i == 0) ? iv[i].e : std::max(max_end[i - 1], iv[i].e);
            }
            auto blocked_span = [&](uint32_t dest, uint32_t src) {
                auto it = std::lower_bound(iv.begin(), iv.end(), src, [](const Iv &a, uint32_t s) { return a.b < s; });
                if (it == iv.begin()) return false;
                return max_end[static_cast<size_t>(it - iv.begin() - 1)] > dest;
            };
            for (size_t g = 0; g < gaps_.size(); ++g) {
                size_t first = 0;
                size_t off = 0;
                locate(gaps_[g].rank, first, off);
                if (first == 0) continue;
                size_t last = pieces_.size();
                if (g + 1 < gaps_.size()) {
                    size_t next = 0;
                    locate(gaps_[g + 1].rank, next, off);
                    last = next;
                }
                bool contig = true;
                uint64_t run = 0;
                for (size_t p = first; p < last; ++p) {
                    if (p > first &&
                        pieces_[p].begin != pieces_[p - 1].begin + pieces_[p - 1].count) {
                        contig = false;
                    }
                    run += pieces_[p].count;
                }
                uint32_t src = pieces_[first].begin;
                uint32_t dest = pieces_[first - 1].begin + pieces_[first - 1].count;
                if (!contig || run == 0 || dest >= src || run >= best_run) continue;
                uint32_t keep = pieces_[first].hole_keep;
                uint32_t hole = src - dest;
                if (hole <= keep) continue;
                if (blocked_span(dest, src)) continue;
                best_run = run;
                best_first = first;
                best_last = last;
                best_slide = hole - keep;
            }
            if (best_run == std::numeric_limits<uint64_t>::max()) break;
            uint32_t src = pieces_[best_first].begin;
            uint32_t dest = pieces_[best_first - 1].begin + pieces_[best_first - 1].count;
            uint32_t keep = pieces_[best_first].hole_keep;
            auto t0 = std::chrono::steady_clock::now();
            std::memmove(
                arena_.data() + dest + keep,
                arena_.data() + src,
                static_cast<size_t>(best_run) * sizeof(uint64_t)
            );
            pieces_[best_first - 1].cap = pieces_[best_first - 1].count;
            for (size_t p = best_first; p < best_last; ++p) pieces_[p].begin -= best_slide;
            auto t1 = std::chrono::steady_clock::now();
            uint64_t spent = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
            compact_ns_ += spent;
            if (spent > compact_max_ns_) compact_max_ns_ = spent;
            ++compactions_;
            rebuild_gaps();
            if (gaps_.size() >= before) break;
            ++guard;
        }
    }

    void compact_arena() {
        auto t0 = std::chrono::steady_clock::now();
        std::vector<uint64_t> fresh;
        fresh.reserve(n_ + n_ / 8 + 4096);
        fresh.resize(n_);
        size_t at = 0;
        for (const Slot &src : pieces_) {
            if (src.count == 0) continue;
            std::memcpy(fresh.data() + at, arena_.data() + src.begin, src.count * sizeof(uint64_t));
            at += src.count;
        }
        arena_.swap(fresh);
        uint32_t begin = 0;
        for (Slot &slot : pieces_) {
            slot.begin = begin;
            slot.cap = slot.count;
            slot.hole_keep = 0;
            begin += slot.count;
        }
        packed_ = true;
        rebuild_gaps();
        auto t1 = std::chrono::steady_clock::now();
        compact_ns_ += static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
        ++compactions_;
    }

    void rebuild_gaps() {
        gap_floor_ = 0;
        for (const Slot &slot : pieces_) {
            if (slot.hole_keep > 0) ++gap_floor_;
        }
        gaps_.clear();
        if (n_ == 0 || pieces_.empty()) return;
        int64_t prev = 0;
        gaps_.reserve(64);
        for (size_t i = 0; i < pieces_.size(); ++i) {
            int64_t shift = static_cast<int64_t>(pieces_[i].begin) - static_cast<int64_t>(prefix_[i]);
            if (shift != prev) {
                gaps_.push_back(Gap{prefix_[i], shift});
                prev = shift;
            }
        }
    }

    void locate(size_t rank, size_t &piece, size_t &off) const {
        size_t lo = 0;
        size_t hi = pieces_.empty() ? 0 : pieces_.size() - 1;
        while (lo < hi) {
            size_t mid = lo + (hi - lo + 1) / 2;
            if (prefix_[mid] <= rank) lo = mid;
            else hi = mid - 1;
        }
        piece = lo;
        off = rank - prefix_[lo];
    }
};

// A segment is emitted only when its intercept fits in int16.
// The fitter ends the segment instead of clamping, so the stored model
// keeps the piecewise-linear error bound.
#pragma pack(push, 1)
struct CanonicalSegment {
    uint64_t start_key;
    float slope;
    int16_t rel_intercept;
    uint16_t key_count;

    CanonicalSegment() noexcept
        : start_key(0), slope(0.0f), rel_intercept(0), key_count(0) {}

    CanonicalSegment(uint64_t k, float s, int32_t rel_int, uint32_t cnt) noexcept
        : start_key(k),
          slope(s),
          rel_intercept(static_cast<int16_t>(rel_int)),
          key_count(static_cast<uint16_t>(cnt)) {}

    uint32_t count() const noexcept {
        return key_count;
    }

    bool operator==(const CanonicalSegment &other) const noexcept {
        return start_key == other.start_key &&
               std::memcmp(&slope, &other.slope, sizeof(float)) == 0 &&
               rel_intercept == other.rel_intercept &&
               key_count == other.key_count;
    }

    bool operator!=(const CanonicalSegment &other) const noexcept {
        return !(*this == other);
    }

    inline int64_t predict_local(uint64_t key) const noexcept {
        int64_t raw = static_cast<int64_t>(static_cast<double>(slope) * static_cast<double>(key - start_key)) +
                      static_cast<int64_t>(rel_intercept);
        uint32_t c = count();
        if (raw < 0) return 0;
        if (c > 0 && raw >= static_cast<int64_t>(c)) return static_cast<int64_t>(c - 1);
        return raw;
    }

    inline size_t predict_fast(uint64_t key) const noexcept {
        if (key < start_key) return 0;
        uint64_t prod = slope_times_gap(slope, key - start_key);
        int64_t raw = static_cast<int64_t>(std::min<uint64_t>(prod, static_cast<uint64_t>(INT64_MAX))) +
                      static_cast<int64_t>(rel_intercept);
        if (raw < 0) return 0;
        uint32_t c = count();
        if (c > 0 && raw >= static_cast<int64_t>(c)) return static_cast<size_t>(c - 1);
        return static_cast<size_t>(raw);
    }
};
#pragma pack(pop)

static_assert(sizeof(CanonicalSegment) == 16, "CanonicalSegment must be 16 packed bytes (same as PGM)");

inline float normalize_zero(float v) noexcept {
    return (v == 0.0f) ? 0.0f : v;
}

inline std::pair<CanonicalSegment, size_t> fit_segment_greedy(
    const uint64_t *keys,
    size_t n,
    size_t start,
    size_t epsilon,
    uint64_t secret,
    uint32_t modulus
) {
    uint64_t x0 = keys[start];
    double low = -std::numeric_limits<double>::infinity();
    double high = std::numeric_limits<double>::infinity();
    size_t last = start;
    double eps = static_cast<double>(epsilon);

    for (size_t j = start + 1; j < n && (j - start) < kMaxSegmentKeys; ++j) {
        uint64_t xj = keys[j];
        if (is_anchor(xj, secret, modulus)) {
            break;
        }
        double dx = static_cast<double>(xj - x0);
        double rel = static_cast<double>(j - start);
        double s_high = (rel + eps) / dx;
        double s_low = (rel - eps) / dx;
        if (s_high < low || s_low > high) {
            break;
        }
        if (s_low > low) low = s_low;
        if (s_high < high) high = s_high;
        last = j;
    }

    float slope = (last == start) ? 0.0f : normalize_zero(static_cast<float>(0.5 * (low + high)));
    uint32_t count = static_cast<uint32_t>(last - start + 1);
    return {CanonicalSegment{x0, slope, 0, count}, last + 1};
}

inline std::pair<CanonicalSegment, size_t> fit_segment_optimal(
    const uint64_t *keys,
    size_t n,
    size_t start,
    size_t epsilon,
    uint64_t secret,
    uint32_t modulus
) {
    using Model = pgm::internal::OptimalPiecewiseLinearModel<uint64_t, size_t>;
    uint64_t x0 = keys[start];

    auto intercept_of = [](Model &model, uint64_t origin, int16_t &out) -> bool {
        auto seg = model.get_segment();
        auto [ld_slope, intercept] = seg.get_floating_point_segment(origin);
        (void)ld_slope;
        long double iv = static_cast<long double>(intercept);
        if (iv < static_cast<long double>(std::numeric_limits<int16_t>::min()) ||
            iv > static_cast<long double>(std::numeric_limits<int16_t>::max())) {
            return false;
        }
        out = static_cast<int16_t>(iv);
        return true;
    };

    auto fit_until = [&](size_t limit) {
        Model model(epsilon);
        model.add_point(x0, 0);
        size_t end = start;
        for (size_t j = start + 1; j < limit; ++j) {
            if (is_anchor(keys[j], secret, modulus)) break;
            if (!model.add_point(keys[j], j - start)) break;
            end = j;
        }
        return std::pair<Model, size_t>(std::move(model), end);
    };

    size_t window_end = std::min(n, start + static_cast<size_t>(kMaxSegmentKeys));
    auto fitted = fit_until(window_end);
    Model *active = &fitted.first;
    size_t last = fitted.second;
    int16_t stored_intercept = 0;
    bool fits = (last == start) || intercept_of(*active, x0, stored_intercept);
    std::unique_ptr<Model> kept;
    if (!fits) {
        // The full piece does not fit in int16. Keep the longest streaming
        // prefix that does, so the stored intercept is never clamped.
        kept = std::make_unique<Model>(epsilon);
        kept->add_point(x0, 0);
        last = start;
        stored_intercept = 0;
        for (size_t j = start + 1; j < window_end; ++j) {
            if (is_anchor(keys[j], secret, modulus)) break;
            auto trial = std::make_unique<Model>(*kept);
            if (!trial->add_point(keys[j], j - start)) break;
            int16_t trial_intercept = 0;
            if (!intercept_of(*trial, x0, trial_intercept)) break;
            kept = std::move(trial);
            stored_intercept = trial_intercept;
            last = j;
        }
        active = kept.get();
    }

    float slope = 0.0f;
    if (last != start) {
        auto cs = active->get_segment();
        auto [ld_slope, intercept] = cs.get_floating_point_segment(x0);
        (void)intercept;
        slope = normalize_zero(static_cast<float>(ld_slope));
    }
    uint32_t count = static_cast<uint32_t>(last - start + 1);
    return {CanonicalSegment{x0, slope, stored_intercept, count}, last + 1};
}

inline std::pair<CanonicalSegment, size_t> fit_one_segment(
    const uint64_t *keys,
    size_t n,
    size_t start,
    size_t epsilon,
    uint64_t secret,
    uint32_t modulus,
    PLAMode mode
) {
    if (mode == PLAMode::OptimalHull) {
        return fit_segment_optimal(keys, n, start, epsilon, secret, modulus);
    }
    return fit_segment_greedy(keys, n, start, epsilon, secret, modulus);
}

inline std::vector<CanonicalSegment> build_segments(
    const uint64_t *keys,
    size_t n,
    size_t epsilon,
    uint64_t secret = 0,
    uint32_t modulus = 0,
    PLAMode mode = PLAMode::OptimalHull
) {
    std::vector<CanonicalSegment> out;
    if (n == 0) return out;
    out.reserve(n / std::max<size_t>(1, epsilon * 2));
    size_t i = 0;
    while (i < n) {
        auto [seg, nxt] = fit_one_segment(keys, n, i, epsilon, secret, modulus, mode);
        out.push_back(seg);
        i = nxt;
    }
    return out;
}

struct RefitStats {
    size_t segments_refit = 0;
    size_t keys_scanned = 0;
    uint64_t fit_ns = 0;
    uint64_t upper_ns = 0;
    uint64_t span_ns = 0;
    size_t span_rewritten = 0;
    size_t restart_segment = 0;
    size_t resync_old_segment = 0;
};

class SHIIndex {
public:
    SHIIndex(
        const std::vector<uint64_t> &sorted_keys,
        size_t epsilon = 16,
        uint64_t secret = 0x5A175EEDULL,
        uint32_t modulus = 4096,
        PLAMode mode = PLAMode::OptimalHull,
        size_t epsilon_recursive = 4,
        bool upper_anchors = true,
        bool drop_half = false
    )
        : epsilon_(epsilon),
          epsilon_recursive_(epsilon_recursive),
          secret_(secret),
          modulus_(modulus),
          mode_(mode),
          upper_anchors_(upper_anchors),
          drop_half_(drop_half) {
        for (size_t i = 1; i < sorted_keys.size(); ++i) {
            if (sorted_keys[i] <= sorted_keys[i - 1]) {
                throw std::invalid_argument("SHIIndex requires strictly increasing keys");
            }
        }
        segments_ = build_segments(sorted_keys.data(), sorted_keys.size(), epsilon_, secret_, modulus_, mode_);
        keys_.assign_sorted(sorted_keys.data(), sorted_keys.size(), secret_, modulus_);
        rebuild_hierarchy();
    }

    size_t size() const noexcept { return keys_.size(); }
    size_t segment_count() const noexcept { return segments_.size(); }
    void set_compact_limit(size_t limit) noexcept { keys_.set_compact_limit(limit); }
    size_t piece_len(size_t rank) const { return keys_.piece_len(rank); }
    bool key_is_piece_tail(uint64_t key) const {
        size_t rank = keys_.lower_bound_rank(key);
        if (rank >= keys_.size() || keys_[rank] != key) return false;
        return keys_.is_piece_tail(rank);
    }
    void set_zero_tail(bool on) noexcept { keys_.set_zero_tail(on); }
    std::vector<size_t> level_segment_counts() const {
        std::vector<size_t> counts;
        counts.push_back(segments_.size());
        for (const auto &lvl : upper_levels_) counts.push_back(lvl.size());
        return counts;
    }
    size_t boundary_count() const noexcept { return keys_.boundary_count(); }
    size_t compaction_count() const noexcept { return keys_.compaction_count(); }
    uint64_t compact_ns() const noexcept { return keys_.compact_ns(); }
    uint64_t compact_max_ns() const noexcept { return keys_.compact_max_ns(); }
    size_t max_piece_keys() const { return keys_.max_piece_keys(); }
    const KeyStore::EditTrace &last_edit() const noexcept { return keys_.last_edit(); }
    uint64_t last_refit_ns() const noexcept { return last_refit_ns_; }

    std::vector<uint64_t> leaf_start_keys() const {
        return leaf_starts_;
    }
    size_t level_count() const noexcept { return upper_levels_.size() + 1; }
    size_t epsilon() const noexcept { return epsilon_; }
    uint32_t modulus() const noexcept { return modulus_; }
    PLAMode mode() const noexcept { return mode_; }
    uint64_t key_at(size_t rank) const { return keys_[rank]; }
    size_t copy_keys(size_t rank, uint64_t *dst, size_t maxn) const {
        return keys_.copy_from(rank, dst, maxn);
    }
    const std::vector<CanonicalSegment> &segments() const noexcept { return segments_; }

    size_t model_bytes() const noexcept {
        size_t total = segments_.size() * sizeof(CanonicalSegment);
        for (const auto &lvl : upper_levels_) {
            total += lvl.size() * sizeof(CanonicalSegment);
        }
        return total;
    }

    size_t total_bytes() const noexcept {
        return model_bytes() + keys_.size() * sizeof(uint64_t);
    }

    std::vector<uint8_t> serialize_bytes() const {
        std::vector<uint8_t> buf;
        const char magic[4] = {'S', 'H', 'I', '2'};
        uint32_t eps32 = static_cast<uint32_t>(epsilon_);
        uint32_t mod32 = modulus_;
        uint8_t mode_u8 = static_cast<uint8_t>(mode_);
        uint32_t n_levels = static_cast<uint32_t>(1 + upper_levels_.size());

        size_t total_segs = segments_.size();
        for (const auto &lvl : upper_levels_) total_segs += lvl.size();
        buf.reserve(24 + n_levels * 4 + total_segs * sizeof(CanonicalSegment));

        auto append = [&](const void *p, size_t len) {
            const auto *b = static_cast<const uint8_t *>(p);
            buf.insert(buf.end(), b, b + len);
        };

        append(magic, 4);
        append(&eps32, sizeof(eps32));
        append(&mod32, sizeof(mod32));
        append(&mode_u8, sizeof(mode_u8));
        append(&n_levels, sizeof(n_levels));

        uint32_t leaf_n = static_cast<uint32_t>(segments_.size());
        append(&leaf_n, sizeof(leaf_n));
        if (!segments_.empty()) {
            append(segments_.data(), segments_.size() * sizeof(CanonicalSegment));
        }

        for (const auto &lvl : upper_levels_) {
            uint32_t ln = static_cast<uint32_t>(lvl.size());
            append(&ln, sizeof(ln));
            if (!lvl.empty()) {
                append(lvl.data(), lvl.size() * sizeof(CanonicalSegment));
            }
        }
        return buf;
    }

    size_t leaf_segment_for_key(uint64_t key) const noexcept {
        if (segments_.size() <= 1 || key <= segments_[0].start_key) {
            return 0;
        }
        if (upper_levels_.empty()) {
            return binary_search_segment(segments_, 0, segments_.size() - 1, key);
        }

        size_t idx = 0;
        for (int lvl = static_cast<int>(upper_levels_.size()) - 1; lvl >= 0; --lvl) {
            const auto &segs = upper_levels_[lvl];
            const auto &seg = segs[idx];
            size_t next_base = (idx + 1 < segs.size())
                                   ? upper_prefixes_[lvl][idx + 1]
                                   : (upper_prefixes_[lvl][idx] + seg.count());
            size_t pred = std::min(next_base, upper_prefixes_[lvl][idx] + seg.predict_fast(key));

            const auto &child = (lvl == 0) ? segments_ : upper_levels_[lvl - 1];
            const size_t child_n = child.size();
            const size_t radius = epsilon_recursive_ + 1;
            size_t lo = (pred > radius) ? (pred - radius) : 0;
            if (lo >= child_n) lo = child_n - 1;
            // The model error is supposed to be a few segments. Past that, binary
            // search. An unbounded walk is correct but, on a wide key space, it
            // scans every segment of the level for every lookup.
            size_t walked = 0;
            const size_t walk_limit = radius + 8;
            for (; walked < walk_limit && lo + 1 < child_n && child[lo + 1].start_key <= key; ++lo, ++walked) {
            }
            if (lo + 1 < child_n && child[lo + 1].start_key <= key) {
                lo = binary_search_segment(child, lo, child_n - 1, key);
            }
            idx = lo;
        }
        return idx;
    }

    bool contains(uint64_t key) const noexcept {
        return lookup(key) != keys_.size();
    }

    size_t lookup(uint64_t key) const noexcept {
        const size_t n = keys_.size();
        if (n == 0) return 0;
        uint64_t k = std::max(keys_.front(), key);
        const FastNode *it = fast_segment_for_key(k);
        size_t idx = static_cast<size_t>(it - fast_nodes_.data());
        int64_t pred = static_cast<int64_t>((*it)(k)) + leaf_bias_at(idx);
        int64_t cap = static_cast<int64_t>((it + 1)->prefix) + leaf_bias_at(idx + 1);
        if (pred < 0) pred = 0;
        if (cap < 0) cap = 0;
        size_t pos = std::min(static_cast<size_t>(pred), static_cast<size_t>(cap));
        size_t lo = (pos <= epsilon_ + 1) ? 0 : (pos - (epsilon_ + 1));
        size_t hi = std::min(n, pos + epsilon_ + 3);
        size_t span = hi - lo;
        if (span > 64) span = 64;
        uint64_t window[64];
        const uint64_t *base = keys_.try_span(lo, span);
        size_t copied = span;
        if (base == nullptr) {
            copied = keys_.copy_from(lo, window, span);
            base = window;
        }
        auto kit = std::lower_bound(base, base + copied, key);
        if (kit != base + copied && *kit == key) {
            return lo + static_cast<size_t>(kit - base);
        }
        return n;
    }

    RefitStats erase(uint64_t key) {
        size_t rank = keys_.lower_bound_rank(key);
        if (rank >= keys_.size() || keys_[rank] != key) {
            throw std::out_of_range("key not found in SHIIndex::erase");
        }
        keys_.erase_at(rank, secret_, modulus_);
        auto t0 = std::chrono::steady_clock::now();
        RefitStats stats = refit_around(key);
        auto t1 = std::chrono::steady_clock::now();
        last_refit_ns_ = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
        return stats;
    }

    RefitStats insert(uint64_t key) {
        size_t rank = keys_.lower_bound_rank(key);
        if (rank < keys_.size() && keys_[rank] == key) {
            throw std::invalid_argument("duplicate key in SHIIndex::insert");
        }
        keys_.insert_key(key, secret_, modulus_);
        return refit_around(key);
    }

    size_t absolute_segments_changed_vs(const SHIIndex &other) const noexcept {
        size_t min_s = std::min(segments_.size(), other.segments_.size());
        size_t max_s = std::max(segments_.size(), other.segments_.size());
        size_t pref_a = 0;
        size_t pref_b = 0;
        size_t i = 0;
        for (; i < min_s; ++i) {
            if (segments_[i] != other.segments_[i] || pref_a != pref_b) {
                break;
            }
            pref_a += segments_[i].count();
            pref_b += other.segments_[i].count();
        }
        return max_s - i;
    }

    // Pure model re-segmentation on pre-updated key array
    void set_keys_raw(std::vector<uint64_t> new_keys) {
        keys_.assign_sorted(new_keys.data(), new_keys.size(), secret_, modulus_);
    }

    RefitStats refit_only(uint64_t pivot_key) {
        return refit_around(pivot_key);
    }

private:
#pragma pack(push, 1)
    struct FastNode {
        uint64_t key;
        float slope;
        int32_t prefix;

        inline size_t operator()(uint64_t k) const noexcept {
            if (k < key) return 0;
            uint64_t prod = slope_times_gap(slope, k - key);
            int64_t raw = static_cast<int64_t>(std::min<uint64_t>(prod, static_cast<uint64_t>(INT64_MAX))) +
                          static_cast<int64_t>(prefix);
            if (raw < 0) return 0;
            return static_cast<size_t>(raw);
        }
    };
#pragma pack(pop)

    KeyStore keys_;
    std::vector<CanonicalSegment> segments_;
    std::vector<uint64_t> leaf_starts_;
    std::vector<size_t> leaf_prefixes_;
    std::vector<int64_t> leaf_bias_;
    std::vector<std::vector<CanonicalSegment>> upper_levels_;
    std::vector<std::vector<size_t>> upper_prefixes_;
    std::vector<FastNode> fast_nodes_;
    std::vector<size_t> fast_offsets_;
    size_t epsilon_;
    size_t epsilon_recursive_;
    uint64_t secret_;
    uint32_t modulus_;
    PLAMode mode_;
    bool upper_anchors_ = true;
    bool drop_half_ = false;
    uint32_t upper_modulus() const noexcept { return upper_anchors_ ? modulus_ : 0; }
    uint64_t last_refit_ns_ = 0;

    const FastNode *fast_segment_for_key(uint64_t key) const noexcept {
        if (fast_offsets_.size() <= 2) {
            const FastNode *begin = fast_nodes_.data();
            const FastNode *end = begin + segments_.size();
            return std::prev(std::upper_bound(
                begin, end, key,
                [](uint64_t k, const FastNode &nd) { return k < nd.key; }
            ));
        }
        const FastNode *it = fast_nodes_.data() + *(fast_offsets_.end() - 2);
        for (int l = static_cast<int>(fast_offsets_.size()) - 3; l >= 0; --l) {
            const FastNode *level_begin = fast_nodes_.data() + fast_offsets_[l];
            const FastNode *level_end = fast_nodes_.data() + fast_offsets_[l + 1] - 1;
            size_t pos = std::min<size_t>((*it)(key), static_cast<size_t>((it + 1)->prefix));
            size_t level_n = static_cast<size_t>(level_end - level_begin);
            if (pos > level_n) pos = level_n;
            const FastNode *lo = level_begin + ((pos <= epsilon_recursive_ + 1) ? 0 : (pos - (epsilon_recursive_ + 1)));
            if (lo >= level_end) lo = level_end - 1;
            const size_t walk_limit = epsilon_recursive_ + 8;
            size_t walked = 0;
            for (; walked < walk_limit && (lo + 1) < level_end && (lo + 1)->key <= key; ++lo, ++walked) {
            }
            if ((lo + 1) < level_end && (lo + 1)->key <= key) {
                const FastNode *found = std::upper_bound(
                    lo, level_end, key,
                    [](uint64_t k, const FastNode &nd) { return k < nd.key; }
                );
                lo = (found == level_begin) ? level_begin : std::prev(found);
            }
            it = lo;
        }
        return it;
    }

    static size_t binary_search_segment(
        const std::vector<CanonicalSegment> &segs,
        size_t lo,
        size_t hi,
        uint64_t key
    ) noexcept {
        size_t ans = lo;
        while (lo <= hi) {
            size_t mid = lo + ((hi - lo) >> 1);
            if (segs[mid].start_key <= key) {
                ans = mid;
                lo = mid + 1;
            } else {
                if (mid == 0) break;
                hi = mid - 1;
            }
        }
        return ans;
    }

    RefitStats refit_around(uint64_t pivot_key) {
        RefitStats stats{};
        if (keys_.empty()) {
            segments_.clear();
            leaf_starts_.clear();
            rebuild_hierarchy();
            return stats;
        }
        if (segments_.empty()) {
            std::vector<uint64_t> flat(keys_.size());
            keys_.copy_from(0, flat.data(), flat.size());
            segments_ = build_segments(flat.data(), flat.size(), epsilon_, secret_, modulus_, mode_);
            stats.segments_refit = segments_.size();
            stats.keys_scanned = keys_.size();
            rebuild_hierarchy();
            return stats;
        }

        size_t s = binary_search_segment(segments_, 0, segments_.size() - 1, pivot_key);
        size_t restart = (s == 0) ? 0 : (s - 1);
        stats.restart_segment = restart;

        size_t start_idx = static_cast<size_t>(true_leaf_prefix(restart));
        auto first_after = std::upper_bound(
            segments_.begin() + restart,
            segments_.end(),
            pivot_key,
            [](uint64_t k, const CanonicalSegment &seg) { return k < seg.start_key; }
        );
        size_t old_after_begin = static_cast<size_t>(first_after - segments_.begin());

        std::vector<CanonicalSegment> mid;
        size_t i = start_idx;
        const size_t n = keys_.size();
        size_t old_cursor = old_after_begin;
        size_t resync_idx = segments_.size();

        auto t_fit = std::chrono::steady_clock::now();
        while (i < n) {
            uint64_t ki = keys_[i];
            while (old_cursor < segments_.size() && segments_[old_cursor].start_key < ki) {
                ++old_cursor;
            }
            if (old_cursor < segments_.size() && segments_[old_cursor].start_key == ki) {
                resync_idx = old_cursor;
                break;
            }
            uint64_t window[kMaxSegmentKeys];
            size_t copied = keys_.copy_from(i, window, kMaxSegmentKeys);
            auto [seg, nxt] = fit_one_segment(window, copied, 0, epsilon_, secret_, modulus_, mode_);
            mid.push_back(seg);
            stats.keys_scanned += nxt;
            i += nxt;
        }
        auto t_upper = std::chrono::steady_clock::now();
        stats.fit_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t_upper - t_fit).count());

        stats.segments_refit = mid.size();
        stats.resync_old_segment = resync_idx;
        uint64_t suffix_key = std::numeric_limits<uint64_t>::max();
        if (resync_idx < segments_.size()) suffix_key = segments_[resync_idx].start_key;
        uint64_t region_begin = segments_[restart].start_key;

        size_t old_span = (resync_idx < segments_.size()) ? (resync_idx - restart) : (segments_.size() - restart);
        bool starts_identical = (mid.size() == old_span);
        if (starts_identical) {
            for (size_t k = 0; k < mid.size(); ++k) {
                if (segments_[restart + k].start_key != mid[k].start_key) {
                    starts_identical = false;
                    break;
                }
            }
        }

        size_t old_n = segments_.size();
        bool had_tail = resync_idx < old_n;
        int64_t old_tail_true = had_tail ? true_leaf_prefix(resync_idx) : 0;

        auto t_span = std::chrono::steady_clock::now();
        if (mid.size() == old_span) {
            for (size_t k = 0; k < mid.size(); ++k) {
                segments_[restart + k] = mid[k];
                leaf_starts_[restart + k] = mid[k].start_key;
            }
        } else {
            stats.span_rewritten = 1;
            std::vector<CanonicalSegment> updated;
            std::vector<uint64_t> starts;
            size_t tail_count = (resync_idx < segments_.size()) ? (segments_.size() - resync_idx) : 0;
            updated.reserve(restart + mid.size() + tail_count);
            starts.reserve(restart + mid.size() + tail_count);
            updated.insert(updated.end(), segments_.begin(), segments_.begin() + restart);
            starts.insert(starts.end(), leaf_starts_.begin(), leaf_starts_.begin() + restart);
            updated.insert(updated.end(), mid.begin(), mid.end());
            for (const auto &seg : mid) starts.push_back(seg.start_key);
            if (resync_idx < segments_.size()) {
                updated.insert(updated.end(), segments_.begin() + resync_idx, segments_.end());
                starts.insert(starts.end(), leaf_starts_.begin() + resync_idx, leaf_starts_.end());
            }
            segments_.swap(updated);
            leaf_starts_.swap(starts);
        }
        stats.span_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - t_span).count());

        if (segments_.empty() || upper_levels_.empty()) {
            rebuild_hierarchy();
        } else {
            patch_leaf_directory(restart, restart + mid.size(), resync_idx, old_n, had_tail, old_tail_true);
            if (!starts_identical) rebuild_upper_local(region_begin, suffix_key);
        }
        auto t_done = std::chrono::steady_clock::now();
        stats.upper_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t_done - t_upper).count());
        return stats;
    }

    static std::vector<size_t> compute_prefixes(const std::vector<CanonicalSegment> &segs) {
        std::vector<size_t> out(segs.size());
        size_t acc = 0;
        for (size_t i = 0; i < segs.size(); ++i) {
            out[i] = acc;
            acc += segs[i].count();
        }
        return out;
    }

    struct SpliceOut {
        bool starts_same = false;
        uint64_t tail_key = std::numeric_limits<uint64_t>::max();
    };

    // Refit segs over keys[region_begin, suffix_key). A tail that starts at or
    // after suffix_key is an unchanged suffix of the child array, so it is copied.
    SpliceOut splice_model(
        std::vector<CanonicalSegment> &segs,
        const uint64_t *keys,
        size_t n,
        uint64_t region_begin,
        uint64_t suffix_key,
        size_t eps,
        const std::vector<size_t> *prefs = nullptr
    ) {
        SpliceOut out;
        if (segs.empty() || n == 0) return out;
        size_t s = binary_search_segment(segs, 0, segs.size() - 1, region_begin);
        size_t restart = (s == 0) ? 0 : (s - 1);
        size_t start_idx = 0;
        if (prefs && restart < prefs->size()) start_idx = (*prefs)[restart];
        else {
            for (size_t i = 0; i < restart; ++i) start_idx += segs[i].count();
        }
        if (start_idx > n) start_idx = n;
        auto first_after = std::upper_bound(
            segs.begin() + static_cast<std::ptrdiff_t>(restart),
            segs.end(),
            region_begin,
            [](uint64_t k, const CanonicalSegment &seg) { return k < seg.start_key; }
        );
        size_t old_cursor = static_cast<size_t>(first_after - segs.begin());
        std::vector<CanonicalSegment> mid;
        size_t i = start_idx;
        size_t resync = segs.size();
        while (i < n) {
            uint64_t ki = keys[i];
            while (old_cursor < segs.size() && segs[old_cursor].start_key < ki) ++old_cursor;
            if (old_cursor < segs.size() && segs[old_cursor].start_key == ki && ki >= suffix_key) {
                resync = old_cursor;
                break;
            }
            size_t take = std::min(n - i, static_cast<size_t>(kMaxSegmentKeys));
            auto [seg, nxt] = fit_one_segment(keys + i, take, 0, eps, secret_, upper_modulus(), mode_);
            if (nxt == 0) break;
            mid.push_back(seg);
            i += nxt;
        }
        size_t old_span = (resync < segs.size()) ? (resync - restart) : (segs.size() - restart);
        out.starts_same = (mid.size() == old_span);
        if (out.starts_same) {
            for (size_t k = 0; k < mid.size(); ++k) {
                if (segs[restart + k].start_key != mid[k].start_key) {
                    out.starts_same = false;
                    break;
                }
            }
        }
        if (resync < segs.size()) out.tail_key = segs[resync].start_key;
        if (mid.size() == old_span) {
            for (size_t k = 0; k < mid.size(); ++k) segs[restart + k] = mid[k];
        } else {
            std::vector<CanonicalSegment> updated;
            size_t tail = (resync < segs.size()) ? (segs.size() - resync) : 0;
            updated.reserve(restart + mid.size() + tail);
            updated.insert(updated.end(), segs.begin(), segs.begin() + static_cast<std::ptrdiff_t>(restart));
            updated.insert(updated.end(), mid.begin(), mid.end());
            if (resync < segs.size()) {
                updated.insert(updated.end(), segs.begin() + static_cast<std::ptrdiff_t>(resync), segs.end());
            }
            segs.swap(updated);
        }
        return out;
    }

    void rebuild_upper_local(uint64_t region_begin, uint64_t suffix_key) {
        if (upper_levels_.empty() || epsilon_recursive_ == 0 || segments_.size() <= 1) {
            rebuild_hierarchy();
            return;
        }
        const uint64_t *child_keys = leaf_starts_.data();
        size_t child_n = leaf_starts_.size();
        std::vector<uint64_t> next;
        for (size_t level = 0; level < upper_levels_.size(); ++level) {
            const std::vector<size_t> *prefs =
                (level < upper_prefixes_.size()) ? &upper_prefixes_[level] : nullptr;
            SpliceOut spliced = splice_model(
                upper_levels_[level], child_keys, child_n, region_begin, suffix_key, epsilon_recursive_, prefs
            );
            if (upper_levels_[level].empty() ||
                (drop_half_ ? upper_levels_[level].size() * 2 > child_n
                             : upper_levels_[level].size() >= child_n)) {
                rebuild_hierarchy();
                return;
            }
            upper_prefixes_[level] = compute_prefixes(upper_levels_[level]);
            if (spliced.starts_same || level + 1 == upper_levels_.size()) break;
            next.resize(upper_levels_[level].size());
            for (size_t i = 0; i < next.size(); ++i) next[i] = upper_levels_[level][i].start_key;
            suffix_key = spliced.tail_key;
            child_keys = next.data();
            child_n = next.size();
        }
        rebuild_upper_fast_only();
    }

    static constexpr size_t kPrefBlock = 1024;

    int64_t leaf_bias_at(size_t i) const {
        size_t b = i / kPrefBlock;
        return (b < leaf_bias_.size()) ? leaf_bias_[b] : 0;
    }

    int64_t true_leaf_prefix(size_t i) const {
        return static_cast<int64_t>(leaf_prefixes_[i]) + leaf_bias_at(i);
    }

    void set_leaf_row(size_t i, size_t acc) {
        int64_t b = leaf_bias_at(i);
        leaf_prefixes_[i] = static_cast<size_t>(static_cast<int64_t>(acc) - b);
        fast_nodes_[i].key = segments_[i].start_key;
        fast_nodes_[i].slope = segments_[i].slope;
        fast_nodes_[i].prefix = static_cast<int32_t>(
            static_cast<int64_t>(acc) + segments_[i].rel_intercept - b
        );
    }

    void add_tail_delta(size_t first, size_t last_inclusive, int64_t delta) {
        if (delta == 0 || first > last_inclusive) return;
        size_t block = first / kPrefBlock;
        size_t partial_end = std::min(last_inclusive + 1, (block + 1) * kPrefBlock);
        for (size_t i = first; i < partial_end; ++i) {
            if (i < leaf_prefixes_.size()) {
                leaf_prefixes_[i] = static_cast<size_t>(static_cast<int64_t>(leaf_prefixes_[i]) + delta);
            }
            fast_nodes_[i].prefix = static_cast<int32_t>(static_cast<int64_t>(fast_nodes_[i].prefix) + delta);
        }
        size_t nblocks = last_inclusive / kPrefBlock + 1;
        if (leaf_bias_.size() < nblocks) leaf_bias_.resize(nblocks, 0);
        for (size_t b = block + 1; b < nblocks; ++b) leaf_bias_[b] += delta;
    }

    void patch_leaf_directory(
        size_t restart,
        size_t new_tail,
        size_t old_tail,
        size_t old_n,
        bool had_tail,
        int64_t old_tail_true
    ) {
        ptrdiff_t shift = static_cast<ptrdiff_t>(segments_.size()) - static_cast<ptrdiff_t>(old_n);
        size_t shift_abs = static_cast<size_t>(shift < 0 ? -shift : shift);
        if (shift_abs >= kPrefBlock) {
            for (size_t i = 0; i < leaf_prefixes_.size(); ++i) {
                int64_t b = leaf_bias_at(i);
                if (b != 0) leaf_prefixes_[i] = static_cast<size_t>(static_cast<int64_t>(leaf_prefixes_[i]) + b);
            }
            for (size_t i = 0; i <= old_n && i < fast_nodes_.size(); ++i) {
                int64_t b = leaf_bias_at(i);
                if (b != 0) {
                    fast_nodes_[i].prefix = static_cast<int32_t>(static_cast<int64_t>(fast_nodes_[i].prefix) + b);
                }
            }
            leaf_bias_.clear();
            old_tail_true = had_tail ? static_cast<int64_t>(leaf_prefixes_[old_tail]) : 0;
        }
        if (shift > 0) {
            fast_nodes_.insert(fast_nodes_.begin() + static_cast<ptrdiff_t>(old_tail), shift_abs, FastNode{});
            leaf_prefixes_.insert(leaf_prefixes_.begin() + static_cast<ptrdiff_t>(old_tail), shift_abs, 0);
        } else if (shift < 0) {
            fast_nodes_.erase(
                fast_nodes_.begin() + static_cast<ptrdiff_t>(new_tail),
                fast_nodes_.begin() + static_cast<ptrdiff_t>(old_tail)
            );
            leaf_prefixes_.erase(
                leaf_prefixes_.begin() + static_cast<ptrdiff_t>(new_tail),
                leaf_prefixes_.begin() + static_cast<ptrdiff_t>(old_tail)
            );
        }
        if (shift != 0 && shift_abs < kPrefBlock) {
            size_t new_n = segments_.size();
            for (size_t block = new_tail / kPrefBlock; block <= new_n / kPrefBlock; ++block) {
                size_t lo = std::max(new_tail, block * kPrefBlock);
                size_t hi = std::min(new_n, (block + 1) * kPrefBlock - 1);
                if (shift < 0) {
                    size_t edge = (hi + 1 > shift_abs) ? (hi + 1 - shift_abs) : 0;
                    lo = std::max(lo, edge);
                } else {
                    hi = std::min(hi, lo + shift_abs - 1);
                }
                if (lo > hi) continue;
                for (size_t j = lo; j <= hi; ++j) {
                    ptrdiff_t oldj = static_cast<ptrdiff_t>(j) - shift;
                    if (oldj < 0) continue;
                    int64_t corr = leaf_bias_at(static_cast<size_t>(oldj)) - leaf_bias_at(j);
                    if (corr == 0) continue;
                    if (j < leaf_prefixes_.size()) {
                        leaf_prefixes_[j] = static_cast<size_t>(static_cast<int64_t>(leaf_prefixes_[j]) + corr);
                    }
                    fast_nodes_[j].prefix = static_cast<int32_t>(static_cast<int64_t>(fast_nodes_[j].prefix) + corr);
                }
            }
        }
        size_t acc = 0;
        if (restart > 0) acc = static_cast<size_t>(true_leaf_prefix(restart - 1)) + segments_[restart - 1].count();
        for (size_t k = restart; k < new_tail; ++k) {
            set_leaf_row(k, acc);
            acc += segments_[k].count();
        }
        if (had_tail) add_tail_delta(new_tail, segments_.size(), static_cast<int64_t>(acc) - old_tail_true);
        else if (segments_.size() < fast_nodes_.size()) {
            int64_t b = leaf_bias_at(segments_.size());
            fast_nodes_[segments_.size()].prefix = static_cast<int32_t>(static_cast<int64_t>(acc) - b);
            fast_nodes_[segments_.size()].key = std::numeric_limits<uint64_t>::max();
            fast_nodes_[segments_.size()].slope = 0.0f;
        }
    }

    void rebuild_upper_fast_only() {
        constexpr uint64_t SENTINEL = std::numeric_limits<uint64_t>::max();
        size_t leaf_end = segments_.size() + 1;
        if (fast_nodes_.size() < leaf_end) fast_nodes_.resize(leaf_end);
        else fast_nodes_.resize(leaf_end);
        fast_offsets_.clear();
        fast_offsets_.push_back(0);
        fast_offsets_.push_back(leaf_end);
        for (size_t l = 0; l < upper_levels_.size(); ++l) {
            const auto &segs = upper_levels_[l];
            const auto &prefs = upper_prefixes_[l];
            size_t child_count = (l == 0) ? segments_.size() : upper_levels_[l - 1].size();
            for (size_t i = 0; i < segs.size(); ++i) {
                fast_nodes_.push_back(FastNode{
                    segs[i].start_key,
                    segs[i].slope,
                    static_cast<int32_t>(prefs[i]) + segs[i].rel_intercept
                });
            }
            fast_nodes_.push_back(FastNode{SENTINEL, 0.0f, static_cast<int32_t>(child_count)});
            fast_offsets_.push_back(fast_nodes_.size());
        }
    }

    void rebuild_fast_nodes() {
        constexpr uint64_t SENTINEL = std::numeric_limits<uint64_t>::max();
        fast_nodes_.clear();
        fast_offsets_.clear();
        leaf_bias_.clear();
        if (segments_.empty()) return;
        size_t total_fast = segments_.size() + 2;
        for (const auto &lvl : upper_levels_) total_fast += lvl.size() + 2;
        fast_nodes_.reserve(total_fast);
        fast_offsets_.push_back(0);
        for (size_t i = 0; i < segments_.size(); ++i) {
            fast_nodes_.push_back(FastNode{
                segments_[i].start_key,
                segments_[i].slope,
                static_cast<int32_t>(leaf_prefixes_[i]) + segments_[i].rel_intercept
            });
        }
        fast_nodes_.push_back(FastNode{SENTINEL, 0.0f, static_cast<int32_t>(keys_.size())});
        fast_offsets_.push_back(fast_nodes_.size());
        for (size_t l = 0; l < upper_levels_.size(); ++l) {
            const auto &segs = upper_levels_[l];
            const auto &prefs = upper_prefixes_[l];
            size_t child_count = (l == 0) ? segments_.size() : upper_levels_[l - 1].size();
            for (size_t i = 0; i < segs.size(); ++i) {
                fast_nodes_.push_back(FastNode{
                    segs[i].start_key,
                    segs[i].slope,
                    static_cast<int32_t>(prefs[i]) + segs[i].rel_intercept
                });
            }
            fast_nodes_.push_back(FastNode{SENTINEL, 0.0f, static_cast<int32_t>(child_count)});
            fast_offsets_.push_back(fast_nodes_.size());
        }
    }

    void rebuild_hierarchy() {
        leaf_starts_.resize(segments_.size());
        for (size_t i = 0; i < segments_.size(); ++i) leaf_starts_[i] = segments_[i].start_key;
        leaf_prefixes_ = compute_prefixes(segments_);
        upper_levels_.clear();
        upper_prefixes_.clear();
        fast_nodes_.clear();
        fast_offsets_.clear();
        if (segments_.empty()) {
            return;
        }
        if (epsilon_recursive_ > 0 && segments_.size() > 1) {
            const std::vector<CanonicalSegment> *curr = &segments_;
            while (curr->size() > 1) {
                std::vector<uint64_t> starts(curr->size());
                for (size_t i = 0; i < curr->size(); ++i) {
                    starts[i] = (*curr)[i].start_key;
                }
                auto upper = build_segments(
                    starts.data(),
                    starts.size(),
                    epsilon_recursive_,
                    secret_,
                    upper_modulus(),
                    mode_
                );
                if (drop_half_ ? (upper.size() * 2 > curr->size())
                               : (upper.size() >= curr->size())) {
                    break;
                }
                upper_prefixes_.push_back(compute_prefixes(upper));
                upper_levels_.push_back(std::move(upper));
                curr = &upper_levels_.back();
            }
        }

        rebuild_fast_nodes();
    }
};

} // namespace shi
