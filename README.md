# Artifact: Do Learned Index Models Remember What You Deleted?

ICDE 2027 research-track artifact for the paper
*Do Learned Index Models Remember What You Deleted? Strong History Independence for Piecewise-Linear Models*.

- Author: Anubhav Pandey, CSE, IIT Delhi
- Email: cs5221136@iitd.ac.in
- ORCID: [0009-0006-9755-5871](https://orcid.org/0009-0006-9755-5871)
- Paper source: `paper/ieee.tex` (IEEE conference, 10pt). Compiled PDF: `paper/ieee.pdf`.

This repository is the supplemental URL for CMT. It contains the code, paper source, and logs used to produce the tables. It does not contain the SOSD binaries.

## What is required to reproduce the tables

1. This repository.
2. A C++17 compiler (`g++` 13 or `clang++`). macOS and Linux both build the fuzzer and the SOSD probes.
3. The SOSD files from Zenodo [15240501](https://zenodo.org/records/15240501), not shipped here:
   - `wiki_ts_200M_uint64` (90,437,011 distinct keys after sort/unique)
   - `fb_200M_uint64` (200,000,000)
   - `books_200M_uint32` (199,999,999, pass `--u32`)
   - `osm_cellids_800M_uint64` (800,000,000)
4. About 32 GiB RAM for wiki/fb/books and about 64 GiB plus a 4n virtual reservation for osm. The numbers in the paper were taken on a Dell PowerEdge R760, Xeon Gold 6430, 251 GiB, GCC 13.3.

PostgreSQL is optional. The paper's live-engine claim is an in-process audit, not a WAL-backed access method.

## Layout

| Path | Role |
| --- | --- |
| `include/shi/shi_pla.hpp` | Relative-count PLA, splice, arena, serialization of Repr |
| `include/shi/hi_btree.hpp` | Hash-promotion tree (byte and lookup control) |
| `include/shi/bptree.hpp` | B+ baseline |
| `src/bench_dist.cpp` | Table of 512 deletes (copy before the clock, packed start keys) |
| `src/bench_probe.cpp` | Levels, modulus sweep, million-delete stream, drop-half + packed-key identity |
| `src/bench_open.cpp` | ALEX / dynamic PGM full-file payload and tombstone checks |
| `src/bench_base.cpp` | B+ / mix / range |
| `src/test_fuzz.cpp` | 4,000-check byte-identity fuzzer |
| `third_party/` | Snapshots of PGM-index, ALEX, and LIPP used in the paper |
| `results/probe/` | Host logs that match the printed SOSD tables |
| `paper/` | `ieee.tex`, `refs.bib`, `ieee.pdf` |

## Build

```bash
make sosd          # bench_probe, bench_dist, bench_base, bench_open
make fuzz          # build/test_fuzz
./build/test_fuzz  # 4,000 fuzzer checks
```

On the measurement host the SOSD binaries were compiled with:

```bash
g++ -std=c++17 -O3 -march=native -Wno-pragmas \
  -Iinclude -Ithird_party/PGM-index/include \
  -o build/bench_probe src/bench_probe.cpp
```

Paper PDF (optional):

```bash
make -C paper
```

## Commands that produced the paper tables

Set `DATA` to the directory that holds the Zenodo files. Do not start a second osm 512-delete job on a host that is already running one.

Packed 512-delete distribution (`tab:dist`), copy outside the clock, vacated tail cleared:

```bash
./build/bench_dist --file $DATA/wiki_ts_200M_uint64 --deletes 512 --checks 8
./build/bench_dist --file $DATA/fb_200M_uint64 --deletes 512 --checks 8
./build/bench_dist --file $DATA/books_200M_uint32 --u32 --deletes 512 --checks 8
./build/bench_dist --file $DATA/osm_cellids_800M_uint64 --deletes 512 --checks 8
```

Million in-place deletes with the slide (`tab:million`):

```bash
./build/bench_probe --file $DATA/wiki_ts_200M_uint64 --zero --stream 1000000 --audit
```

and the same `--stream 1000000 --audit` on fb, books (`--u32`), and osm.

Modulus sweep (`tab:modtime`): `--sweep --sample 32 --zero`.

2× upper-level shrink variant (space, then Repr + packed-key identity):

```bash
./build/bench_probe --file $DATA/wiki_ts_200M_uint64 --levels --drop-half
./build/bench_probe --file $DATA/wiki_ts_200M_uint64 --drop-half --zero --checks 8
```

Logs already in the artifact: `results/probe/fresh_dist.txt`, `drop2_levels.txt`, `drop2_keys.txt`.

## Status of this artifact

- Available: public source and paper source.
- Functional: fuzzer and SOSD probes build from `make sosd` / `make fuzz`.
- Replicated: the printed SOSD means need the Zenodo files and a quiet machine in the same class as the paper host. Pilot JSON under `results/` is the cache-resident 200,000-key layout, not the SOSD tables.

LIPP is linked for the observation that the sources used in the paper provide no erase. Dynamic PGM and ALEX erase wall-clock is not a time for the byte condition and was not reported as such.

## License

Original code is MIT (see `LICENSE`). Vendored indexes keep Apache-2.0 / MIT as in `NOTICE`.
