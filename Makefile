CXX ?= clang++
CXXFLAGS ?= -std=c++17 -O3 -Wno-\#pragma-messages -Wno-deprecated-declarations -Wno-nontrivial-memcall
INCLUDES = -Iinclude -Ithird_party/PGM-index/include -Ithird_party/ALEX/src/core -Ithird_party/lipp/src/core

PG_CONFIG ?= pg_config
PG_INC_SERVER := $(shell $(PG_CONFIG) --includedir-server)
PG_INC := $(shell $(PG_CONFIG) --includedir)
PG_BINDIR := $(shell $(PG_CONFIG) --bindir)

.PHONY: all test fuzz attacks perf pg sosd clean

all: build/test_fuzz build/bench_attacks build/bench_perf build/bench_e2e build/bench_probe build/bench_dist build/bench_base build/bench_open

build:
	mkdir -p build results

build/test_fuzz: src/test_fuzz.cpp include/shi/shi_pla.hpp include/shi/hi_btree.hpp include/shi/datasets.hpp | build
	$(CXX) $(CXXFLAGS) -Iinclude -Ithird_party/PGM-index/include src/test_fuzz.cpp -o build/test_fuzz

build/bench_attacks: src/bench_attacks.cpp include/shi/shi_pla.hpp include/shi/hi_btree.hpp include/shi/dp_index.hpp include/shi/datasets.hpp | build
	$(CXX) $(CXXFLAGS) $(INCLUDES) src/bench_attacks.cpp -o build/bench_attacks

build/bench_perf: src/bench_perf.cpp include/shi/shi_pla.hpp include/shi/hi_btree.hpp include/shi/datasets.hpp | build
	$(CXX) $(CXXFLAGS) $(INCLUDES) src/bench_perf.cpp -o build/bench_perf

build/bench_e2e: src/bench_e2e.cpp include/shi/shi_pla.hpp include/shi/datasets.hpp | build
	$(CXX) $(CXXFLAGS) $(INCLUDES) src/bench_e2e.cpp -o build/bench_e2e

build/bench_probe: src/bench_probe.cpp include/shi/shi_pla.hpp | build
	$(CXX) $(CXXFLAGS) -Iinclude -Ithird_party/PGM-index/include src/bench_probe.cpp -o build/bench_probe

build/bench_dist: src/bench_dist.cpp include/shi/shi_pla.hpp | build
	$(CXX) $(CXXFLAGS) -Iinclude -Ithird_party/PGM-index/include src/bench_dist.cpp -o build/bench_dist

build/bench_base: src/bench_base.cpp include/shi/shi_pla.hpp include/shi/bptree.hpp include/shi/hi_btree.hpp | build
	$(CXX) $(CXXFLAGS) $(INCLUDES) src/bench_base.cpp -o build/bench_base

build/bench_open: src/bench_open.cpp include/shi/shi_pla.hpp | build
	$(CXX) $(CXXFLAGS) $(INCLUDES) src/bench_open.cpp -o build/bench_open

sosd: build/bench_probe build/bench_dist build/bench_base build/bench_open

build/pg_shi_index.dylib: pg_shi_index/pg_shi_index.cpp include/shi/shi_pla.hpp include/shi/hi_btree.hpp | build
	$(CXX) $(CXXFLAGS) -fPIC -dynamiclib -undefined dynamic_lookup \
		-I"$(PG_INC_SERVER)" -I"$(PG_INC)" -Iinclude -Ithird_party/PGM-index/include \
		pg_shi_index/pg_shi_index.cpp -o build/pg_shi_index.dylib

fuzz: build/test_fuzz
	./build/test_fuzz

attacks: build/bench_attacks
	./build/bench_attacks

perf: build/bench_perf
	./build/bench_perf

pg: build/pg_shi_index.dylib
	rm -rf /tmp/pg_shi_data
	"$(PG_BINDIR)/initdb" -D /tmp/pg_shi_data --no-locale -E UTF8 >/dev/null
	"$(PG_BINDIR)/pg_ctl" -D /tmp/pg_shi_data -l /tmp/pg_shi_data/logfile -o "-p 55439 -k /tmp" start
	"$(PG_BINDIR)/psql" -h /tmp -p 55439 -d postgres \
		-v shi_lib_path="$(CURDIR)/build/pg_shi_index.dylib" \
		-f pg_shi_index/test_pg_shi.sql | tee results/pg14_audit.txt
	"$(PG_BINDIR)/pg_ctl" -D /tmp/pg_shi_data stop >/dev/null

test: fuzz
	python3 -m unittest discover -s tests -v

clean:
	rm -rf build /tmp/pg_shi_data
