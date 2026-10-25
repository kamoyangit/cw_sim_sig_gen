CXX ?= c++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Iinclude
LDFLAGS ?=

CORE_SRC := src/cwchannel.cpp src/cwkeyer.cpp
BIN := build/gen_cw

.PHONY: all clean test roundtrip

all: $(BIN)

$(BIN): $(CORE_SRC) src/gen_cw.cpp | build
	$(CXX) $(CXXFLAGS) -o $@ $(CORE_SRC) src/gen_cw.cpp $(LDFLAGS)

build:
	mkdir -p build

test: $(BIN)
	python3 -m unittest discover -s tests -p 'test_*.py' -v

# Closed-loop check of the measurement estimators (needs python3 + numpy + scipy).
roundtrip: $(BIN)
	python3 calibrate.py roundtrip

clean:
	rm -rf build
