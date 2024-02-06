SHELL = /bin/bash

pkg_name = mlx-llm
src_pkg = src/$(pkg_name)

.PHONY: all install clean check test test_fast full coverage ci doc doc_publish
all:
	make clean
	make install

install:
    # Uncomment the following line if you want to run a prebuild script (must exist)
	pip install -e .[dev,test,docs]
	make full

clean:
	-rm -rf .benchmarks
	-rm -rf .mypy_cache
	-rm -rf prof
	-rm -rf build
	-rm -rf .eggs
	-rm -rf dist
	-rm -rf *.egg-info
	-find . -not -path "./.git/*" -name logs -exec rm -rf {} \;
	-rm -rf logs
	-rm .coverage
	-rm -rf .ruff_cache
	-find . -not -path "./.git/*" -name '.benchmarks' -exec rm -rf {} \;

check:
	ruff check --diff .
	black --check --diff .

format:
	ruff check --show-fixes .
	black .

full:
	make check
