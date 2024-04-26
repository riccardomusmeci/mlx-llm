SHELL = /bin/bash

pkg_name = mlx-llm
src_pkg = src/$(pkg_name)

.PHONY: all install clean

all:
	make clean
	make format
	make install

install:
    # Uncomment the following line if you want to run a prebuild script (must exist)
	pip install -e .[dev]

clean:
	-rm -rf .benchmarks
	-rm -rf .mypy_cache
	-rm -rf prof
	-rm -rf build
	-rm -rf .eggs
	-rm -rf dist
	-rm -rf *.egg-info
	-find . -not -path "./.git/*" -name logs -exec rm -rf {} \;
	-rm -rf .ruff_cache
	-find . -not -path "./.git/*" -name '.benchmarks' -exec rm -rf {} \;

format:
	ruff check --show-fixes .
	black .
	mypy .

