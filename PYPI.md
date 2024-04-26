### Packaging for PyPI

Install `build` and `twine`:

```
pip install --user --upgrade build
pip install --user --upgrade twine
```

Clean the build directory:

```
make clean
```

Generate the source distribution and wheel:

```
python -m build
```

> [!warning]
> Use a test server first

#### Test Upload

Upload to test server:

```
python -m twine upload --repository testpypi dist/*
```

Install from test server and check that it works:

```
python -m pip install --index-url https://test.pypi.org/simple/ --no-deps mlx-llm
```

#### Upload

```
python -m twine upload dist/*
```

#### Acknowledgements

Thanks Apple mlx team for the (guide)[https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/UPLOAD.md].
