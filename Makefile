.PHONY: install quality test demo clean

install:
	python -m pip install -e ".[dev]"

quality:
	ruff check src tests
	mypy src

test:
	pytest

demo:
	atlas-rv demo --output reports/demo

clean:
	python -c "from pathlib import Path; import shutil; [shutil.rmtree(p, ignore_errors=True) for p in [Path('.pytest_cache'), Path('.ruff_cache'), Path('.mypy_cache'), Path('reports/demo')]]"

