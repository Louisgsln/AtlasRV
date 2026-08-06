.PHONY: install quality test demo research compare dashboard build docker clean

install:
	python -m pip install -e ".[dev]"

quality:
	ruff check src tests
	mypy src

test:
	pytest

demo:
	atlas-rv demo --output reports/demo

research:
	atlas-rv research --provider synthetic --config configs/universe.yml --output reports/research

compare:
	atlas-rv compare-models --provider synthetic --config configs/universe.yml --pair oil_energy

dashboard:
	streamlit run dashboard/app.py

build:
	python -m build
	twine check dist/*

docker:
	docker build -t atlasrv .

clean:
	python -c "from pathlib import Path; import shutil; [shutil.rmtree(p, ignore_errors=True) for p in [Path('.pytest_cache'), Path('.ruff_cache'), Path('.mypy_cache'), Path('build'), Path('dist'), Path('reports/demo'), Path('reports/research')]]"
