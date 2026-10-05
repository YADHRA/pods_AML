.PHONY: install stub test data
install:
	pip install -e .
stub:
	python scripts/make_stubs.py
test:
	pytest -q
data:
	python scripts/run_data.py
