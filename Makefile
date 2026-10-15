PYTHON ?= python3

.PHONY: install test figures smoke

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	PYTHONPATH=code $(PYTHON) -m unittest discover -s code/tests -v

figures:
	$(PYTHON) code/plot_figures.py --output-dir build/figures

smoke:
	$(PYTHON) code/run_experiments.py all --smoke --output-dir build/smoke
