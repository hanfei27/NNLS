PYTHON ?= python3

.PHONY: install test figures

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	PYTHONPATH=code $(PYTHON) -m unittest discover -s code/tests -p 'test_reproduce_paper_figures.py' -v

figures:
	PYTHONPATH=code $(PYTHON) code/reproduce_paper_figures.py --output-dir build/figures
