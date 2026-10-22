PYTHON ?= python

.PHONY: results figures test paper overleaf

results:
	$(PYTHON) scripts/reproduce_summary.py

figures:
	$(PYTHON) scripts/plot_paper.py

test:
	$(PYTHON) -m pytest -q tests

paper:
	cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

overleaf:
	$(PYTHON) scripts/package_overleaf.py
