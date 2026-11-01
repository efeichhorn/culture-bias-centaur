PYTHON ?= python
WORKSPACE ?= data/workspace
export MPLCONFIGDIR ?= $(CURDIR)/build/.matplotlib

.PHONY: results data analysis figures test

results:
	$(PYTHON) scripts/reproduce_summary.py

data:
	$(PYTHON) scripts/prepare_paper_data.py --output "$(WORKSPACE)"

analysis:
	$(PYTHON) scripts/reproduce_analyses.py --workspace "$(WORKSPACE)"

figures:
	$(PYTHON) scripts/plot_finetuning_pipeline.py
	$(PYTHON) scripts/plot_figure2_distributions_nll.py
	$(PYTHON) scripts/plot_h2_h3.py
	$(PYTHON) scripts/plot_results_revision.py
	$(PYTHON) scripts/plot_h4.py

test:
	$(PYTHON) -m pytest -q tests
