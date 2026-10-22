.PHONY: help install data train evaluate pipeline test api dashboard report clean

PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip
PYTEST ?= .venv/bin/pytest
UVICORN ?= .venv/bin/uvicorn
STREAMLIT ?= .venv/bin/streamlit

help:
	@echo "HarborSense AI - Commands:"
	@echo "  make install     Create venv and install dependencies"
	@echo "  make data        Generate synthetic port congestion dataset"
	@echo "  make train       Train baseline and primary models"
	@echo "  make evaluate    Evaluate models and save metrics/figures"
	@echo "  make pipeline    Run full end-to-end pipeline and generate report"
	@echo "  make test        Run automated test suite"
	@echo "  make api         Launch FastAPI server on port 8000"
	@echo "  make dashboard   Launch Streamlit dashboard on port 8501"
	@echo "  make report      Compile PDF technical report"
	@echo "  make clean       Remove temporary files and caches"

install:
	python3.11 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

data:
	$(PYTHON) -m harborsense.data_generation

train:
	$(PYTHON) -m harborsense.train

evaluate:
	$(PYTHON) -m harborsense.evaluate

pipeline:
	$(PYTHON) scripts/run_pipeline.py

test:
	$(PYTEST) tests/

api:
	$(UVICORN) api.main:app --host 0.0.0.0 --port 8000 --reload

dashboard:
	$(STREAMLIT) run dashboard/app.py --server.port 8501

report:
	$(PYTHON) -m harborsense.report_generator

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf .coverage htmlcov/

