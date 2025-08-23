VENV = $(HOME)/.venvs/ag06-mixer
PY = $(VENV)/bin/python
PIP = $(VENV)/bin/pip
PYTEST = $(VENV)/bin/pytest

.PHONY: venv install health test run docker-build docker-up docker-down clean

venv:
	python3 -m venv $(VENV)
	$(PY) -m pip install --upgrade pip

install: venv
	$(PIP) install -r requirements.txt
	$(PIP) install pytest

health:
	PYTHONPATH=$(PWD) $(PY) health_check.py

test: health
	@echo "Health check passed, project structure is valid"

run:
	PYTHONPATH=$(PWD) $(PY) web_app.py

docker-build:
	docker compose -f deploy/docker-compose.yml build

docker-up: docker-build
	docker compose -f deploy/docker-compose.yml up -d

docker-down:
	docker compose -f deploy/docker-compose.yml down

docker-logs:
	docker compose -f deploy/docker-compose.yml logs -f ag06

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache

status:
	@echo "AGMixer Status:"
	@echo "  Web App: http://localhost:8001"
	@echo "  Health: make health"
	@echo "  Docker: make docker-up"