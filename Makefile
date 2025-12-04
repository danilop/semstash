# semstash Development Makefile

.PHONY: help install check test test-integration lint format pre-commit pre-commit-install docs coverage clean

help:
	@echo "semstash Development Commands"
	@echo ""
	@echo "  make install            Install dev dependencies"
	@echo "  make check              Run all checks (pre-commit hooks)"
	@echo "  make check-fast         Run checks without mypy and AWS tests"
	@echo "  make test               Run unit tests (mocked AWS)"
	@echo "  make test-integration   Run integration tests (real AWS)"
	@echo "  make coverage           Run tests with coverage report"
	@echo "  make lint               Run ruff linter"
	@echo "  make format             Format code with ruff"
	@echo "  make pre-commit-install Install pre-commit git hooks"
	@echo "  make docs               Generate API documentation"
	@echo "  make clean              Remove build artifacts"

install:
	uv sync --all-extras

check:
	uv run pre-commit run --all-files

check-fast:
	SKIP=mypy,pytest-aws uv run pre-commit run --all-files

test:
	uv run pytest tests/ --ignore=tests/test_integration.py

test-integration:
	uv run pytest tests/test_integration.py --use-aws

lint:
	uv run ruff check src/ tests/

format:
	uv run ruff format src/ tests/

vulture:
	uv run vulture src/ vulture_whitelist.py --min-confidence 80

mypy:
	uv run mypy src/

pre-commit-install:
	@echo "Installing pre-commit hooks..."
	@echo "Note: If you have core.hooksPath set globally, you may need to unset it first:"
	@echo "  git config --unset-all core.hooksPath"
	uv run pre-commit install

docs:
	uv run pdoc --output-dir docs/api src/semstash

coverage:
	uv run pytest tests/ --ignore=tests/test_integration.py --cov=semstash --cov-report=term-missing --cov-report=html
	@echo "HTML report: open htmlcov/index.html"

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .ruff_cache
	rm -rf docs/api/* htmlcov/ .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
