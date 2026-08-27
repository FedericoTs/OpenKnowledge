.PHONY: help install test test-browser lint fmt typecheck check serve index costs learn review conflicts eval eval-conflicts eval-safety docker clean

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## create the venv and install with dev extras
	uv venv && uv pip install -e ".[dev,anthropic]"

test:  ## run the test suite
	uv run pytest -q

test-browser:  ## also check the public page in a real browser
	uv pip install -e ".[dev,browser]" && uv run playwright install chromium
	uv run pytest tests/test_website_layout.py -q

lint:  ## check formatting and lint rules
	uv run ruff check src tests tools
	uv run ruff format --check src tests tools

fmt:  ## apply formatting and autofixes
	uv run ruff format src tests tools
	uv run ruff check --fix src tests tools

typecheck:  ## run mypy
	uv run mypy

check: lint typecheck test  ## everything CI runs

learn:  ## draft answers from changed documents (spends tokens)
	uv run openknowledge learn

review:  ## drafted answers awaiting approval
	uv run openknowledge review

conflicts:  ## documents that disagree with each other
	uv run openknowledge conflicts

eval:  ## run the golden set (accuracy and cost)
	uv run openknowledge eval

eval-conflicts:  ## measure contradiction detection (no model needed)
	uv run openknowledge eval-conflicts

eval-safety:  ## run only the must-refuse cases
	uv run openknowledge eval --only refusal

serve:  ## run the server with reload
	uv run openknowledge serve --reload --port 8080

index:  ## re-read the document folder
	uv run openknowledge index

costs:  ## what the bot has actually cost
	uv run openknowledge costs

docker:  ## build and run the container stack
	docker compose up --build

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache dist build data
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
