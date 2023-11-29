isort-format:
	isort .

isort-check:
	isort --check-only .

black-format:
	black .

black-check:
	black . --check 

ruff-format:
	ruff --fix .

ruff-check:
	ruff .

mypy:
	mypy .

format: isort-format black-format ruff-format

lint: isort-check black-check mypy ruff-check

.PHONY: clean install lint format
