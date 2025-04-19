.PHONY: run install test clean

run:
	@.venv/bin/python src/main.py

remove_default_features:
	@.venv/bin/python src/remove_default_features.py

install:
	@.venv/bin/pip install -r requirements.txt

test:
	@.venv/bin/python -m pytest

clean:
	@rm -rf __pycache__ .pytest_cache
