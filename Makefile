.PHONY: run install test clean

agent:
	@.venv/bin/python src/dqn_agent.py

run:
	@.venv/bin/python src/main.py

remove_default_features:
	@.venv/bin/python src/remove_default_features.py

convert_to_numeric_type:
	@.venv/bin/python src/convert_to_numeric_type.py

install:
	@.venv/bin/pip install -r requirements.txt

test:
	@.venv/bin/python -m pytest

clean:
	@rm -rf __pycache__ .pytest_cache
