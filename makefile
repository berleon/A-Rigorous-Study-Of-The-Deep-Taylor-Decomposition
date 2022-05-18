sources = lrp_relations

.PHONY: test format lint unittest coverage pre-commit clean
test: format lint unittest

format:
	isort $(sources) tests
	black $(sources) tests

lint:
	flake8 $(sources) tests
	mypy $(sources) tests

unittest:
	pytest

coverage:
	pytest --cov=$(sources) --cov-branch --cov-report=term-missing tests

pre-commit:
	pre-commit run --all-files

clean:
	rm -rf .mypy_cache .pytest_cache
	rm -rf *.egg-info
	rm -rf .tox dist site
	rm -rf coverage.xml .coverage


download_clevr:
	mkdir -p ../data/clevr
	cd ../data/clevr && \
		wget https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip && \
		unzip CLEVR_v1.0.zip

nltk:
	# which python /srv/data/leonsixt/lrp_fails_the_sanity_check/venv/nltk_data
	python -m nltk.downloader all -d "`dirname \`which python\``/../nltk_data"
