.PHONY: $(MAKECMDGOALS)
.EXPORT_ALL_VARIABLES:
SHELL := /bin/bash -euo pipefail

BLACK ?= \033[0;30m
RED ?= \033[0;31m
GREEN ?= \033[0;32m
YELLOW ?= \033[0;33m
BLUE ?= \033[0;34m
PURPLE ?= \033[0;35m
CYAN ?= \033[0;36m
GRAY ?= \033[0;37m
COFF ?= \033[0m


##################
# Local commands #
##################

initialize:
ifneq ($(wildcard .git),)
	@printf "$(CYAN)>>> Repository already initialized.$(COFF)\n"
else
	@printf "$(CYAN)>>> Initializing git repository.$(COFF)\n"
	git init
endif

## Install dependencies, including dev & test dependencies
deps: initialize
	@printf "$(CYAN)>>> Creating environment for project...$(COFF)\n"
	poetry install
	poetry run pre-commit install

## Run unit tests
test:
	@printf "$(CYAN)Running test suite$(COFF)\n"
	poetry run pytest --cov=src --cov-fail-under 60

## Run static code checkers and linters
check:
	@printf "$(CYAN)Running static code analysis and license generation$(COFF)\n"
	poetry run ruff check src tests notebooks
	@printf "All $(GREEN)done$(COFF)\n"

## Runs black formatter
lint:
	@printf "$(CYAN)Auto-formatting with black$(COFF)\n"
	poetry run ruff src tests --fix
	poetry run ruff format src tests notebooks
	@printf " >>> Generating $(CYAN)licenses.md$(COFF) file\n"
	poetry run pip-licenses --with-authors -f markdown --output-file ./licenses.md

## Removed the build, dist directories, pycache, pyo or pyc and swap files
clean:
	@printf "$(CYAN)Cleaning EVERYTHING!$(COFF)\n"
	@rm -rf build/
	@rm -rf dist/
	@rm -rf *.egg-info
	@find . -type d -name '__pycache__' -exec rm -rf {} +
	@find . -type f -name '*.py[co]' -exec rm -f {} +
	@find . -name '*~' -exec rm -f {} +
	@find . -type f -name '.DS_Store' -delete
	@printf "$(GREEN)>>> Removed$(COFF) pycache, .pyc, .pyo, .DS_Store files and files with ~\n"



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help
# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
