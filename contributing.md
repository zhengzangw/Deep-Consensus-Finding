# How to Contribute

We welcome your contribution to the project.

## Format

For more information, check `.vscode/settings.json` and `.pre-commit-config.yaml`

- formatter: black
- linter: flake8
- import sort: isort

Install pre-commit for commit code style checking.

```sh
pre-commit install
```

## Git Commit

Please follow commit message format (checked by commitizen)

```txt
pattern: (build|ci|docs|feat|fix|perf|refactor|style|test|chore|revert|bump)(\(\S+\))?!?:(\s.*)
```

## Test

```sh
pytest --cov=src tests/
```
