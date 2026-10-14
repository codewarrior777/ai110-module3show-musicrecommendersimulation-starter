# Contributing to Music Recommender

Thanks for your interest in improving the Music Recommender!

## Setup

```bash
git clone https://github.com/codewarrior777/ai110-module3show-musicrecommendersimulation-starter.git
cd ai110-module3show-musicrecommendersimulation-starter
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows
pip install -e ".[dev]"
```

## Tests

```bash
python -m pytest tests/ -v
python -m pytest tests/ --cov=src --cov-config=.coveragerc --cov-report=term-missing
```

## Code Style

Uses **ruff** for linting and formatting. Pre-commit hooks run automatically on `git commit` if installed:

```bash
pre-commit install
```

Manual checks:

```bash
ruff check .
ruff format .
```

## PR Checklist

- [ ] Tests pass
- [ ] Coverage remains at 100% on `src/recommender.py`
- [ ] `ruff check .` passes
- [ ] `ruff format --check .` passes
- [ ] README updated if a user-facing feature was added

## Commit Convention

We use [Conventional Commits](https://www.conventionalcommits.org/):
`feat:`, `fix:`, `docs:`, `test:`, `chore:`, `style:`, `ci:`.