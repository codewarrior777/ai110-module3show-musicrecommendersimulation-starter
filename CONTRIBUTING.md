# Contributing to Music Recommender

Thanks for your interest in improving the Music Recommender!

## Setup

```bash
git clone https://github.com/codewarrior777/ai110-module3show-musicrecommendersimulation-starter.git
cd ai110-module3show-musicrecommendersimulation-starter
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows
# source .venv/bin/activate     # macOS / Linux

# Install project + dev dependencies
pip install -e ".[dev]"
# (Alternatively: pip install -r requirements.txt)
```

## Running the Project

The project ships **four entry points**, all built on the same core
module (`src/recommender.py`):

```bash
# CLI — runs 3 profiles × 3 ranking modes, prints tables
python src/main.py

# REST API — Swagger UI at http://127.0.0.1:8000/docs
cd src && uvicorn api:app --reload

# Streamlit UI — interactive front-end at http://localhost:8501
streamlit run app.py

# Benchmark — measures score_song and recommend_songs scaling
python benchmark.py
```

## Tests

```bash
# Run the suite (28 tests)
python -m pytest tests/ -v

# With coverage report
python -m pytest tests/ --cov=src --cov-config=.coveragerc --cov-report=term-missing
```

**Why coverage is scoped to `src/recommender.py`:** the core domain
module holds all business logic (loading, scoring, ranking). The wrapper
files (`api.py`, `main.py`, `app.py`) contain no logic that isn't
already tested through the domain layer, so they are excluded via
`.coveragerc`. See ADR-006 in `ARCHITECTURE.md` for the rationale.

## Code Style

Uses **ruff** for linting and formatting. Pre-commit hooks run
automatically on `git commit` if installed:

```bash
pre-commit install
```

Manual checks:

```bash
ruff check .
ruff format --check .
```

To auto-fix issues:

```bash
ruff check . --fix
ruff format .
```

Ruff is configured in `pyproject.toml` under `[tool.ruff]`. Markdown
files are excluded — code blocks in docs are illustration, not source.

## Docs You May Need to Update

If your change affects any of the following, update the corresponding doc:

| Change | Update |
|---|---|
| Scoring weights, new mode | `README.md` §2, `ARCHITECTURE.md` ADR-003 |
| Ranking behavior, diversity | `README.md` §2c, `model_card.md` §6 |
| Dataset shape (songs.csv) | `README.md` §2a, `model_card.md` §2 & §5 |
| REST endpoints, schemas | `README.md` §4, `ARCHITECTURE.md` ADR-006 |
| Streamlit UI features | `README.md` §5 |
| AI workflow | `ai_interactions.md` |

## PR Checklist

- [ ] Tests pass (`python -m pytest tests/ -v`)
- [ ] Coverage remains at 100% on `src/recommender.py`
- [ ] `ruff check .` passes
- [ ] `ruff format --check .` passes
- [ ] README updated if a user-facing feature was added
- [ ] Model Card updated if a limitation or bias changed
- [ ] ADRs updated if an architectural decision changed

## Commit Convention

We use [Conventional Commits](https://www.conventionalcommits.org/):

| Type | When to use |
|---|---|
| `feat:` | New feature (mode, endpoint, CLI flag) |
| `fix:` | Bug fix |
| `docs:` | Documentation only |
| `test:` | Test changes |
| `chore:` | Tooling, deps, config |
| `style:` | Formatting, whitespace |
| `ci:` | GitHub Actions workflow changes |

Examples:

```
feat: add "mood-first" ranking mode
fix(api): validate k in GET /recommend
docs: align model card with scoring formula
```