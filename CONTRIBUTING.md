# Contributing to the Hierarchical RCA System

Thank you for your interest in contributing!

## Getting Started

1. **Fork** the repository and clone your fork.
2. **Create a branch** for your change:
   ```bash
   git checkout -b feat/my-improvement
   ```
3. **Install dev dependencies:**
   ```bash
   make install-dev
   ```
4. **Make your changes**, then run tests to verify nothing is broken:
   ```bash
   make test
   ```
5. **Open a pull request** against `main` with a clear description of what changed and why.

---

## High-Impact Areas

The easiest places to make a meaningful contribution:

| Area | File(s) | Difficulty |
|------|---------|------------|
| Improve Level 3 accuracy (currently 40% F1) | `src/model.py`, `src/trainer.py` | Medium |
| Add cross-validation to training | `src/trainer.py` | Easy |
| Implement transformer training loop | `src/model.py` `_train_transformer` | Hard |
| Extend the taxonomy | `data/taxonomy.json` | Easy |
| Add a REST API (FastAPI) | `src/api.py` (new) | Medium |
| Add a batch prediction CLI flag | `predict.py` | Easy |

---

## Code Style

- Follow **PEP 8** (use a linter like `flake8` or `ruff`).
- Use Python **type annotations** for function signatures.
- Use `logging` instead of `print()` for diagnostic output.
- Write a docstring for every public function and class.

---

## Adding Tests

All new functionality must have tests in `tests/`. Run the suite with:

```bash
make test          # all tests
make test-cov      # with coverage report
```

Tests use `pytest`. Place fixtures shared across test files in `tests/conftest.py`.

---

## Pull Request Checklist

- [ ] Tests added/updated and passing (`make test`)
- [ ] No new linting errors
- [ ] PR description explains the *why*, not just the *what*
- [ ] `CHANGELOG` entry added (if user-visible change)

---

## Reporting Issues

Open a GitHub issue with:
- A clear title
- Steps to reproduce
- Expected vs. actual behaviour
- Python version and OS
