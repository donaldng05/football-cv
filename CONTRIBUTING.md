# Contributing to `football-cv`

Thank you for your interest in contributing to `football-cv`! This document outlines our development workflow, engineering standards, and submission process.

---

## 1. Branch Strategy & Workflow

We follow a GitHub-driven branch and PR workflow:

1. **GitHub Issue First:** Every non-trivial change, refactor, or feature starts with a tracked GitHub Issue detailing the problem, scope, and acceptance criteria.
2. **Branch Naming:** Create a dedicated branch off `main` using descriptive prefixes:
   - `docs/<feature-or-fix>`
   - `refactor/<component>`
   - `feat/<feature-name>`
   - `test/<test-suite>`
   - `fix/<bug-description>`
   - `analysis/<benchmark-or-evaluation>`
3. **Atomic Commits:** Structure your work into small, logical, conventional commits. Do not bundle refactoring, new features, and test additions into a single commit.
4. **Pull Requests:** Open a PR targeting `main` using the repository's PR template, linking the relevant issue (e.g. `Closes #<id>`).

---

## 2. Development Setup

Clone the repository and install with development dependencies:

```bash
git clone https://github.com/donaldng05/football-cv.git
cd football-cv
python -m venv .venv

# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -e ".[dev]"
```

---

## 3. Code Standards & Quality Checks

All code must pass static analysis and formatting checks before merging:

* **Linter & Formatter:** [Ruff](https://astral.sh/ruff)
  ```bash
  ruff check .
  ruff format --check .
  ```
* **Test Suite:** [pytest](https://docs.pytest.org/)
  ```bash
  pytest tests/ -v
  ```
* **Type Hints & Clean APIs:** Use clear type annotations and avoid raw dictionary passing where dataclasses or schemas exist.

---

## 4. Conventional Commit Guidelines

We use [Conventional Commits](https://www.conventionalcommits.org/):

* `feat:` A new user-facing feature or analytics capability
* `fix:` A bug fix in detection, tracking, or analytics logic
* `refactor:` Code restructuring without changing external behavior
* `test:` Adding or improving unit/integration tests
* `docs:` Documentation updates, attribution, or README revisions
* `chore:` Build scripts, dependencies, CI configuration, or repo hygiene
* `bench:` Benchmarks, profiling scripts, and runtime evaluation

---

## 5. Attribution & Integrity

`football-cv` began with an adapted tutorial tracking baseline and evolved into a production package with original analytics. Never remove attribution notices or claim third-party assets (weights, datasets) as proprietary. See [docs/attribution.md](docs/attribution.md) for details.
