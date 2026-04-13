# Contributing to `mtopt`

Thank you for your interest in contributing! This document explains how to get started, submit changes, and keep your work aligned with the project's standards.

## Table of contents

- [Code of conduct](#code-of-conduct)
- [Getting started](#getting-started)
- [Development workflow](#development-workflow)
- [Commit messages](#commit-messages)
- [Pull request checklist](#pull-request-checklist)
- [Coding style](#coding-style)
- [Running tests](#running-tests)
- [Building documentation](#building-documentation)

---

## Code of conduct

This project adheres to the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/1/4/code-of-conduct.html). By participating, you are expected to uphold this code. Please report unacceptable behaviour to the project maintainers.

---

## Getting started

### Fork and clone

1. **Fork** the repository on GitHub — click the _Fork_ button on [https://github.com/terra-quantum-public/mtopt](https://github.com/terra-quantum-public/mtopt).
2. **Clone** your fork locally:

   ```bash
   # HTTPS
   git clone https://github.com/<your-username>/mtopt.git

   # or SSH
   git clone git@github.com:<your-username>/mtopt.git

   cd mtopt
   ```

3. **Add the upstream remote** so you can keep your fork in sync with the canonical repository:

   ```bash
   # HTTPS
   git remote add upstream https://github.com/terra-quantum-public/mtopt.git

   # or SSH
   git remote add upstream git@github.com:terra-quantum-public/mtopt.git
   ```

4. **Verify** your remotes:

   ```bash
   git remote -v
   # origin    git@github.com:<your-username>/mtopt.git (fetch)
   # origin    git@github.com:<your-username>/mtopt.git (push)
   # upstream  git@github.com:terra-quantum-public/mtopt.git (fetch)
   # upstream  git@github.com:terra-quantum-public/mtopt.git (push)
   ```

### Install dependencies

The project uses [Poetry](https://python-poetry.org/) for dependency management:

```bash
poetry install
```

---

## Development workflow

1. **Sync your fork** before starting new work:

   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   git push origin main
   ```

2. **Create a feature branch** from `main`:

   ```bash
   git checkout -b feat/my-awesome-feature
   ```

3. Make your changes, write tests, and commit using [conventional commits](#commit-messages).

4. **Push** your branch to your fork:

   ```bash
   git push origin feat/my-awesome-feature
   ```

5. **Open a pull request** against `terra-quantum-public/mtopt` `main` from GitHub. Fill in the PR template and link any relevant issues.

6. Address review feedback — push additional commits to the same branch; the PR will update automatically.

---

## Commit messages

This project follows the [Conventional Commits](https://www.conventionalcommits.org/) specification. Well-structured commit messages make the changelog and review process much easier.

### Format

```
<type>(<optional scope>): <short summary>

[optional body]

[optional footer(s)]
```

### Common types

| Type       | When to use                                                   |
|------------|---------------------------------------------------------------|
| `feat`     | A new feature                                                 |
| `fix`      | A bug fix                                                     |
| `docs`     | Documentation-only changes                                    |
| `style`    | Formatting, whitespace — no logic change                      |
| `refactor` | Code restructuring without adding features or fixing bugs     |
| `test`     | Adding or correcting tests                                    |
| `chore`    | Build process, dependency updates, tooling                    |
| `perf`     | Performance improvements                                      |
| `ci`       | CI/CD configuration changes                                   |

### Examples

```
feat(trc): add support for adaptive rank selection

fix(grid): handle empty coordinate list gracefully

docs: add fork-and-clone instructions to CONTRIBUTING.md

chore: bump numpy from 2.3.5 to 2.4.4
```

Breaking changes should be indicated with a `!` after the type/scope and a `BREAKING CHANGE:` footer:

```
feat(api)!: rename `optimize` to `run_optimization`

BREAKING CHANGE: the `optimize` method has been renamed to `run_optimization`
across all optimizer classes.
```

---

## Pull request checklist

Before opening a PR, please confirm that:

- [ ] Tests have been added for any new functionality in `tests/`
- [ ] All existing tests still pass (`pytest tests`)
- [ ] New public functions include [NumPy-style docstrings](https://numpydoc.readthedocs.io/en/latest/format.html)
- [ ] Code is formatted with `black .`
- [ ] Linting passes (`ruff check .` / `pylint mtopt`)
- [ ] If suitable, the change is documented in `docs/` or demonstrated in an example

---

## Coding style

The project follows the [Black](https://black.readthedocs.io/en/stable/) code style. Run Black before submitting:

```bash
black .
```

Additional linting is handled by [Ruff](https://docs.astral.sh/ruff/) and [Pylint](https://pylint.readthedocs.io/):

```bash
ruff check .
pylint mtopt
```

---

## Running tests

Tests are executed with [pytest](https://docs.pytest.org/):

```bash
pytest tests
```

---

## Building documentation

Documentation is built with [Sphinx](https://www.sphinx-doc.org/). A convenience script is provided:

```bash
./generate_docs.sh
```

Full documentation is published at [mtopt.readthedocs.io](https://mtopt.readthedocs.io/en/latest/).
