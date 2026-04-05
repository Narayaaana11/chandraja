# Contributing to Chandraja

Thanks for contributing.

## Development Setup

1. Clone the repository and enter the folder.
2. Create and activate a virtual environment.
3. Install dependencies using one of:
   - `pip install -r requirements.txt`
   - `pip install -r smart_eval_requirements.txt`
4. Run checks before opening a PR:
   - `python -m ruff check . --select F401,F841`
   - `python -m pytest -q`
   - `python verify_system.py`

## Branch and Commit Guidelines

- Use small, focused pull requests.
- Write clear commit messages in imperative mood.
- Keep unrelated changes out of the same PR.

## Pull Request Checklist

- Tests pass locally.
- New behavior includes tests when practical.
- Documentation is updated when behavior changes.
- No secrets or local machine paths are introduced.

## Reporting Issues

Use GitHub Issues and include:

- Steps to reproduce
- Expected behavior
- Actual behavior
- Environment details (OS, Python version)
