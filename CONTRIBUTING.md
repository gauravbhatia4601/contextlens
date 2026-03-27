# Contributing to ContextLens

Thank you for your interest in contributing to ContextLens!

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/contextlens.git
   cd contextlens
   ```

2. Install in editable mode with dev dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

3. Run tests:
   ```bash
   pytest tests/ -v
   ```

## Code Style

- Follow PEP 8 style guidelines
- Use type hints for all function signatures
- Write docstrings for all public functions and classes
- Keep lines under 100 characters

## Testing

All PRs must include tests for new functionality. Run the test suite before submitting:

```bash
pytest tests/ -v --tb=short
```

## PR Guidelines

1. Create a feature branch from `main`
2. Make your changes with clear commit messages
3. Add/update tests as needed
4. Ensure all tests pass
5. Submit a PR with a clear description of changes

## Reporting Issues

When reporting bugs, please include:

- Python version
- OS and version
- Steps to reproduce
- Expected vs actual behavior
- Any relevant error messages

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
