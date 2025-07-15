# Contributing to MCP Whisper Transcription Server

Thank you for your interest in contributing to the MCP Whisper Transcription Server! This guide will help you get started with contributing to the project.

## 📋 Table of Contents

- [Code of Conduct](#-code-of-conduct)
- [Getting Started](#-getting-started)
- [Development Setup](#-development-setup)
- [Contributing Guidelines](#-contributing-guidelines)
- [Pull Request Process](#-pull-request-process)
- [Issue Reporting](#-issue-reporting)
- [Code Style and Standards](#-code-style-and-standards)
- [Testing Guidelines](#-testing-guidelines)
- [Documentation](#-documentation)
- [Community](#-community)

## 🤝 Code of Conduct

This project follows a standard code of conduct to ensure a welcoming environment for all contributors:

### Our Standards

- **Be respectful**: Treat all community members with respect and kindness
- **Be inclusive**: Welcome newcomers and help them get started
- **Be constructive**: Provide helpful feedback and suggestions
- **Be collaborative**: Work together towards common goals
- **Be patient**: Remember that everyone has different experience levels

### Unacceptable Behavior

- Harassment, discrimination, or intimidation of any kind
- Offensive, inappropriate, or unprofessional language
- Personal attacks or trolling
- Publishing private information without permission
- Any conduct that would be inappropriate in a professional setting

## 🚀 Getting Started

### Prerequisites

Before contributing, ensure you have:

- **Apple Silicon Mac** (M1, M2, M3, or later) for testing MLX functionality
- **Python 3.10 or later**
- **Git** for version control
- **Poetry** for dependency management
- **FFmpeg** for media file processing

### Areas for Contribution

We welcome contributions in several areas:

1. **🐛 Bug Fixes**: Fix issues reported in GitHub Issues
2. **✨ New Features**: Add new functionality or improve existing features
3. **📚 Documentation**: Improve guides, examples, and API documentation
4. **🧪 Testing**: Expand test coverage and add integration tests
5. **🎨 UX/UI**: Improve error messages and user experience
6. **⚡ Performance**: Optimize processing speed and memory usage
7. **🌐 Internationalization**: Add support for new languages

## 🛠️ Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/yourusername/mcp-whisper-transcription.git
cd mcp-whisper-transcription

# Add upstream remote
git remote add upstream https://github.com/galacoder/mcp-whisper-transcription.git
```

### 2. Install Dependencies

```bash
# Install Poetry if not already installed
curl -sSL https://install.python-poetry.org | python3 -

# Install project dependencies
poetry install

# Install development dependencies
poetry install --with dev
```

### 3. Install Pre-commit Hooks

```bash
# Install pre-commit hooks for code quality
poetry run pre-commit install

# Test pre-commit setup
poetry run pre-commit run --all-files
```

### 4. Verify Installation

```bash
# Run tests to ensure everything works
poetry run pytest

# Start the MCP server
poetry run python src/whisper_mcp_server.py

# Check code formatting
poetry run black --check .
poetry run isort --check-only .
```

## 📝 Contributing Guidelines

### Branch Naming

Use descriptive branch names with prefixes:

- `feature/add-new-model-support` - New features
- `fix/memory-leak-in-batch-processing` - Bug fixes
- `docs/improve-api-reference` - Documentation updates
- `test/add-integration-tests` - Test improvements
- `refactor/optimize-whisper-loading` - Code refactoring

### Commit Messages

Follow conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding or updating tests
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `style`: Code style changes
- `ci`: CI/CD changes

**Examples**:
```bash
feat(transcription): add support for WebM video format

fix(batch): resolve memory leak in concurrent processing

docs(api): update transcribe_file parameter documentation

test(mcp): add integration tests for MCP resource endpoints
```

### Code Organization

```
mcp-whisper-transcription/
├── src/
│   └── whisper_mcp_server.py    # Main MCP server implementation
├── tests/                       # Comprehensive test suite
│   ├── test_mcp_tools.py       # MCP tool tests
│   ├── test_resources.py       # MCP resource tests
│   ├── test_whisper_utils.py   # Utility function tests
│   └── test_performance.py     # Performance tests
├── examples/                    # Usage examples
├── docs/                        # Additional documentation
├── transcribe_mlx.py           # MLX Whisper integration
├── whisper_utils.py            # Utility functions
└── pyproject.toml              # Project configuration
```

## 🔄 Pull Request Process

### 1. Before Starting Work

1. **Check existing issues**: Look for related issues or discussions
2. **Create an issue**: For new features, create an issue to discuss the approach
3. **Get feedback**: Discuss your planned changes with maintainers
4. **Assign yourself**: Assign the issue to yourself to avoid duplicate work

### 2. Development Process

1. **Create a branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make changes**: Implement your feature or fix
3. **Write tests**: Add tests for new functionality
4. **Update documentation**: Update relevant documentation
5. **Test thoroughly**: Run all tests and manual testing

### 3. Before Submitting

```bash
# Format code
poetry run black .
poetry run isort .

# Run linting
poetry run flake8 src/ tests/

# Run all tests
poetry run pytest --cov=src --cov-report=html

# Type checking (optional but recommended)
poetry run mypy src/ --ignore-missing-imports

# Test MCP server functionality
poetry run python src/whisper_mcp_server.py --help
```

### 4. Submit Pull Request

1. **Push your branch**:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create PR**: Create a pull request on GitHub
3. **Fill out template**: Use the PR template and provide detailed information
4. **Request review**: Request review from maintainers
5. **Address feedback**: Respond to review comments promptly

### Pull Request Template

```markdown
## Description
Brief description of what this PR does.

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] All existing tests pass
- [ ] New tests added for new functionality
- [ ] Manual testing completed
- [ ] Performance impact assessed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] No breaking changes (or breaking changes documented)

## Related Issues
Fixes #123
Related to #456
```

## 🐛 Issue Reporting

### Before Creating an Issue

1. **Search existing issues**: Check if the issue already exists
2. **Check documentation**: Review docs and troubleshooting guide
3. **Test with latest version**: Ensure you're using the latest version
4. **Gather information**: Collect relevant system information

### Bug Report Template

```markdown
## Bug Description
Clear description of the bug.

## Steps to Reproduce
1. Step one
2. Step two
3. Step three

## Expected Behavior
What you expected to happen.

## Actual Behavior
What actually happened.

## Environment
- macOS version: 
- Python version: 
- Poetry version: 
- MLX version: 
- Hardware: (M1/M2/M3, RAM)

## Additional Context
- Error messages/logs
- Screenshots (if applicable)
- Sample files (if relevant)
```

### Feature Request Template

```markdown
## Feature Description
Clear description of the feature you'd like to see.

## Use Case
Describe your use case and why this feature would be valuable.

## Proposed Solution
Your ideas for how this could be implemented.

## Alternatives Considered
Other approaches you've considered.

## Additional Context
Any other relevant information.
```

## 🎨 Code Style and Standards

### Python Style Guide

We follow PEP 8 with some modifications:

```python
# Line length: 100 characters (configured in pyproject.toml)
# Use Black for formatting
# Use isort for import sorting

# Good examples:
async def transcribe_file(
    file_path: str,
    model: str = "mlx-community/whisper-large-v3-turbo",
    output_formats: str = "txt,md",
    language: str = "en"
) -> Dict[str, Any]:
    """Transcribe audio/video file using MLX Whisper.
    
    Args:
        file_path: Path to the audio/video file
        model: Whisper model to use
        output_formats: Comma-separated output formats
        language: Language code for transcription
        
    Returns:
        Dict containing transcription results
        
    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If invalid parameters provided
    """
    # Implementation here
    pass
```

### Naming Conventions

- **Functions/variables**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private methods**: `_leading_underscore`
- **MCP tools**: `descriptive_verb_noun` (e.g., `transcribe_file`, `batch_transcribe`)

### Documentation Standards

```python
def example_function(param1: str, param2: int = 10) -> bool:
    """One-line summary of what the function does.
    
    Longer description if needed. Explain the purpose,
    behavior, and any important notes.
    
    Args:
        param1: Description of param1
        param2: Description of param2 with default value
        
    Returns:
        Description of return value
        
    Raises:
        SpecificError: When this error occurs
        
    Examples:
        >>> example_function("test", 5)
        True
    """
```

## 🧪 Testing Guidelines

### Test Structure

```python
# tests/test_feature.py
import pytest
from unittest.mock import AsyncMock, patch
from src.whisper_mcp_server import app


class TestFeature:
    """Test class for specific feature."""
    
    @pytest.mark.asyncio
    async def test_basic_functionality(self):
        """Test basic functionality works correctly."""
        # Test implementation
        pass
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling works correctly."""
        # Test implementation
        pass
    
    @pytest.mark.parametrize("input_val,expected", [
        ("test1", "result1"),
        ("test2", "result2"),
    ])
    async def test_multiple_inputs(self, input_val, expected):
        """Test function with multiple input values."""
        # Test implementation
        pass
```

### Test Categories

1. **Unit Tests**: Test individual functions and classes
2. **Integration Tests**: Test MCP tool and resource endpoints
3. **Performance Tests**: Test speed and memory usage
4. **Error Handling Tests**: Test error conditions and edge cases
5. **File Format Tests**: Test various audio/video formats

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=src --cov-report=html

# Run specific test file
poetry run pytest tests/test_mcp_tools.py -v

# Run tests with specific marker
poetry run pytest -m "not slow"

# Run performance tests
poetry run pytest tests/test_performance.py
```

### Writing Good Tests

```python
# Good test example
@pytest.mark.asyncio
async def test_transcribe_file_success(self, mock_client):
    """Test successful file transcription."""
    # Arrange
    test_file = "test_audio.mp3"
    expected_text = "Hello, this is a test."
    
    # Act
    async with Client(mock_client) as client:
        result = await client.call_tool("transcribe_file", {
            "file_path": test_file
        })
    
    # Assert
    assert result["text"] == expected_text
    assert result["duration"] > 0
    assert len(result["output_files"]) > 0
```

## 📚 Documentation

### Documentation Types

1. **API Documentation**: Function/class docstrings
2. **User Guides**: Setup, usage, and troubleshooting
3. **Examples**: Practical usage examples
4. **Code Comments**: Inline explanations for complex logic

### Documentation Standards

- Use clear, concise language
- Provide practical examples
- Include error handling guidance
- Update documentation with code changes
- Test documentation examples

### Building Documentation

```bash
# Generate API docs (if using sphinx/mkdocs)
poetry run sphinx-build docs/ docs/_build/

# Test documentation examples
poetry run python docs/examples/test_examples.py
```

## 👥 Community

### Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Documentation**: Check existing guides and examples

### Communication Guidelines

- Be respectful and professional
- Provide clear, detailed information
- Include relevant context and examples
- Be patient with responses
- Help others when you can

### Recognition

Contributors will be recognized:

- In the CHANGELOG.md file
- In the repository contributors list
- In release notes for significant contributions

## 🎯 Development Roadmap

### Current Priorities

1. **Performance Optimization**: Improve transcription speed and memory usage
2. **Error Handling**: Better error messages and recovery
3. **Model Support**: Add support for new Whisper models
4. **Documentation**: Expand examples and guides
5. **Testing**: Increase test coverage to 95%+

### Future Goals

- **Cloud Integration**: Support for cloud-based models
- **Real-time Processing**: Live transcription capabilities
- **Advanced Features**: Speaker diarization, sentiment analysis
- **Platform Support**: Windows and Linux compatibility

### How to Help

1. **Pick an issue**: Look for "good first issue" or "help wanted" labels
2. **Join discussions**: Participate in issue discussions
3. **Review PRs**: Help review other contributors' work
4. **Improve docs**: Update and expand documentation
5. **Test new features**: Help test beta features

## 📄 License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.

## 🙏 Thank You

Thank you for contributing to the MCP Whisper Transcription Server! Your contributions help make this tool better for everyone in the community.

For questions about contributing, feel free to create an issue or start a discussion on GitHub.