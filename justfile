# Teeny Data Framework - Just commands
# Uses Podman for container management

# Set Podman socket as DOCKER_HOST (Podman forwards to /var/run/docker.sock)
export DOCKER_HOST := "unix:///var/run/docker.sock"

# List all available commands
default:
    @just --list

# Run all tests (using Podman)
test:
    pytest tests/test_table_functions_postgres.py -v

# Run tests with verbose output
test-verbose:
    pytest tests/test_table_functions_postgres.py -vv

# Run specific test class
test-class CLASS:
    pytest tests/test_table_functions_postgres.py::{{CLASS}} -v

# Run specific test function
test-func FUNC:
    pytest tests/test_table_functions_postgres.py::{{FUNC}} -v

# Run tests and stop on first failure
test-fast:
    pytest tests/test_table_functions_postgres.py -v -x

# Run only failed tests from last run
test-failed:
    pytest tests/test_table_functions_postgres.py -v --lf

# Check if Podman is running
check-podman:
    @podman info > /dev/null 2>&1 && echo "✓ Podman is running" || echo "✗ Podman is not running"

# Install development dependencies
install:
    uv sync --group dev

# Install git hooks (pre-push hook runs tests)
install-hooks:
    #!/usr/bin/env bash
    cat > .git/hooks/pre-push << 'EOF'
    #!/bin/bash
    # Pre-push hook: Run tests before pushing to remote

    echo "🧪 Running tests before push..."
    echo ""

    # Check if Podman is running
    if ! podman info > /dev/null 2>&1; then
        echo "❌ Podman is not running!"
        echo "Please start Podman with: podman machine start"
        exit 1
    fi

    # Run tests using just
    if command -v just &> /dev/null; then
        just test
    else
        # Fallback to pytest if just is not installed
        export DOCKER_HOST=unix:///var/run/docker.sock
        pytest tests/test_table_functions_postgres.py -v
    fi

    # Check exit code
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ All tests passed! Proceeding with push..."
        exit 0
    else
        echo ""
        echo "❌ Tests failed! Push aborted."
        echo "Fix the failing tests before pushing."
        echo ""
        echo "To skip this hook (not recommended), use: git push --no-verify"
        exit 1
    fi
    EOF
    chmod +x .git/hooks/pre-push
    @echo "✅ Git pre-push hook installed!"
    @echo "Tests will run automatically before every git push"

# Clean Python cache files
clean:
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
