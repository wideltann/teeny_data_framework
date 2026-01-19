# Teeny Data Framework - Just commands
# Uses Podman for container management

# Set Podman socket as DOCKER_HOST (Podman forwards to /var/run/docker.sock)
export DOCKER_HOST := "unix:///var/run/docker.sock"

# List all available commands
default:
    @just --list

# Run all tests (using Podman)
test:
    pytest tests/ -v

# Run tests with verbose output
test-verbose:
    pytest tests/ -vv

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
    # Pre-push hook: Run ALL tests before pushing to remote

    set -e  # Exit immediately on any error

    echo "🧪 Running tests before push..."
    echo ""

    # Check if Podman is running (required for container tests)
    if ! podman info > /dev/null 2>&1; then
        echo "❌ Podman is not running!"
        echo "Container tests require Podman. Start it with: podman machine start"
        echo ""
        echo "Push aborted."
        exit 1
    fi

    # Run ALL tests (not just a subset)
    export DOCKER_HOST=unix:///var/run/docker.sock

    if command -v just &> /dev/null; then
        just test
    else
        pytest tests/ -v
    fi

    echo ""
    echo "✅ All tests passed! Proceeding with push..."
    EOF
    chmod +x .git/hooks/pre-push
    @echo "✅ Git pre-push hook installed!"
    @echo "Tests will run automatically before every git push"

# Clean Python cache files
clean:
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
