dev:
	uv run marimo edit .

setup:
	pip install uv
	uv sync

install-hooks:
	cp scripts/pre-push .git/hooks/pre-push
	chmod +x .git/hooks/pre-push
	@echo "✅ Git pre-push hook installed"
