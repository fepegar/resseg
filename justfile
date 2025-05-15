default:
    @just --list

bump part="patch":
    uv run bump-my-version bump {{part}} --verbose

bump-dry part="patch":
    uv run bump-my-version bump {{part}} --dry-run --verbose --allow-dirty

push:
    git push && git push --tags
