# Local Docs

This project supports local documentation preview with MkDocs.

## Install Docs Dependency

```bash
pip install mkdocs
```

If you already use the dev extra:

```bash
pip install -e .[dev]
```

## Run Local Site

```bash
mkdocs serve
```

Open:

- `http://127.0.0.1:8000`

For LAN sharing (same network):

```bash
mkdocs serve -a 0.0.0.0:8000
```

Then teammates can open `http://<your-ip>:8000`.

## Build Static Site

```bash
mkdocs build --strict
```

Generated output is under `site/`.

## Recommended Workflow

1. Edit files in `docs/`
2. Keep `mkdocs serve` running for live reload
3. Run `mkdocs build --strict` before commit
