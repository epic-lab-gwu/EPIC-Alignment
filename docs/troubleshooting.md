# Troubleshooting

## `mkdocs: command not found`

Install MkDocs first:

```bash
pip install mkdocs
```

## Docs build fails with `--strict`

Common causes:

- broken markdown link
- nav entry in `mkdocs.yml` points to a missing file

Check:

```bash
mkdocs build --strict
```

## Cannot publish with GitHub Pages

If repository is private and your plan does not include private Pages,
GitHub will block deployment.

Current recommended mode for this repo:

- local preview (`mkdocs serve`)
- local docs build check in CI (`mkdocs build --strict`)

## PR cannot merge due workflow conflicts

If conflict appears in `.github/workflows/ci.yml`:

1. keep `main` branch version as baseline
2. re-apply only your intended workflow edits
3. push conflict resolution commit to the same PR branch

## `git push` rejected with `fetch first`

Run:

```bash
git fetch origin main
git rebase origin/main
git push origin main
```

If there are local uncommitted files, stash first:

```bash
git stash push -m "temp"
```
