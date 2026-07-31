# Public repository audit

Audit date: 2026-07-31. This is a basic pattern/name review, not a guarantee that the repository or its history contains no sensitive data.

## Checks performed

- Reviewed tracked filenames for common credential, key, database, archive, log, and personal-document patterns.
- Searched the working tree and Git diffs/history for common API-key/private-key signatures, secret assignments, and user-specific absolute paths.
- Reviewed tracked file sizes and likely generated evaluation artifacts.
- Reviewed `.gitignore` and `.dockerignore` coverage.

No private keys or populated API credentials were identified by the pattern scan. `.env.example` contains an empty API-key placeholder; real `.env` variants are ignored.

## Findings requiring owner review

Historical evaluation JSON/CSV artifacts under `docs/runbooks/` contain Windows user paths and filenames from a personal document corpus. Some filenames reveal categories such as career records, taxes, government/benefit paperwork, travel, resumes, and academic/employment documents. The repository also contains those strings in Git history. The files reviewed appear to be retrieval/evaluation metadata rather than the original documents, but filenames and paths can still be sensitive personal information.

The artifacts may provide useful experimental evidence, so they were **not silently deleted**, rewritten, or removed from history during this audit. Before making the repository public, the owner should decide whether to:

1. keep a deliberately reviewed and redacted subset;
2. replace paths/titles with stable synthetic identifiers while preserving metric meaning; or
3. remove the artifacts in a normal commit and separately decide whether a history rewrite is warranted.

History rewriting is disruptive and was not performed. If the repository has already been shared, assume historical objects may remain available even after a normal deletion.

The source tree also contained a user-specific Windows default path in the file-sorter UI and personal paths in maintenance-script examples. Public defaults/examples were replaced with environment-driven or generic paths; test-only path strings were retained where they exercise Windows normalization behavior.

## Ignore coverage

`.gitignore` now covers environment/virtualenv files, caches, coverage output, logs, local databases, common private-key/container formats, model/download/dataset directories, backend snapshots, and local document directories. `.dockerignore` excludes those categories plus tests/docs/assets from worker image build context where they are not runtime dependencies.

Ignore rules only prevent new untracked files from being added accidentally. They do not remove already tracked files or detect all secret formats. Before publication, run a dedicated history-aware scanner such as Gitleaks or TruffleHog and manually inspect retained evaluation fixtures/artifacts.
