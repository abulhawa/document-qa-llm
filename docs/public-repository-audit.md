# Public repository privacy audit

Audit date: 2026-07-31.

This document records the public-tree privacy cleanup performed before any
history rewrite. It intentionally avoids raw private paths, personal document
filenames, checksums, or reversible mappings.

## Scope

The cleanup reviewed tracked files and current-tree references for:

- personal Windows home paths and machine-specific examples;
- private corpus paths, filenames, checksums, and expected-document metadata;
- generated evaluation reports containing per-document private corpus metadata;
- obvious populated credentials such as private keys, API keys, and tokens.

Legitimate public project attribution, licensing, and authorship were preserved.

## Current-tree cleanup

Completed local cleanup on the `privacy-cleanup` branch:

- Three human-authored technical documents were redacted in place while
  preserving their retrieval, evaluation, and RAG reasoning.
- Three active fixture/template files were replaced with coherent synthetic
  examples that preserve schemas, query identifiers, target areas, expected
  checksum relationships, and evaluation-test behavior.
- Forty-four generated historical evaluation artifacts were removed from the
  current tree because they repeated private corpus metadata at scale.
- Documentation and script defaults that referenced removed artifacts were
  updated to use generic, newly generated output names.
- Hard-coded private Windows path examples were replaced with neutral synthetic
  path examples.

## Prevention

Generated evaluation outputs and local corpus-derived reports should stay out of
Git unless they have been reviewed and synthesized. Fixture data intended for
publication must use stable synthetic identifiers and must not retain reversible
mappings to private corpus filenames or checksums.

## Remaining work

This current-tree cleanup does not remove data from historical Git objects.
Before treating the repository as fully sanitized, run a reviewed history rewrite
in a disposable clone, verify all rewritten refs, and fresh-clone the remote for
post-push validation.
