# Security Policy

## Supported versions

This project is pre-1.0 and does not yet publish supported release branches. Security fixes are applied to the current development branch when maintainers can reproduce and address them. This policy will be updated when versioned releases have a defined support window.

## Reporting a vulnerability

Please **do not open a public issue** for a suspected vulnerability. Use GitHub's **Report a vulnerability** private vulnerability reporting feature on the repository's **Security** tab. Include:

- the affected component and revision;
- reproduction steps or a minimal proof of concept;
- potential impact and affected data;
- any suggested mitigation; and
- whether disclosure is time-sensitive.

Remove real credentials, private documents, personal data, and unnecessary exploit data. If private vulnerability reporting is not enabled, open a public issue containing no vulnerability details and ask the maintainer to enable a private reporting channel.

The maintainer will aim to acknowledge the report, assess severity, coordinate a fix, and agree on disclosure timing. Response times are not currently guaranteed because this is an early-stage open-source project.

## Deployment considerations

The default Compose services expose ports on the host and disable OpenSearch's security plugin for local development. Do not expose this stack directly to an untrusted network. Operators are responsible for authentication, network isolation, document access control, secrets management, backups, and log/trace data. LLM-generated answers and retrieved citations should be treated as untrusted output and verified against source documents.
