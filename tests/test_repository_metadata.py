from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_public_repository_files_are_present() -> None:
    expected = {
        "LICENSE",
        "CONTRIBUTING.md",
        "CODE_OF_CONDUCT.md",
        "SECURITY.md",
        "CHANGELOG.md",
        ".github/pull_request_template.md",
        ".github/ISSUE_TEMPLATE/bug_report.yml",
        ".github/ISSUE_TEMPLATE/feature_request.yml",
        ".github/ISSUE_TEMPLATE/documentation.yml",
    }

    missing = sorted(path for path in expected if not (ROOT / path).is_file())

    assert not missing, f"Missing public repository files: {missing}"


def test_example_environment_does_not_contain_an_api_key() -> None:
    assignments = {}
    for raw_line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#") and "=" in line:
            name, value = line.split("=", 1)
            assignments[name] = value

    assert "GROQ_API_KEY" in assignments
    assert assignments["GROQ_API_KEY"] == ""


def test_ci_workflows_run_for_pull_requests_with_read_only_permissions() -> None:
    for relative_path in (".github/workflows/tests.yml", ".github/workflows/e2e.yml"):
        workflow = (ROOT / relative_path).read_text(encoding="utf-8")

        assert "pull_request:" in workflow
        assert "permissions:\n  contents: read" in workflow
