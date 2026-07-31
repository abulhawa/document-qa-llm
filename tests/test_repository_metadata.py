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


def test_ci_workflow_triggers_and_permissions_match_policy() -> None:
    unit_workflow = (ROOT / ".github/workflows/tests.yml").read_text(encoding="utf-8")
    e2e_workflow = (ROOT / ".github/workflows/e2e.yml").read_text(encoding="utf-8")

    assert "pull_request:" in unit_workflow
    assert "push:\n    branches:\n      - master" in unit_workflow
    assert "workflow_dispatch:" not in unit_workflow

    assert "workflow_dispatch:" in e2e_workflow
    assert "pull_request:" not in e2e_workflow
    assert "push:" not in e2e_workflow

    for workflow in (unit_workflow, e2e_workflow):
        assert "permissions:\n  contents: read" in workflow
