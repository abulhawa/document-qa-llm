"""Regression tests for host-to-worker document path translation."""

from worker.tasks import host_to_container_path


def test_maps_windows_path_with_spaces(monkeypatch):
    monkeypatch.setenv("DOC_PATH_MAP", "C:/=>/host-documents")

    assert host_to_container_path(r"C:\My Documents\report 2026.pdf") == (
        "/host-documents/My Documents/report 2026.pdf"
    )


def test_maps_linux_absolute_path_with_spaces(monkeypatch):
    monkeypatch.setenv("DOC_PATH_MAP", "/home/user/My Documents=>/host-documents")

    assert host_to_container_path("/home/user/My Documents/report.pdf") == (
        "/host-documents/report.pdf"
    )


def test_does_not_match_partial_path_segment(monkeypatch):
    monkeypatch.setenv("DOC_PATH_MAP", "/home/user/docs=>/host-documents")

    assert host_to_container_path("/home/user/docs-old/report.pdf") == (
        "/home/user/docs-old/report.pdf"
    )


def test_relative_path_is_unchanged(monkeypatch):
    monkeypatch.setenv("DOC_PATH_MAP", "/home/user/docs=>/host-documents")

    assert host_to_container_path("documents/report.pdf") == "documents/report.pdf"
