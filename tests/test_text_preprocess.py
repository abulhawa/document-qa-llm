import pytest
from core.text_preprocess import (
    PreprocessConfig,
    preprocess_document,
    _normalize_text,
    _detect_repeating_headers_footers,
    _strip_headers_footers,
    _strip_page_artifacts,
    _join_hyphenated_linebreaks,
    _should_apply_hyphenation,
    _repair_soft_wraps,
    _should_apply_softwrap,
    _mark_table_blocks,
    _clean_symbol_only_and_bullets,
    _final_whitespace_cleanup,
)


def test_normalize_text_handles_ftfy_and_unicode():
    cfg = PreprocessConfig()
    raw = "Bad encoding: cafÃ©\r\nResume\u0301"
    norm = _normalize_text(raw, cfg)
    assert norm == "Bad encoding: café\nResumé"


def test_detect_and_strip_headers_footers():
    cfg = PreprocessConfig()
    pages = [
        "HEADER\nContent A\nFOOTER\n1",
        "HEADER\nContent B\nFOOTER\n2",
        "HEADER\nContent C\nFOOTER\n3",
    ]
    headers, footers = _detect_repeating_headers_footers(pages, cfg)
    assert headers == {"HEADER"}
    assert footers == {"FOOTER"}
    stripped = _strip_headers_footers(pages[0], headers, footers)
    assert "HEADER" not in stripped
    assert "FOOTER" not in stripped
    assert "Content A" in stripped


def test_strip_page_artifacts_removes_numbers():
    cfg = PreprocessConfig()
    text = "Title\nPage 1 of 3\nSome text\n3"
    cleaned = _strip_page_artifacts(text, cfg)
    lines = cleaned.split("\n")
    assert "Page 1 of 3" not in lines
    assert "3" not in lines


@pytest.mark.parametrize(
    "strategy,expected",
    [
        ("merge", "leftright"),
        ("space", "left right"),
        ("keep", "left-\nright"),
        ("smart", "leftright"),
    ],
)
def test_join_hyphenated_linebreaks_strategies(strategy, expected):
    cfg = PreprocessConfig(hyphenation_strategy=strategy)
    text = "left-\nright"
    assert _join_hyphenated_linebreaks(text, cfg) == expected


def test_join_hyphenated_linebreaks_smart_behavior():
    cfg = PreprocessConfig(hyphenation_strategy="smart")
    text = "intro-\nduction\nstate-\nof-the-art\ncost-benefit-\nanalysis"
    result = _join_hyphenated_linebreaks(text, cfg)
    assert "introduction" in result
    assert "state-of-the-art" in result
    assert "cost-benefit analysis" in result


def test_should_apply_hyphenation():
    cfg = PreprocessConfig(apply_hyphenation_pdf_only=True)
    assert _should_apply_hyphenation("pdf", cfg)
    assert not _should_apply_hyphenation("txt", cfg)
    cfg2 = PreprocessConfig(apply_hyphenation_pdf_only=False)
    assert _should_apply_hyphenation("txt", cfg2)


def test_repair_soft_wraps_joins_and_skips():
    text = (
        "- bullet\n"
        "item\n"
        "HELLO WORLD\n"
        "next line\n"
        "Intro:\n"
        "next step\n"
        "This is a line\n"
        "continued here"
    )
    repaired = _repair_soft_wraps(text)
    lines = repaired.split("\n")
    assert lines == [
        "- bullet",
        "item",
        "HELLO WORLD",
        "next line",
        "Intro:",
        "next step",
        "This is a line continued here",
    ]


def test_should_apply_softwrap():
    cfg = PreprocessConfig(apply_softwrap_pdf_only=True)
    assert _should_apply_softwrap("pdf", cfg)
    assert not _should_apply_softwrap("docx", cfg)
    cfg2 = PreprocessConfig(apply_softwrap_pdf_only=False)
    assert _should_apply_softwrap("docx", cfg2)


def test_mark_table_blocks_pipe_and_tab_and_inconsistent():
    pipe_text = "Name | Age\nJohn | 30\n---\nJane | 25\nEnd"
    tagged_pipe = _mark_table_blocks(pipe_text)
    assert (
        tagged_pipe
        == "[TABLE]\nName | Age\nJohn | 30\n---\nJane | 25\n[/TABLE]\nEnd"
    )

    tab_text = "A\tB\tC\n1\t2\t3\n4\t5\t6"
    tagged_tab = _mark_table_blocks(tab_text)
    assert tagged_tab == "[TABLE]\nA\tB\tC\n1\t2\t3\n4\t5\t6\n[/TABLE]"

    bad_text = "Name | Age\nJohn | 30\nAlice | 30 | 5"
    tagged_bad = _mark_table_blocks(bad_text)
    assert (
        tagged_bad
        == "[TABLE]\nName | Age\nJohn | 30\n[/TABLE]\nAlice | 30 | 5"
    )


def test_clean_symbol_only_and_bullets():
    text = "[TABLE]\n•\n[/TABLE]\n•\nNext line\n•\n\n----\nWord"
    cleaned = _clean_symbol_only_and_bullets(text)
    assert cleaned == "[TABLE]\n•\n[/TABLE]\n• Next line\n\nWord"


def test_final_whitespace_cleanup():
    text = "line1 \n\n\nline2\n   \n"
    cleaned = _final_whitespace_cleanup(text)
    assert cleaned == "line1\n\nline2"


def test_preprocess_document_integration():
    cfg = PreprocessConfig()
    p1 = (
        "HEADER\n"
        "Page 1 of 3\n"
        "Intro-\n"
        "duction\n"
        "•\n"
        "Bullet text\n"
        "Name | Age\n"
        "John | 30\n"
        "---\n"
        "Jane | 25\n"
        "FOOTER\n"
        "1"
    )
    p2 = (
        "HEADER\n"
        "Page 2 of 3\n"
        "state-\n"
        "of-the-art\n"
        "Another line that\n"
        "continues here\n"
        "FOOTER\n"
        "2"
    )
    p3 = (
        "HEADER\n"
        "Page 3 of 3\n"
        "Final page\n"
        "FOOTER\n"
        "3"
    )
    full, pages = preprocess_document([p1, p2, p3], cfg, doc_type="pdf")
    assert "HEADER" not in full
    assert "FOOTER" not in full
    assert "Page 1 of 3" not in full
    assert "Page 2 of 3" not in full
    assert "Page 3 of 3" not in full
    assert "introduction" in full.lower()
    assert "state-of-the-art" in full
    assert "Another line that continues here" in pages[1]
    assert "• Bullet text" in pages[0]
    assert "[TABLE]" in full and "[/TABLE]" in full
