from qa_pipeline.grounding import evaluate_grounding


def test_evaluate_grounding_ignores_citation_markers():
    result = evaluate_grounding(
        answer="Jane Doe led the migration [1234] successfully.",
        context_chunks=["Jane Doe led the migration successfully in 2025."],
        threshold=0.90,
    )

    assert result.score == 1.0
    assert result.is_grounded is True


def test_evaluate_grounding_handles_empty_context():
    result = evaluate_grounding(
        answer="The migration completed in 2025.",
        context_chunks=[],
        threshold=0.20,
    )

    assert result.score == 0.0
    assert result.is_grounded is False


def test_evaluate_grounding_applies_overlap_threshold():
    result = evaluate_grounding(
        answer="alpha beta gamma delta",
        context_chunks=["alpha beta theta"],
        threshold=0.60,
    )

    assert result.score == 0.5
    assert result.is_grounded is False

    relaxed = evaluate_grounding(
        answer="alpha beta gamma delta",
        context_chunks=["alpha beta theta"],
        threshold=0.40,
    )
    assert relaxed.is_grounded is True
