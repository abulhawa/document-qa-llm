import logging

from utils.timing import timed_block


def test_timed_block_emits_start_end(caplog) -> None:
    logger = logging.getLogger("tests.timing")
    with caplog.at_level(logging.INFO, logger="tests.timing"):
        with timed_block(
            "unit.test",
            extra={"run_id": "test-run", "payload": {"a": 1}},
            logger=logger,
        ):
            pass

    messages = [record.message for record in caplog.records]
    assert any("START unit.test" in message for message in messages)
    assert any("END   unit.test" in message for message in messages)


def test_timed_block_handles_circular_extra(caplog) -> None:
    logger = logging.getLogger("tests.timing.circular")
    circular: dict[str, object] = {}
    circular["self"] = circular
    with caplog.at_level(logging.INFO, logger="tests.timing.circular"):
        with timed_block(
            "unit.circular",
            extra={"payload": circular},
            logger=logger,
        ):
            pass

    messages = [record.message for record in caplog.records]
    assert any("START unit.circular" in message for message in messages)
    assert any("END   unit.circular" in message for message in messages)
