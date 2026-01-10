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
