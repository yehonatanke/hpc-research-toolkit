import logging

def test_runloger():
    logger = logging.getLogger(__name__)

    logger.debug("Debug message", extra={"request_id": "abc123"})
    logger.info("Application started")
    logger.warning("Something suspicious")
    logger.error("Failed to connect", exc_info=True)
    # Use .extra={} for structured extras
    logger.info("Loading model from:", extra={"path": "./models/best.pth"})
    # Multiple extras
    logger.info("Processing file:", extra={"path": "/data/train.csv", "request_id": "abc123"})
    exit()