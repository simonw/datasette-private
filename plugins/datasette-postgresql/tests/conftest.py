import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "asyncio: mark a test as an asyncio test"
    )
