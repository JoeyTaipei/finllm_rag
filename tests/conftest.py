def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: marks tests as requiring API keys (slow)",
    )
