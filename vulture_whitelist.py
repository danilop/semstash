# Vulture whitelist - false positives

# Context manager protocol requires these arguments
exc_type  # noqa
exc_val  # noqa
exc_tb  # noqa

# Pydantic field_validator classmethod requires cls
cls  # noqa
