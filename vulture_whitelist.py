# Vulture whitelist - false positives

# Context manager protocol requires these arguments
exc_type  # noqa
exc_val  # noqa
exc_tb  # noqa

# Pydantic field_validator classmethod requires cls
cls  # noqa

# Pydantic model_post_init requires __context parameter
__context  # noqa
