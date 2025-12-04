# semstash Documentation

Unlimited semantic storage for humans and AI agents.

## Overview

semstash is a Python library that combines:
- **Amazon S3** for content storage
- **Amazon S3 Vectors** for vector embeddings
- **Amazon Nova Multimodal Embeddings** for unified text/image/audio/video embeddings

## Quick Start

```python
from semstash import SemStash

# Create and initialize storage
stash = SemStash("my-bucket")
stash.init()

# Upload content
result = stash.upload("photo.jpg", tags=["vacation"])
print(f"Stored as: {result.key}")

# Query semantically
for item in stash.query("sunset on beach", top_k=5):
    print(f"{item.score:.2f} - {item.key}")

# Get content
content = stash.get("photo.jpg")
print(f"URL: {content.url}")

# Delete content
stash.delete("photo.jpg")
```

## Documentation Sections

- **[API Reference](api/)** - Auto-generated from code docstrings
- **[Guides](guides/)** - Usage tutorials and examples

## Installation

```bash
pip install semstash
```

## CLI Usage

```bash
# Initialize storage
semstash init my-bucket

# Upload files
semstash my-bucket upload photo.jpg

# Query
semstash my-bucket query "sunset on beach"

# Browse content
semstash my-bucket browse
```

## License

MIT License
