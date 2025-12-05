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

# Upload content to root (target is required)
result = stash.upload("photo.jpg", target="/", tags=["vacation"])
print(f"Stored at: {result.path}")  # /photo.jpg

# Upload to a folder (preserves filename)
result = stash.upload("notes.txt", target="/docs/")
print(f"Stored at: {result.path}")  # /docs/notes.txt

# Query semantically
for item in stash.query("sunset on beach", top_k=5):
    print(f"{item.score:.2f} - {item.path}")

# Query with path filter
for item in stash.query("meeting notes", path="/docs/"):
    print(f"{item.path}: {item.score:.2f}")

# Get content by path
content = stash.get("/photo.jpg")
print(f"URL: {content.url}")

# Delete content by path
stash.delete("/photo.jpg")
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

# Upload files (target path is required)
semstash my-bucket upload photo.jpg /              # Upload to root
semstash my-bucket upload notes.txt /docs/         # Upload to folder

# Query
semstash my-bucket query "sunset on beach"
semstash my-bucket query "meeting notes" --path /docs/  # Filter by path

# Browse content by path
semstash my-bucket browse /                        # Browse root
semstash my-bucket browse /docs/                   # Browse folder

# Get/delete by full path
semstash my-bucket get /photo.jpg
semstash my-bucket delete /photo.jpg
```

## License

MIT License
