# semstash

> **Unlimited semantic storage for humans and AI agents**

Store anything. Find everything. Pay only for what you use.

[![PyPI](https://img.shields.io/pypi/v/semstash)](https://pypi.org/project/semstash/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

---

## What is semstash?

**semstash** is a semantic storage system that lets you store and search multimodal content (text, images, audio, video) using natural language. Built on AWS S3 + S3 Vectors + Nova Embeddings, it offers:

- 🚀 **Unlimited scale** — no vector limits, no storage caps
- 💰 **Pay-as-you-go** — zero idle costs, 90% cheaper than traditional vector DBs  
- 🎯 **Multimodal search** — one model understands text, images, audio, and video
- 🤖 **AI-ready** — works as CLI, MCP server, or REST API

## Quick Start

```bash
# Install
pip install semstash

# Initialize your stash
semstash init my-stash --region us-east-1

# Upload anything
semstash upload ./vacation-photo.jpg --tags travel,beach
semstash upload ./meeting-notes.md --tags work
semstash upload ./podcast-episode.mp3 --tags audio

# Search with natural language
semstash search "sunset over the ocean"
semstash search "discussion about quarterly goals"
```

## Three Ways to Use

### 1. CLI — For Humans

```bash
semstash upload ./document.pdf
semstash search "machine learning techniques"
semstash browse --type image/
semstash costs estimate-upload ./large-video.mp4
```

### 2. MCP Server — For AI Agents

```bash
# Start the MCP server
semstash-mcp

# Or add to Claude Desktop config:
# "semstash": {"command": "semstash-mcp"}
```

Your AI agent can now upload, search, and manage your semantic stash.

### 3. REST API — For Applications

```bash
# Start the web server
semstash-web

# Then use the API
curl -X POST localhost:8000/upload -F "file=@photo.jpg"
curl "localhost:8000/search?query=sunset+beach"
```

## Cost Transparency

semstash shows you costs **before** and **after** every operation:

```bash
$ semstash costs estimate-upload ./video.mp4

Cost Estimate: upload
Dimension: 1024 (6.1 KB per vector)

ACTIVE COSTS (one-time):
  S3 PUT request:        $0.00000500
  S3 Vectors PUT:        $0.00000119
  Nova embedding:        $0.01200000  (video: 1.0 min)
  ─────────────────────────────────────
  Subtotal:              $0.01200619

RECURRING COSTS (monthly):
  S3 content storage:    $0.00023000  (10 MB)
  S3 Vectors storage:    $0.00000036  (6.1 KB)
  ─────────────────────────────────────
  Subtotal:              $0.00023036/month
```

## Configuration

```bash
# Set embedding dimension (256, 384, 1024, or 3072)
semstash config set embeddings.dimension 1024

# Compare dimension costs
semstash costs compare-dimensions --vectors 1000000
```

| Dimension | Accuracy | Cost (1M vectors/month) |
|-----------|----------|-------------------------|
| 3072 | Maximum | $0.80 |
| 1024 | Balanced | $0.34 |
| 384 | Reduced | $0.20 |
| 256 | Minimum | $0.17 |

## Requirements

- Python 3.10+
- AWS account with credentials configured
- Nova Embeddings access (us-east-1)

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                        semstash                          │
├──────────────────────────────────────────────────────────┤
│   CLI (Typer)   │  MCP Server  │   Web API (FastAPI)    │
├──────────────────────────────────────────────────────────┤
│                    semstash.core                         │
│   storage.py │ embeddings.py │ costs.py │ config.py     │
├──────────────────────────────────────────────────────────┤
│  Amazon S3      │  S3 Vectors      │  Nova Embeddings   │
│  (content)      │  (vectors)       │  (multimodal)      │
└──────────────────────────────────────────────────────────┘
```

## License

Apache 2.0

---

**semstash** — *Unlimited semantic storage for humans and AI agents*
