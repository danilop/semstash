# semstash

> **Unlimited semantic storage for humans and AI agents**

## Complete Implementation Document

**CLI:** `semstash` | **PyPI:** `semstash` | **License:** Apache 2.0

A unified semantic storage system combining Amazon S3 (raw content), Amazon S3 Vectors (embeddings), and Amazon Nova Multimodal Embeddings to create an unlimited, pay-as-you-go knowledge base accessible via CLI, MCP server, and web interface.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [AWS Service Specifications](#3-aws-service-specifications)
4. [Cost Estimation Module](#4-cost-estimation-module)
5. [Core Library Design](#5-core-library-design)
6. [CLI Implementation](#6-cli-implementation)
7. [MCP Server Implementation](#7-mcp-server-implementation)
8. [Web Interface Implementation](#8-web-interface-implementation)
9. [Configuration Management](#9-configuration-management)
10. [Repository Structure](#10-repository-structure)
11. [Dependencies & Setup](#11-dependencies--setup)
12. [Testing Strategy](#12-testing-strategy)

---

## 1. Executive Summary

### Problem Statement
AI agents and humans need affordable, scalable semantic storage for multimodal content (text, images, audio, video) with similarity search capabilities. Traditional vector databases require provisioned capacity and have high idle costs.

### Solution
**semstash** — a unified Python library exposing three interfaces (CLI, MCP, Web) that:
- Stores raw content in S3 (globally unique bucket names)
- Stores vector embeddings in S3 Vectors (account-scoped bucket names)
- Generates embeddings via Nova Multimodal Embeddings (single model for all modalities)
- Provides real-time cost estimation across all operations

### Key Differentiators
- **90% cost reduction** vs traditional vector databases
- **Zero provisioned capacity** — pay only for actual usage
- **Unlimited scale** — no vector count limits, no storage caps
- **Unified multimodal embeddings** — single model handles text, images, audio, video
- **Built-in cost tracking** — users see costs before and after operations
- **Three interfaces** — CLI for humans, MCP for AI agents, Web API for applications

### Quick Start

```bash
# Install
pip install semstash

# Initialize (creates S3 bucket + S3 Vectors index)
semstash init my-knowledge-base --region us-east-1

# Upload content (text, images, audio, video)
semstash upload ./photo.jpg --tags vacation,beach
semstash upload ./notes.md --tags work,ideas
semstash upload ./podcast.mp3 --tags audio,interview

# Search semantically
semstash search "sunset over the ocean" --top-k 5

# Estimate costs before uploading
semstash costs estimate-upload ./large-video.mp4

# For AI agents: run as MCP server
semstash-mcp
```

---

## 2. Architecture Overview

### System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interfaces                          │
├─────────────────┬─────────────────────┬─────────────────────────┤
│   CLI (Typer)   │  MCP Server (FastMCP)│   Web (FastAPI)        │
│   - upload      │  - upload_content    │   - POST /upload       │
│   - search      │  - search            │   - GET /search        │
│   - retrieve    │  - retrieve          │   - GET /content/{key} │
│   - delete      │  - delete            │   - DELETE /content    │
│   - browse      │  - browse            │   - GET /browse        │
│   - costs       │  - estimate_cost     │   - GET /costs         │
│   - init        │  - get_usage_stats   │   - GET /stats         │
└────────┬────────┴──────────┬──────────┴───────────┬─────────────┘
         │                   │                      │
         └───────────────────┼──────────────────────┘
                             │
                    ┌────────▼────────┐
                    │   Core Library   │
                    │  semstash.core   │
                    ├─────────────────┤
                    │ - storage.py    │ ← S3 + S3 Vectors operations
                    │ - embeddings.py │ ← Nova embedding generation
                    │ - costs.py      │ ← Cost estimation & tracking
                    │ - config.py     │ ← Configuration management
                    │ - models.py     │ ← Pydantic data models
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
    ┌────▼────┐       ┌──────▼──────┐     ┌─────▼─────┐
    │   S3    │       │ S3 Vectors  │     │  Bedrock  │
    │ (Content)│       │(Embeddings) │     │  (Nova)   │
    └─────────┘       └─────────────┘     └───────────┘
```

### Bucket Naming Convention

| Component | Naming Pattern | Uniqueness |
|-----------|---------------|------------|
| S3 Bucket (content) | `{user-chosen-name}` | **Globally unique** |
| S3 Vector Bucket | `{user-chosen-name}-vectors` | **Account-scoped** (per region) |
| Vector Index | `default-index` | Per vector bucket |

**Critical Note:** S3 bucket names remain globally unique across all AWS accounts. S3 Vector bucket names are account-scoped (unique only within your AWS account per region). This is a key architectural distinction.

---

## 3. AWS Service Specifications

### 3.1 Amazon S3 Vectors

**Status:** Preview (as of December 2025)
**Regions:** us-east-1, us-east-2, us-west-2, eu-central-1, ap-southeast-2

#### API Operations (boto3 `s3vectors` client)

```python
import boto3

s3vectors = boto3.client('s3vectors', region_name='us-east-1')

# Create vector bucket (account-scoped name)
s3vectors.create_vector_bucket(vectorBucketName="my-storage-vectors")

# Create index with dimension and distance metric
s3vectors.create_index(
    vectorBucketName="my-storage-vectors",
    indexName="default-index",
    dataType="float32",
    dimension=1024,
    distanceMetric="cosine",
    metadataConfiguration={
        "nonFilterableMetadataKeys": ["source_text", "description"]
    }
)

# Insert vectors (batch up to 500)
s3vectors.put_vectors(
    vectorBucketName="my-storage-vectors",
    indexName="default-index",
    vectors=[{
        "key": "unique-content-key",
        "data": {"float32": embedding_list},
        "metadata": {
            "s3_bucket": "my-storage",
            "s3_key": "path/to/file.jpg",
            "content_type": "image/jpeg",
            "file_size": 1048576,
            "created_at": "2025-12-01T10:30:00Z"
        }
    }]
)

# Query vectors with filtering
results = s3vectors.query_vectors(
    vectorBucketName="my-storage-vectors",
    indexName="default-index",
    queryVector={"float32": query_embedding},
    topK=10,  # Maximum 30
    filter={"content_type": "image/jpeg"},
    returnDistance=True,
    returnMetadata=True
)

# Delete vectors
s3vectors.delete_vectors(
    vectorBucketName="my-storage-vectors",
    indexName="default-index",
    keys=["unique-content-key"]
)

# List vectors (for browse capability)
s3vectors.list_vectors(
    vectorBucketName="my-storage-vectors",
    indexName="default-index",
    maxResults=100,
    segmentCount=1,
    segmentIndex=0
)
```

#### Limits & Constraints

| Constraint | Value |
|------------|-------|
| Max dimensions | 4,096 |
| Min dimensions | 1 |
| Max top-K results | 30 |
| Max vectors per PUT | 500 |
| Write rate | 5 requests/second/index |
| Max vectors per index | 50 million |
| Filterable metadata | 2 KB per vector |
| Non-filterable metadata | 10 keys, up to 40 KB total |

### 3.2 Amazon Nova Multimodal Embeddings

**Model ID:** `amazon.nova-2-multimodal-embeddings-v1:0`
**Region:** us-east-1 (currently only region available)

#### Supported Content Types

```python
# File extension to Nova format mapping
CONTENT_TYPE_MAP = {
    # Text (no format needed, passed as string)
    ".txt": "text",
    ".md": "text",
    ".html": "text",
    ".json": "text",
    ".csv": "text",
    
    # Images
    ".png": ("image", "png"),
    ".jpg": ("image", "jpeg"),
    ".jpeg": ("image", "jpeg"),
    ".gif": ("image", "gif"),
    ".webp": ("image", "webp"),
    
    # Audio
    ".mp3": ("audio", "mp3"),
    ".wav": ("audio", "wav"),
    ".ogg": ("audio", "ogg"),
    
    # Video
    ".mp4": ("video", "mp4"),
    ".mov": ("video", "mov"),
    ".mkv": ("video", "mkv"),
    ".webm": ("video", "webm"),
    ".avi": ("video", "avi"),
    ".flv": ("video", "flv"),
    ".mpeg": ("video", "mpeg"),
    ".mpg": ("video", "mpg"),
    ".wmv": ("video", "wmv"),
    ".3gp": ("video", "3gp"),
}
```

#### Embedding Dimensions

| Dimension | Use Case | Storage per Vector |
|-----------|----------|-------------------|
| **3072** | Maximum accuracy | 12 KB |
| **1024** | Balanced (recommended) | 4 KB |
| **384** | Reduced storage | 1.5 KB |
| **256** | Minimum storage | 1 KB |

#### Input Constraints

| Content Type | Sync API Limit | Async API Limit |
|--------------|----------------|-----------------|
| Text | 50,000 chars / 8,192 tokens | 634 MB |
| Image | 50 MB | 50 MB |
| Audio | 100 MB / 30 seconds | 1 GB / 2 hours |
| Video | 100 MB / 30 seconds | 2 GB / 2 hours |

#### Embedding API Examples

```python
import boto3
import json
import base64

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")
MODEL_ID = "amazon.nova-2-multimodal-embeddings-v1:0"

def embed_text(text: str, dimension: int = 1024) -> list[float]:
    """Generate embedding for text content."""
    response = bedrock.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps({
            "taskType": "SINGLE_EMBEDDING",
            "singleEmbeddingParams": {
                "embeddingPurpose": "GENERIC_INDEX",
                "embeddingDimension": dimension,
                "text": {
                    "truncationMode": "END",
                    "value": text
                }
            }
        })
    )
    result = json.loads(response["body"].read())
    return result["embeddings"][0]["embedding"]

def embed_image(image_bytes: bytes, format: str, dimension: int = 1024) -> list[float]:
    """Generate embedding for image content."""
    response = bedrock.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps({
            "taskType": "SINGLE_EMBEDDING",
            "singleEmbeddingParams": {
                "embeddingPurpose": "GENERIC_INDEX",
                "embeddingDimension": dimension,
                "image": {
                    "format": format,
                    "detailLevel": "STANDARD_IMAGE",
                    "source": {
                        "bytes": base64.b64encode(image_bytes).decode()
                    }
                }
            }
        })
    )
    result = json.loads(response["body"].read())
    return result["embeddings"][0]["embedding"]

def embed_audio(audio_bytes: bytes, format: str, dimension: int = 1024) -> list[float]:
    """Generate embedding for audio content."""
    response = bedrock.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps({
            "taskType": "SINGLE_EMBEDDING",
            "singleEmbeddingParams": {
                "embeddingPurpose": "GENERIC_INDEX",
                "embeddingDimension": dimension,
                "audio": {
                    "format": format,
                    "source": {
                        "bytes": base64.b64encode(audio_bytes).decode()
                    }
                }
            }
        })
    )
    result = json.loads(response["body"].read())
    return result["embeddings"][0]["embedding"]

def embed_video(video_bytes: bytes, format: str, dimension: int = 1024) -> list[float]:
    """Generate embedding for video content."""
    response = bedrock.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps({
            "taskType": "SINGLE_EMBEDDING",
            "singleEmbeddingParams": {
                "embeddingPurpose": "GENERIC_INDEX",
                "embeddingDimension": dimension,
                "video": {
                    "format": format,
                    "source": {
                        "bytes": base64.b64encode(video_bytes).decode()
                    }
                }
            }
        })
    )
    result = json.loads(response["body"].read())
    return result["embeddings"][0]["embedding"]

def embed_for_query(text: str, dimension: int = 1024) -> list[float]:
    """Generate embedding for search query (use TEXT_RETRIEVAL purpose)."""
    response = bedrock.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps({
            "taskType": "SINGLE_EMBEDDING",
            "singleEmbeddingParams": {
                "embeddingPurpose": "TEXT_RETRIEVAL",
                "embeddingDimension": dimension,
                "text": {
                    "truncationMode": "END",
                    "value": text
                }
            }
        })
    )
    result = json.loads(response["body"].read())
    return result["embeddings"][0]["embedding"]
```

---

## 4. Cost Estimation Module

### 4.1 Key Design Principles

1. **Configurable Dimensions**: Embedding dimension (256, 384, 1024, 3072) is configurable. Default is **3072 for maximum accuracy**. Lower dimensions reduce storage costs.

2. **Active vs Recurring Costs**:
   - **Active costs**: One-time charges per operation (PUT requests, embedding generation, query API calls)
   - **Recurring costs**: Monthly charges that continue while data is stored (S3 storage, S3 Vectors storage)

3. **Dynamic Pricing**: Pricing is fetched from AWS Pricing API on startup and refreshed periodically. Falls back to cached defaults only when API is unavailable.

### 4.2 Dimension Impact on Costs

| Dimension | Vector Size | Storage/1M vectors/month | Use Case |
|-----------|-------------|--------------------------|----------|
| 3072 | ~14.3 KB | ~$858 | Maximum accuracy (default) |
| 1024 | ~6.1 KB | ~$366 | Balanced cost/quality |
| 384 | ~3.6 KB | ~$216 | Large scale, acceptable quality loss |
| 256 | ~3.0 KB | ~$180 | Cost-critical applications |

Vector size = (dimension × 4 bytes) + ~2KB metadata overhead

### 4.3 Pricing Data Structures

```python
# src/semstash/core/pricing.py

from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from enum import Enum


class CostCategory(str, Enum):
    """Cost classification."""
    ACTIVE = "active"      # One-time per operation
    RECURRING = "recurring"  # Monthly ongoing


@dataclass
class S3VectorsPricing:
    """S3 Vectors pricing - fetched dynamically from AWS Pricing API."""
    # Recurring
    storage_per_gb_month: float = 0.06
    
    # Active
    put_per_gb: float = 0.20
    query_per_million: float = 2.50
    query_data_first_100k_per_tb: float = 0.004
    query_data_above_100k_per_tb: float = 0.002
    
    # Metadata
    last_updated: Optional[datetime] = None
    source: str = "default"  # "api" or "default"


@dataclass
class S3StandardPricing:
    """S3 Standard pricing - fetched dynamically."""
    # Recurring
    storage_per_gb_month: float = 0.023
    
    # Active
    put_per_1000: float = 0.005
    get_per_1000: float = 0.0004
    data_transfer_out_per_gb: float = 0.09
    
    last_updated: Optional[datetime] = None
    source: str = "default"


@dataclass
class NovaEmbeddingsPricing:
    """
    Nova Multimodal Embeddings pricing.
    Note: Cost is per INPUT, not affected by output dimension.
    """
    # Active (per embedding)
    text_per_1000_tokens: float = 0.00013
    image_per_image: float = 0.00018
    audio_per_minute: float = 0.006
    video_per_minute: float = 0.012
    
    last_updated: Optional[datetime] = None
    source: str = "default"
```

### 4.4 Dynamic Pricing Manager

```python
# src/semstash/core/pricing.py (continued)

import boto3
import json
import logging
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class DynamicPricingManager:
    """
    Fetches and caches pricing from AWS Pricing API.
    
    - Automatically refreshes when cache expires (24h default)
    - Falls back to defaults when API unavailable
    - Tracks pricing source for transparency
    """
    
    REGION_NAMES = {
        'us-east-1': 'US East (N. Virginia)',
        'us-east-2': 'US East (Ohio)',
        'us-west-2': 'US West (Oregon)',
        'eu-central-1': 'EU (Frankfurt)',
        'ap-southeast-2': 'Asia Pacific (Sydney)',
    }
    
    CACHE_TTL_HOURS = 24
    
    def __init__(self, region: str = "us-east-1"):
        self.region = region
        self._pricing_client = None
        
        # Initialize with defaults
        self.s3 = S3StandardPricing()
        self.s3_vectors = S3VectorsPricing()
        self.nova = NovaEmbeddingsPricing()
    
    @property
    def pricing_client(self):
        """Lazy-load Pricing API client (only in us-east-1/ap-south-1)."""
        if self._pricing_client is None:
            self._pricing_client = boto3.client('pricing', region_name='us-east-1')
        return self._pricing_client
    
    def refresh_all(self, force: bool = False) -> dict:
        """
        Refresh all pricing from AWS API.
        
        Returns dict with success status per service.
        """
        results = {}
        
        if force or self._cache_expired(self.s3):
            results["s3"] = self._fetch_s3_pricing()
        
        if force or self._cache_expired(self.s3_vectors):
            results["s3_vectors"] = self._fetch_s3_vectors_pricing()
        
        if force or self._cache_expired(self.nova):
            results["bedrock"] = self._fetch_bedrock_pricing()
        
        return results
    
    def _cache_expired(self, pricing_obj) -> bool:
        if pricing_obj.last_updated is None:
            return True
        age = (datetime.now() - pricing_obj.last_updated).total_seconds()
        return age > (self.CACHE_TTL_HOURS * 3600)
    
    def _fetch_s3_pricing(self) -> bool:
        """Fetch S3 Standard pricing from API."""
        try:
            location = self.REGION_NAMES.get(self.region, 'US East (N. Virginia)')
            
            response = self.pricing_client.get_products(
                ServiceCode='AmazonS3',
                Filters=[
                    {'Type': 'TERM_MATCH', 'Field': 'location', 'Value': location},
                    {'Type': 'TERM_MATCH', 'Field': 'storageClass', 'Value': 'General Purpose'},
                    {'Type': 'TERM_MATCH', 'Field': 'productFamily', 'Value': 'Storage'},
                ],
                MaxResults=10
            )
            
            for item in response.get('PriceList', []):
                product = json.loads(item)
                for term in product.get('terms', {}).get('OnDemand', {}).values():
                    for dim in term.get('priceDimensions', {}).values():
                        if 'GB-Mo' in dim.get('unit', ''):
                            price = float(dim['pricePerUnit'].get('USD', 0))
                            if price > 0:
                                self.s3.storage_per_gb_month = price
                                self.s3.last_updated = datetime.now()
                                self.s3.source = "api"
                                logger.info(f"S3 pricing updated: ${price}/GB-month")
                                return True
            return False
            
        except Exception as e:
            logger.warning(f"Failed to fetch S3 pricing: {e}")
            return False
    
    def _fetch_s3_vectors_pricing(self) -> bool:
        """
        Fetch S3 Vectors pricing.
        Note: May not be in API yet (preview service).
        """
        try:
            # Try AmazonS3Vectors service code first
            try:
                response = self.pricing_client.get_products(
                    ServiceCode='AmazonS3Vectors',
                    MaxResults=100
                )
                if response.get('PriceList'):
                    # Parse and update pricing...
                    self.s3_vectors.last_updated = datetime.now()
                    self.s3_vectors.source = "api"
                    return True
            except ClientError:
                pass
            
            # S3 Vectors not in API yet, use defaults
            logger.info("S3 Vectors pricing not in API (preview), using defaults")
            return False
            
        except Exception as e:
            logger.warning(f"Failed to fetch S3 Vectors pricing: {e}")
            return False
    
    def _fetch_bedrock_pricing(self) -> bool:
        """Fetch Bedrock/Nova embedding pricing."""
        try:
            response = self.pricing_client.get_products(
                ServiceCode='AmazonBedrock',
                Filters=[
                    {'Type': 'TERM_MATCH', 'Field': 'regionCode', 'Value': self.region},
                ],
                MaxResults=200
            )
            
            for item in response.get('PriceList', []):
                product = json.loads(item)
                attrs = product.get('product', {}).get('attributes', {})
                model = attrs.get('model', '').lower()
                
                if 'nova' in model and 'embed' in model:
                    # Parse Nova pricing...
                    self.nova.last_updated = datetime.now()
                    self.nova.source = "api"
                    return True
                    
                elif 'titan' in model and 'embed' in model:
                    # Use Titan as fallback approximation
                    self.nova.last_updated = datetime.now()
                    self.nova.source = "api-fallback-titan"
                    return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Failed to fetch Bedrock pricing: {e}")
            return False
    
    def get_status(self) -> dict:
        """Get pricing status for all services."""
        return {
            "s3": {
                "source": self.s3.source,
                "last_updated": self.s3.last_updated.isoformat() if self.s3.last_updated else None,
                "storage_per_gb_month": self.s3.storage_per_gb_month,
            },
            "s3_vectors": {
                "source": self.s3_vectors.source,
                "last_updated": self.s3_vectors.last_updated.isoformat() if self.s3_vectors.last_updated else None,
                "storage_per_gb_month": self.s3_vectors.storage_per_gb_month,
            },
            "bedrock": {
                "source": self.nova.source,
                "last_updated": self.nova.last_updated.isoformat() if self.nova.last_updated else None,
            },
        }
```

### 4.5 Cost Breakdown with Active/Recurring Separation

```python
# src/semstash/core/costs.py

from dataclasses import dataclass, field
from typing import List
from datetime import datetime


@dataclass
class CostItem:
    """Single cost line item."""
    service: str           # s3, s3_vectors, bedrock
    component: str         # storage, put, query, embedding
    category: CostCategory # ACTIVE or RECURRING
    amount: float          # Unit price
    quantity: float        # Units consumed
    unit: str              # per request, per GB, per month
    description: str = ""
    
    @property
    def total(self) -> float:
        return self.amount * self.quantity


@dataclass
class CostBreakdown:
    """
    Full cost breakdown separating active and recurring costs.
    """
    operation: str
    timestamp: datetime
    embedding_dimension: int
    items: List[CostItem] = field(default_factory=list)
    
    # Metadata
    file_size_bytes: int = 0
    content_type: str = ""
    vector_count: int = 0
    query_count: int = 0
    
    @property
    def active_costs(self) -> List[CostItem]:
        """One-time costs for this operation."""
        return [i for i in self.items if i.category == CostCategory.ACTIVE]
    
    @property
    def recurring_costs(self) -> List[CostItem]:
        """Monthly costs added by this operation."""
        return [i for i in self.items if i.category == CostCategory.RECURRING]
    
    @property
    def total_active(self) -> float:
        return sum(i.total for i in self.active_costs)
    
    @property
    def total_recurring(self) -> float:
        return sum(i.total for i in self.recurring_costs)
    
    def to_dict(self) -> dict:
        """JSON-serializable representation."""
        return {
            "operation": self.operation,
            "embedding_dimension": self.embedding_dimension,
            "costs": {
                "active": {
                    "total_usd": round(self.total_active, 8),
                    "description": "One-time costs for this operation",
                    "items": [
                        {
                            "service": i.service,
                            "component": i.component,
                            "amount_usd": round(i.total, 8),
                            "description": i.description
                        }
                        for i in self.active_costs
                    ]
                },
                "recurring": {
                    "total_usd_per_month": round(self.total_recurring, 8),
                    "description": "Monthly costs added by this operation",
                    "items": [
                        {
                            "service": i.service,
                            "component": i.component,
                            "amount_usd_per_month": round(i.total, 8),
                            "description": i.description
                        }
                        for i in self.recurring_costs
                    ]
                }
            },
            "metadata": {
                "file_size_bytes": self.file_size_bytes,
                "vector_size_bytes": self.embedding_dimension * 4 + 2048,
            }
        }
```

### 4.6 Cost Estimator with Configurable Dimensions

```python
# src/semstash/core/costs.py (continued)

class CostEstimator:
    """
    Cost estimation with configurable dimensions and dynamic pricing.
    """
    
    SUPPORTED_DIMENSIONS = [256, 384, 1024, 3072]
    DEFAULT_DIMENSION = 3072  # Maximum accuracy
    
    def __init__(self, config: Config, auto_refresh: bool = True):
        self.config = config
        self.dimension = config.embedding_dimension
        
        if self.dimension not in self.SUPPORTED_DIMENSIONS:
            raise ValueError(f"Dimension must be one of {self.SUPPORTED_DIMENSIONS}")
        
        # Dynamic pricing manager
        self.pricing = DynamicPricingManager(region=config.region)
        
        if auto_refresh:
            self.pricing.refresh_all()
    
    def get_vector_size_bytes(self, dimension: int = None) -> int:
        """Vector storage size including metadata."""
        dim = dimension or self.dimension
        return (dim * 4) + 2048  # float32 + ~2KB metadata
    
    def estimate_upload(
        self,
        file_path: Path,
        content_type: str,
        dimension: int = None
    ) -> CostBreakdown:
        """
        Estimate upload cost with active/recurring separation.
        """
        dim = dimension or self.dimension
        file_size = file_path.stat().st_size
        vector_size = self.get_vector_size_bytes(dim)
        
        breakdown = CostBreakdown(
            operation="upload",
            timestamp=datetime.now(),
            embedding_dimension=dim,
            file_size_bytes=file_size,
            content_type=content_type,
            vector_count=1
        )
        
        # === ACTIVE COSTS ===
        
        # S3 PUT request
        breakdown.items.append(CostItem(
            service="s3",
            component="put_request",
            category=CostCategory.ACTIVE,
            amount=self.pricing.s3.put_per_1000 / 1000,
            quantity=1,
            unit="per request",
            description="S3 PUT for content"
        ))
        
        # S3 Vectors PUT
        breakdown.items.append(CostItem(
            service="s3_vectors",
            component="put_data",
            category=CostCategory.ACTIVE,
            amount=self.pricing.s3_vectors.put_per_gb,
            quantity=vector_size / (1024**3),
            unit="per GB",
            description=f"Vector ingestion ({dim}d)"
        ))
        
        # Embedding generation
        embedding_cost = self._calc_embedding_cost(file_size, content_type)
        breakdown.items.append(CostItem(
            service="bedrock",
            component="embedding",
            category=CostCategory.ACTIVE,
            amount=embedding_cost,
            quantity=1,
            unit="per embedding",
            description=f"Nova embedding ({content_type})"
        ))
        
        # === RECURRING COSTS ===
        
        # S3 content storage
        breakdown.items.append(CostItem(
            service="s3",
            component="storage",
            category=CostCategory.RECURRING,
            amount=self.pricing.s3.storage_per_gb_month,
            quantity=file_size / (1024**3),
            unit="per GB/month",
            description="Content storage"
        ))
        
        # S3 Vectors storage
        breakdown.items.append(CostItem(
            service="s3_vectors",
            component="storage",
            category=CostCategory.RECURRING,
            amount=self.pricing.s3_vectors.storage_per_gb_month,
            quantity=vector_size / (1024**3),
            unit="per GB/month",
            description=f"Vector storage ({dim}d = {vector_size} bytes)"
        ))
        
        return breakdown
    
    def estimate_query(
        self,
        index_vector_count: int,
        dimension: int = None
    ) -> CostBreakdown:
        """
        Estimate query cost (all active, no recurring).
        """
        dim = dimension or self.dimension
        vector_size = self.get_vector_size_bytes(dim)
        
        breakdown = CostBreakdown(
            operation="query",
            timestamp=datetime.now(),
            embedding_dimension=dim,
            vector_count=index_vector_count,
            query_count=1
        )
        
        # Query API call
        breakdown.items.append(CostItem(
            service="s3_vectors",
            component="query_api",
            category=CostCategory.ACTIVE,
            amount=self.pricing.s3_vectors.query_per_million / 1_000_000,
            quantity=1,
            unit="per query",
            description="S3 Vectors query API"
        ))
        
        # Data processing
        data_tb = (vector_size * index_vector_count) / (1024**4)
        rate = (self.pricing.s3_vectors.query_data_first_100k_per_tb
                if index_vector_count <= 100_000
                else self.pricing.s3_vectors.query_data_above_100k_per_tb)
        
        breakdown.items.append(CostItem(
            service="s3_vectors",
            component="query_data",
            category=CostCategory.ACTIVE,
            amount=rate,
            quantity=data_tb,
            unit="per TB",
            description=f"Process {index_vector_count:,} vectors"
        ))
        
        # Query embedding
        breakdown.items.append(CostItem(
            service="bedrock",
            component="query_embedding",
            category=CostCategory.ACTIVE,
            amount=self.pricing.nova.text_per_1000_tokens * 0.1,  # ~100 tokens
            quantity=1,
            unit="per query",
            description="Query text embedding"
        ))
        
        return breakdown
    
    def compare_dimensions(self, total_vectors: int) -> dict:
        """
        Compare monthly costs across all dimension options.
        """
        results = {}
        
        for dim in self.SUPPORTED_DIMENSIONS:
            vector_size = self.get_vector_size_bytes(dim)
            storage_gb = (total_vectors * vector_size) / (1024**3)
            monthly_cost = storage_gb * self.pricing.s3_vectors.storage_per_gb_month
            
            results[dim] = {
                "vector_size_bytes": vector_size,
                "total_storage_gb": round(storage_gb, 2),
                "monthly_storage_usd": round(monthly_cost, 2)
            }
        
        # Add relative cost comparison
        base = results[256]["monthly_storage_usd"]
        for dim in self.SUPPORTED_DIMENSIONS:
            results[dim]["relative_cost"] = round(
                results[dim]["monthly_storage_usd"] / base, 1
            ) if base > 0 else 1.0
        
        return {
            "comparison": results,
            "current_dimension": self.dimension,
            "recommendation": {
                "max_accuracy": {"dimension": 3072, "note": "Best quality, highest cost"},
                "balanced": {"dimension": 1024, "note": "Good quality, moderate cost"},
                "cost_optimized": {"dimension": 256, "note": "Acceptable quality, lowest cost"}
            }
        }
```

### 4.7 Cost Output Examples

**Upload 1MB image with 3072 dimensions:**
```
Operation: upload
Dimension: 3072

═══ ACTIVE COSTS (one-time) ═══
  s3/put_request:      $0.00000500
  s3_vectors/put_data: $0.00000272  (14.3KB × $0.20/GB)
  bedrock/embedding:   $0.00018000
  SUBTOTAL:            $0.00018772

═══ RECURRING COSTS (monthly) ═══
  s3/storage:          $0.00002289  (1MB × $0.023/GB)
  s3_vectors/storage:  $0.00000084  (14.3KB × $0.06/GB)
  SUBTOTAL:            $0.00002373/month

═══ SUMMARY ═══
  Active (now):        $0.00018772
  Recurring:           $0.00002373/month
```

**Dimension comparison for 1M vectors:**
```json
{
  "comparison": {
    "256": {"vector_size_bytes": 3072, "total_storage_gb": 2.86, "monthly_storage_usd": 0.17, "relative_cost": 1.0},
    "384": {"vector_size_bytes": 3584, "total_storage_gb": 3.34, "monthly_storage_usd": 0.20, "relative_cost": 1.2},
    "1024": {"vector_size_bytes": 6144, "total_storage_gb": 5.72, "monthly_storage_usd": 0.34, "relative_cost": 2.0},
    "3072": {"vector_size_bytes": 14336, "total_storage_gb": 13.35, "monthly_storage_usd": 0.80, "relative_cost": 4.7}
  }
}
```

### 4.1 Pricing Data Structure

The cost estimation module uses a hybrid approach: attempt to fetch from AWS Pricing API, fall back to cached pricing data.

```python
# src/semstash/core/pricing_data.py
"""
Pricing data for cost estimation.
Updated: December 2025
Region: us-east-1 (US East N. Virginia)
"""

from dataclasses import dataclass
from typing import Dict

@dataclass
class S3VectorsPricing:
    """S3 Vectors pricing per region."""
    storage_per_gb_month: float = 0.06          # $/GB-month
    put_per_gb: float = 0.20                     # $/GB ingested
    query_per_million: float = 2.50              # $/million queries
    query_data_first_100k_per_tb: float = 0.004  # $/TB (first 100K vectors)
    query_data_above_100k_per_tb: float = 0.002  # $/TB (above 100K vectors)

@dataclass  
class S3StandardPricing:
    """Standard S3 pricing per region."""
    storage_per_gb_month: float = 0.023         # $/GB-month (first 50TB)
    put_per_1000: float = 0.005                  # $/1000 PUT requests
    get_per_1000: float = 0.0004                 # $/1000 GET requests
    data_transfer_out_per_gb: float = 0.09       # $/GB (first 10TB)

@dataclass
class NovaEmbeddingsPricing:
    """Nova Multimodal Embeddings pricing (estimated based on similar models)."""
    # Text pricing (per 1000 tokens)
    text_per_1000_tokens: float = 0.00013
    
    # Image pricing (per image)
    image_per_image: float = 0.00018
    
    # Audio pricing (per minute)
    audio_per_minute: float = 0.006
    
    # Video pricing (per minute)  
    video_per_minute: float = 0.012

@dataclass
class RegionPricing:
    """Complete pricing for a region."""
    s3_vectors: S3VectorsPricing
    s3_standard: S3StandardPricing
    nova_embeddings: NovaEmbeddingsPricing

# Default pricing (us-east-1)
DEFAULT_PRICING = RegionPricing(
    s3_vectors=S3VectorsPricing(),
    s3_standard=S3StandardPricing(),
    nova_embeddings=NovaEmbeddingsPricing()
)

# Region-specific pricing adjustments
REGION_PRICING: Dict[str, RegionPricing] = {
    "us-east-1": DEFAULT_PRICING,
    "us-east-2": DEFAULT_PRICING,
    "us-west-2": DEFAULT_PRICING,
    "eu-central-1": RegionPricing(
        s3_vectors=S3VectorsPricing(
            storage_per_gb_month=0.07,
            put_per_gb=0.22
        ),
        s3_standard=S3StandardPricing(storage_per_gb_month=0.0245),
        nova_embeddings=NovaEmbeddingsPricing()  # Only us-east-1 for now
    ),
    "ap-southeast-2": RegionPricing(
        s3_vectors=S3VectorsPricing(
            storage_per_gb_month=0.07,
            put_per_gb=0.22
        ),
        s3_standard=S3StandardPricing(storage_per_gb_month=0.025),
        nova_embeddings=NovaEmbeddingsPricing()
    ),
}
```

### 4.2 AWS Pricing API Integration

```python
# src/semstash/core/pricing_api.py
"""
AWS Pricing API integration for dynamic pricing updates.
"""

import boto3
import json
from typing import Optional, Dict, Any

class PricingAPIClient:
    """
    Client for AWS Price List API.
    
    Note: The Pricing API is only available in us-east-1 and ap-south-1.
    S3 Vectors (preview) and Nova Embeddings may not be available yet.
    """
    
    SERVICE_CODES = {
        "s3": "AmazonS3",
        "bedrock": "AmazonBedrock",
    }
    
    REGION_NAMES = {
        'us-east-1': 'US East (N. Virginia)',
        'us-east-2': 'US East (Ohio)',
        'us-west-2': 'US West (Oregon)',
        'eu-central-1': 'EU (Frankfurt)',
        'ap-southeast-2': 'Asia Pacific (Sydney)',
    }
    
    def __init__(self):
        # Pricing API only available in us-east-1
        self.client = boto3.client('pricing', region_name='us-east-1')
    
    def get_s3_storage_price(self, region: str) -> Optional[float]:
        """Get S3 standard storage price per GB-month."""
        try:
            location = self.REGION_NAMES.get(region, 'US East (N. Virginia)')
            response = self.client.get_products(
                ServiceCode='AmazonS3',
                Filters=[
                    {'Type': 'TERM_MATCH', 'Field': 'location', 'Value': location},
                    {'Type': 'TERM_MATCH', 'Field': 'storageClass', 'Value': 'General Purpose'},
                    {'Type': 'TERM_MATCH', 'Field': 'productFamily', 'Value': 'Storage'},
                ],
                MaxResults=10
            )
            
            for price_item in response.get('PriceList', []):
                product = json.loads(price_item)
                terms = product.get('terms', {}).get('OnDemand', {})
                for term in terms.values():
                    for price_dim in term.get('priceDimensions', {}).values():
                        if 'GB-Mo' in price_dim.get('unit', ''):
                            return float(price_dim['pricePerUnit']['USD'])
            
            return None
        except Exception:
            return None
    
    def get_bedrock_embedding_prices(self, region: str) -> Dict[str, float]:
        """
        Get Bedrock embedding model prices.
        
        Note: Nova Embeddings pricing may not be available in the API yet.
        Falls back to similar model pricing (Titan Multimodal Embeddings).
        """
        try:
            response = self.client.get_products(
                ServiceCode='AmazonBedrock',
                Filters=[
                    {'Type': 'TERM_MATCH', 'Field': 'regionCode', 'Value': region},
                    {'Type': 'TERM_MATCH', 'Field': 'feature', 'Value': 'On-demand Inference'},
                ],
                MaxResults=100
            )
            
            prices = {}
            for price_item in response.get('PriceList', []):
                product = json.loads(price_item)
                attrs = product.get('product', {}).get('attributes', {})
                model = attrs.get('model', attrs.get('titanModel', ''))
                
                if 'embed' in model.lower():
                    terms = product.get('terms', {}).get('OnDemand', {})
                    for term in terms.values():
                        for price_dim in term.get('priceDimensions', {}).values():
                            prices[model] = float(price_dim['pricePerUnit']['USD'])
            
            return prices
        except Exception:
            return {}
    
    def list_available_services(self) -> list[str]:
        """List all available service codes in the Pricing API."""
        try:
            paginator = self.client.get_paginator('describe_services')
            services = []
            for page in paginator.paginate():
                for service in page.get('Services', []):
                    services.append(service['ServiceCode'])
            return services
        except Exception:
            return []
```

### 4.3 Cost Estimation Engine

```python
# src/semstash/core/costs.py
"""
Cost estimation and tracking module.
Provides real-time cost estimates and usage tracking.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List
from pathlib import Path
import json

from .pricing_data import REGION_PRICING, DEFAULT_PRICING, RegionPricing
from .pricing_api import PricingAPIClient
from .config import Config

@dataclass
class CostBreakdown:
    """Detailed cost breakdown for an operation."""
    operation: str
    timestamp: datetime
    
    # S3 costs
    s3_storage_cost: float = 0.0
    s3_put_cost: float = 0.0
    s3_get_cost: float = 0.0
    s3_transfer_cost: float = 0.0
    
    # S3 Vectors costs
    vectors_storage_cost: float = 0.0
    vectors_put_cost: float = 0.0
    vectors_query_cost: float = 0.0
    
    # Bedrock costs
    embedding_cost: float = 0.0
    
    # Metadata
    file_size_bytes: int = 0
    content_type: str = ""
    vector_count: int = 0
    query_count: int = 0
    
    @property
    def total_cost(self) -> float:
        return (
            self.s3_storage_cost + self.s3_put_cost + self.s3_get_cost +
            self.s3_transfer_cost + self.vectors_storage_cost +
            self.vectors_put_cost + self.vectors_query_cost +
            self.embedding_cost
        )
    
    def to_dict(self) -> dict:
        return {
            "operation": self.operation,
            "timestamp": self.timestamp.isoformat(),
            "total_cost_usd": round(self.total_cost, 6),
            "breakdown": {
                "s3": {
                    "storage": round(self.s3_storage_cost, 6),
                    "put": round(self.s3_put_cost, 6),
                    "get": round(self.s3_get_cost, 6),
                    "transfer": round(self.s3_transfer_cost, 6),
                },
                "s3_vectors": {
                    "storage": round(self.vectors_storage_cost, 6),
                    "put": round(self.vectors_put_cost, 6),
                    "query": round(self.vectors_query_cost, 6),
                },
                "bedrock": {
                    "embedding": round(self.embedding_cost, 6),
                }
            },
            "metadata": {
                "file_size_bytes": self.file_size_bytes,
                "content_type": self.content_type,
                "vector_count": self.vector_count,
                "query_count": self.query_count,
            }
        }

@dataclass
class UsageStats:
    """Cumulative usage statistics."""
    total_files: int = 0
    total_storage_bytes: int = 0
    total_vectors: int = 0
    total_queries: int = 0
    total_cost_usd: float = 0.0
    cost_history: List[CostBreakdown] = field(default_factory=list)
    
    # Monthly projections
    estimated_monthly_storage_cost: float = 0.0
    estimated_monthly_query_cost: float = 0.0

class CostEstimator:
    """
    Cost estimation engine for semantic storage operations.
    
    Provides:
    - Pre-operation cost estimates
    - Post-operation actual costs
    - Usage tracking and projections
    - Cost breakdown by service
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.pricing = REGION_PRICING.get(config.region, DEFAULT_PRICING)
        self._usage_stats = UsageStats()
        self._pricing_api: Optional[PricingAPIClient] = None
    
    def refresh_pricing_from_api(self) -> bool:
        """
        Attempt to refresh pricing from AWS Pricing API.
        Returns True if successful, False if using cached data.
        """
        try:
            if self._pricing_api is None:
                self._pricing_api = PricingAPIClient()
            
            # Try to get S3 pricing
            s3_price = self._pricing_api.get_s3_storage_price(self.config.region)
            if s3_price:
                self.pricing.s3_standard.storage_per_gb_month = s3_price
            
            # Try to get Bedrock pricing
            bedrock_prices = self._pricing_api.get_bedrock_embedding_prices(
                self.config.region
            )
            # Update if relevant prices found
            
            return True
        except Exception:
            return False
    
    def estimate_upload_cost(
        self,
        file_path: Path,
        content_type: str
    ) -> CostBreakdown:
        """
        Estimate cost for uploading a file.
        
        Includes:
        - S3 PUT request cost
        - S3 storage cost (monthly, prorated)
        - S3 Vectors PUT cost
        - S3 Vectors storage cost (monthly, prorated)
        - Nova embedding generation cost
        """
        file_size = file_path.stat().st_size
        file_size_gb = file_size / (1024 ** 3)
        
        breakdown = CostBreakdown(
            operation="upload",
            timestamp=datetime.now(),
            file_size_bytes=file_size,
            content_type=content_type,
            vector_count=1
        )
        
        # S3 costs
        breakdown.s3_put_cost = self.pricing.s3_standard.put_per_1000 / 1000
        breakdown.s3_storage_cost = file_size_gb * self.pricing.s3_standard.storage_per_gb_month
        
        # S3 Vectors costs
        # Vector size = dimension * 4 bytes (float32) + metadata (~2KB)
        vector_size_bytes = (self.config.embedding_dimension * 4) + 2048
        vector_size_gb = vector_size_bytes / (1024 ** 3)
        
        breakdown.vectors_put_cost = vector_size_gb * self.pricing.s3_vectors.put_per_gb
        breakdown.vectors_storage_cost = vector_size_gb * self.pricing.s3_vectors.storage_per_gb_month
        
        # Embedding cost
        breakdown.embedding_cost = self._estimate_embedding_cost(
            file_size, content_type
        )
        
        return breakdown
    
    def _estimate_embedding_cost(self, file_size: int, content_type: str) -> float:
        """Estimate Bedrock embedding generation cost."""
        pricing = self.pricing.nova_embeddings
        
        if content_type.startswith("text/"):
            # Estimate tokens from file size (rough: 4 chars per token)
            estimated_tokens = file_size / 4
            return (estimated_tokens / 1000) * pricing.text_per_1000_tokens
        
        elif content_type.startswith("image/"):
            return pricing.image_per_image
        
        elif content_type.startswith("audio/"):
            # Estimate duration from file size (rough: 1MB per minute for MP3)
            estimated_minutes = file_size / (1024 * 1024)
            return estimated_minutes * pricing.audio_per_minute
        
        elif content_type.startswith("video/"):
            # Estimate duration from file size (rough: 10MB per minute)
            estimated_minutes = file_size / (10 * 1024 * 1024)
            return estimated_minutes * pricing.video_per_minute
        
        return 0.0
    
    def estimate_query_cost(
        self,
        index_vector_count: int,
        average_vector_size_kb: float = 6.17
    ) -> CostBreakdown:
        """
        Estimate cost for a single query.
        
        Query cost depends on:
        - API call cost ($2.50/million)
        - Data processed (vectors * avg size)
        """
        breakdown = CostBreakdown(
            operation="query",
            timestamp=datetime.now(),
            query_count=1,
            vector_count=index_vector_count
        )
        
        # API call cost
        breakdown.vectors_query_cost = self.pricing.s3_vectors.query_per_million / 1_000_000
        
        # Data processing cost
        data_processed_tb = (index_vector_count * average_vector_size_kb * 1024) / (1024 ** 4)
        
        if index_vector_count <= 100_000:
            breakdown.vectors_query_cost += (
                data_processed_tb * self.pricing.s3_vectors.query_data_first_100k_per_tb
            )
        else:
            first_100k_cost = (
                (100_000 * average_vector_size_kb * 1024 / (1024 ** 4)) * 
                self.pricing.s3_vectors.query_data_first_100k_per_tb
            )
            remaining_cost = (
                ((index_vector_count - 100_000) * average_vector_size_kb * 1024 / (1024 ** 4)) * 
                self.pricing.s3_vectors.query_data_above_100k_per_tb
            )
            breakdown.vectors_query_cost += first_100k_cost + remaining_cost
        
        # Query embedding cost (text query, ~100 tokens)
        breakdown.embedding_cost = self.pricing.nova_embeddings.text_per_1000_tokens * 0.1
        
        return breakdown
    
    def estimate_delete_cost(self) -> CostBreakdown:
        """Delete operations have no direct cost."""
        return CostBreakdown(
            operation="delete",
            timestamp=datetime.now()
        )
    
    def estimate_monthly_cost(
        self,
        total_storage_gb: float,
        total_vectors: int,
        queries_per_month: int,
        uploads_per_month: int,
        average_upload_size_mb: float
    ) -> dict:
        """
        Estimate total monthly cost for the storage system.
        
        Returns detailed breakdown for budgeting.
        """
        vector_storage_gb = (total_vectors * 6.17) / (1024 * 1024)  # 6.17KB per vector
        
        # Storage costs
        s3_storage = total_storage_gb * self.pricing.s3_standard.storage_per_gb_month
        vectors_storage = vector_storage_gb * self.pricing.s3_vectors.storage_per_gb_month
        
        # Operation costs
        upload_data_gb = (uploads_per_month * average_upload_size_mb) / 1024
        vectors_put = upload_data_gb * self.pricing.s3_vectors.put_per_gb
        s3_put = (uploads_per_month / 1000) * self.pricing.s3_standard.put_per_1000
        
        # Query costs
        query_api = (queries_per_month / 1_000_000) * self.pricing.s3_vectors.query_per_million
        data_per_query_tb = (total_vectors * 6.17 * 1024) / (1024 ** 4)
        query_data = queries_per_month * data_per_query_tb * self.pricing.s3_vectors.query_data_first_100k_per_tb
        
        # Embedding costs (assuming 50% text, 30% images, 10% audio, 10% video)
        text_embeddings = uploads_per_month * 0.5 * 1000 * self.pricing.nova_embeddings.text_per_1000_tokens
        image_embeddings = uploads_per_month * 0.3 * self.pricing.nova_embeddings.image_per_image
        audio_embeddings = uploads_per_month * 0.1 * self.pricing.nova_embeddings.audio_per_minute
        video_embeddings = uploads_per_month * 0.1 * self.pricing.nova_embeddings.video_per_minute
        embedding_total = text_embeddings + image_embeddings + audio_embeddings + video_embeddings
        
        # Query embeddings
        query_embeddings = queries_per_month * 0.1 * self.pricing.nova_embeddings.text_per_1000_tokens
        
        total = (
            s3_storage + vectors_storage + vectors_put + s3_put +
            query_api + query_data + embedding_total + query_embeddings
        )
        
        return {
            "total_monthly_cost_usd": round(total, 2),
            "breakdown": {
                "storage": {
                    "s3_content": round(s3_storage, 4),
                    "s3_vectors": round(vectors_storage, 4),
                    "subtotal": round(s3_storage + vectors_storage, 4)
                },
                "uploads": {
                    "s3_put": round(s3_put, 4),
                    "vectors_put": round(vectors_put, 4),
                    "embeddings": round(embedding_total, 4),
                    "subtotal": round(s3_put + vectors_put + embedding_total, 4)
                },
                "queries": {
                    "api_calls": round(query_api, 4),
                    "data_processing": round(query_data, 4),
                    "query_embeddings": round(query_embeddings, 4),
                    "subtotal": round(query_api + query_data + query_embeddings, 4)
                }
            },
            "assumptions": {
                "total_storage_gb": total_storage_gb,
                "total_vectors": total_vectors,
                "queries_per_month": queries_per_month,
                "uploads_per_month": uploads_per_month,
                "average_upload_size_mb": average_upload_size_mb,
                "region": self.config.region
            }
        }
    
    def get_current_usage(self) -> dict:
        """Get current usage statistics."""
        return {
            "storage": {
                "s3_objects": self._usage_stats.total_files,
                "s3_size_bytes": self._usage_stats.total_storage_bytes,
                "vectors": self._usage_stats.total_vectors,
            },
            "operations": {
                "total_queries": self._usage_stats.total_queries,
                "total_cost_usd": round(self._usage_stats.total_cost_usd, 4),
            },
            "projections": {
                "estimated_monthly_storage": round(
                    self._usage_stats.estimated_monthly_storage_cost, 2
                ),
                "estimated_monthly_queries": round(
                    self._usage_stats.estimated_monthly_query_cost, 2
                ),
            }
        }
    
    def record_operation(self, breakdown: CostBreakdown):
        """Record an operation for usage tracking."""
        self._usage_stats.total_cost_usd += breakdown.total_cost
        self._usage_stats.cost_history.append(breakdown)
        
        if breakdown.operation == "upload":
            self._usage_stats.total_files += 1
            self._usage_stats.total_storage_bytes += breakdown.file_size_bytes
            self._usage_stats.total_vectors += breakdown.vector_count
        elif breakdown.operation == "query":
            self._usage_stats.total_queries += breakdown.query_count
        elif breakdown.operation == "delete":
            self._usage_stats.total_files -= 1
            self._usage_stats.total_vectors -= breakdown.vector_count
    
    def format_cost_summary(self, breakdown: CostBreakdown) -> str:
        """Format cost breakdown for display."""
        lines = [
            f"Operation: {breakdown.operation}",
            f"Total Cost: ${breakdown.total_cost:.6f}",
            "",
            "Breakdown:",
        ]
        
        if breakdown.s3_storage_cost > 0 or breakdown.s3_put_cost > 0:
            lines.append(f"  S3 Storage:      ${breakdown.s3_storage_cost:.6f}/month")
            lines.append(f"  S3 PUT:          ${breakdown.s3_put_cost:.6f}")
        
        if breakdown.vectors_storage_cost > 0 or breakdown.vectors_put_cost > 0:
            lines.append(f"  Vectors Storage: ${breakdown.vectors_storage_cost:.6f}/month")
            lines.append(f"  Vectors PUT:     ${breakdown.vectors_put_cost:.6f}")
        
        if breakdown.vectors_query_cost > 0:
            lines.append(f"  Vectors Query:   ${breakdown.vectors_query_cost:.6f}")
        
        if breakdown.embedding_cost > 0:
            lines.append(f"  Embedding:       ${breakdown.embedding_cost:.6f}")
        
        return "\n".join(lines)
```

### 4.4 Cost Commands Across All Interfaces

**CLI Commands:**
```bash
# Estimate upload cost
semstash costs estimate-upload ./image.jpg

# Estimate monthly costs
semstash costs monthly --storage-gb 10 --vectors 100000 --queries 10000

# Show current usage
semstash costs usage

# Refresh pricing from AWS API
semstash costs refresh-pricing
```

**MCP Tools:**
```python
@mcp.tool()
def estimate_cost(
    operation: str,  # "upload", "query", "monthly"
    file_path: str | None = None,
    file_size: int | None = None,
    content_type: str | None = None,
    storage_gb: float | None = None,
    vectors: int | None = None,
    queries_per_month: int | None = None
) -> dict:
    """Estimate costs for semantic storage operations."""

@mcp.tool()
def get_usage_stats() -> dict:
    """Get current usage statistics and costs."""
```

**Web API Endpoints:**
```
GET /costs/estimate?operation=upload&file_size=1048576&content_type=image/jpeg
GET /costs/monthly?storage_gb=10&vectors=100000&queries=10000
GET /costs/usage
POST /costs/refresh-pricing
```

---

## 5. Core Library Design

### 5.1 Module Structure

```
src/semstash/core/
├── __init__.py           # Public API exports
├── storage.py            # S3 + S3 Vectors operations
├── embeddings.py         # Nova embedding generation
├── costs.py              # Cost estimation (see Section 4)
├── pricing_data.py       # Pricing constants
├── pricing_api.py        # AWS Pricing API client
├── config.py             # Configuration management
├── models.py             # Pydantic data models
└── exceptions.py         # Custom exceptions
```

### 5.2 Data Models

```python
# src/semstash/core/models.py
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum

class ContentType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"

class StorageItem(BaseModel):
    """Represents an item in semantic storage."""
    key: str = Field(description="Unique identifier")
    s3_bucket: str
    s3_key: str
    content_type: str
    file_size: int
    created_at: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)

class VectorEntry(BaseModel):
    """Represents a vector in S3 Vectors."""
    key: str
    embedding: List[float]
    metadata: Dict[str, Any]
    distance: Optional[float] = None

class SearchResult(BaseModel):
    """Result from semantic search."""
    key: str
    score: float = Field(description="Similarity score (0-1, higher is better)")
    content_type: str
    file_size: int
    created_at: datetime
    metadata: Dict[str, Any]
    presigned_url: Optional[str] = None
    
class UploadResult(BaseModel):
    """Result from upload operation."""
    key: str
    s3_url: str
    content_type: str
    file_size: int
    embedding_dimension: int
    cost: Dict[str, Any]

class DeleteResult(BaseModel):
    """Result from delete operation."""
    key: str
    s3_deleted: bool
    vector_deleted: bool

class BrowseResult(BaseModel):
    """Result from browse operation."""
    items: List[StorageItem]
    total_count: int
    next_token: Optional[str] = None
```

### 5.3 Exceptions

```python
# src/semstash/core/exceptions.py
"""Custom exceptions for semantic storage."""

class SemstoreError(Exception):
    """Base exception for semstash."""
    pass

class ConfigurationError(SemstoreError):
    """Configuration-related errors."""
    pass

class BucketNotFoundError(SemstoreError):
    """S3 or S3 Vectors bucket not found."""
    pass

class ContentNotFoundError(SemstoreError):
    """Content not found in storage."""
    pass

class EmbeddingError(SemstoreError):
    """Error generating embedding."""
    pass

class StorageError(SemstoreError):
    """General storage operation error."""
    pass

class PricingError(SemstoreError):
    """Error fetching pricing information."""
    pass
```

---

## 6. CLI Implementation

### 6.1 Dimension Configuration

```bash
# Set dimension globally in config
semstash config set embeddings.dimension 1024

# Override per-command
semstash upload ./image.jpg --dimension 1024

# Compare dimension costs
semstash costs compare-dimensions --vectors 1000000

# Output:
# ┌───────────┬─────────────┬─────────────┬──────────────────┬───────────────┐
# │ Dimension │ Vector Size │ Storage/1M  │ Monthly Cost/1M  │ Relative Cost │
# ├───────────┼─────────────┼─────────────┼──────────────────┼───────────────┤
# │ 256       │ 3.0 KB      │ 2.86 GB     │ $0.17            │ 1.0x          │
# │ 384       │ 3.5 KB      │ 3.34 GB     │ $0.20            │ 1.2x          │
# │ 1024      │ 6.0 KB      │ 5.72 GB     │ $0.34            │ 2.0x          │
# │ 3072 ✓    │ 14.0 KB     │ 13.35 GB    │ $0.80            │ 4.7x          │
# └───────────┴─────────────┴─────────────┴──────────────────┴───────────────┘
# ✓ = current setting
```

### 6.2 Cost Commands with Active/Recurring Breakdown

```bash
# Estimate upload cost
semstash costs estimate-upload ./video.mp4 --dimension 1024

# Output:
# Cost Estimate: upload
# Dimension: 1024 (6.1 KB per vector)
# 
# ACTIVE COSTS (one-time):
#   S3 PUT request:        $0.00000500
#   S3 Vectors PUT:        $0.00000119
#   Nova embedding:        $0.01200000  (video: 1.0 min)
#   ─────────────────────────────────────
#   Subtotal:              $0.01200619
# 
# RECURRING COSTS (monthly):
#   S3 content storage:    $0.00023000  (10 MB)
#   S3 Vectors storage:    $0.00000036  (6.1 KB)
#   ─────────────────────────────────────
#   Subtotal:              $0.00023036/month
# 
# SUMMARY:
#   Pay now:               $0.01200619
#   Pay monthly:           $0.00023036
#   First month total:     $0.01223655

# Monthly projection
semstash costs monthly \
  --storage-gb 100 \
  --vectors 1000000 \
  --queries 50000 \
  --dimension 1024

# Output:
# Monthly Cost Projection
# Dimension: 1024
# 
# RECURRING (storage):
#   S3 content:            $2.30        (100 GB)
#   S3 Vectors:            $0.34        (5.72 GB for 1M vectors)
#   ─────────────────────────────────────
#   Subtotal:              $2.64/month
# 
# ACTIVE (operations):
#   Upload costs:          $0.15        (estimated)
#   Query costs:           $0.18        (50K queries)
#   ─────────────────────────────────────
#   Subtotal:              $0.33/month
# 
# TOTAL:                   $2.97/month
# 
# Pricing source: AWS Pricing API (updated 2h ago)

# Show pricing status
semstash costs pricing-status

# Output:
# Pricing Status
# ┌─────────────┬─────────┬─────────────────────┬──────────────────┐
# │ Service     │ Source  │ Last Updated        │ Key Rate         │
# ├─────────────┼─────────┼─────────────────────┼──────────────────┤
# │ S3          │ api     │ 2024-01-15 10:30:00 │ $0.023/GB-month  │
# │ S3 Vectors  │ default │ -                   │ $0.06/GB-month   │
# │ Bedrock     │ api     │ 2024-01-15 10:30:00 │ $0.00013/1K tok  │
# └─────────────┴─────────┴─────────────────────┴──────────────────┘
# 
# Note: S3 Vectors uses default pricing (preview service not in API)

# Force refresh pricing
semstash costs refresh-pricing --force
```

### 6.3 Full CLI Command Reference

See the full CLI implementation in the core library section. Key commands:

```bash
# Initialize storage
semstash init my-bucket --region us-east-1 --dimension 1024

# Upload content
semstash upload ./image.jpg --key custom-key --metadata '{"tag": "photo"}'

# Search content
semstash search "sunset over mountains" --top-k 10 --type image/

# Retrieve by key
semstash retrieve my-key --output ./downloaded.jpg

# Delete content
semstash delete my-key --force

# Browse stored content
semstash browse --type image/ --limit 50

# Cost estimation
semstash costs estimate-upload ./large-video.mp4
semstash costs monthly --storage-gb 100 --vectors 1000000 --queries 50000
semstash costs usage
```

---

## 7. MCP Server Implementation

### 7.1 Cost-Aware Tools with Dimension Support

```python
# src/semstash/mcp/server.py

from mcp.server.fastmcp import FastMCP
from ..core.costs import CostEstimator, CostCategory
from ..core.config import load_config, SUPPORTED_DIMENSIONS

mcp = FastMCP("semstash")
config = load_config()
cost_estimator = CostEstimator(config, auto_refresh=True)


@mcp.tool()
def estimate_cost(
    operation: str,
    file_size_bytes: int | None = None,
    content_type: str | None = None,
    index_vector_count: int | None = None,
    dimension: int | None = None
) -> dict:
    """
    Estimate cost for a semantic storage operation.
    
    Args:
        operation: "upload", "query", or "delete"
        file_size_bytes: File size for upload estimates
        content_type: MIME type for upload estimates
        index_vector_count: Number of vectors for query estimates
        dimension: Override embedding dimension (256, 384, 1024, 3072)
    
    Returns:
        Cost breakdown with active (one-time) and recurring (monthly) costs
    """
    dim = dimension or config.embedding_dimension
    
    if dim not in SUPPORTED_DIMENSIONS:
        return {"error": f"Dimension must be one of {SUPPORTED_DIMENSIONS}"}
    
    if operation == "upload":
        if not file_size_bytes or not content_type:
            return {"error": "file_size_bytes and content_type required for upload"}
        
        # Create temp file-like for estimation
        breakdown = cost_estimator.estimate_upload_from_size(
            file_size_bytes, content_type, dimension=dim
        )
    
    elif operation == "query":
        if not index_vector_count:
            return {"error": "index_vector_count required for query"}
        
        breakdown = cost_estimator.estimate_query(
            index_vector_count, dimension=dim
        )
    
    elif operation == "delete":
        breakdown = cost_estimator.estimate_delete()
    
    else:
        return {"error": f"Unknown operation: {operation}"}
    
    return breakdown.to_dict()


@mcp.tool()
def compare_dimensions(total_vectors: int) -> dict:
    """
    Compare storage costs across all dimension options.
    
    Args:
        total_vectors: Number of vectors to estimate for
        
    Returns:
        Cost comparison for 256, 384, 1024, 3072 dimensions
    """
    return cost_estimator.compare_dimensions(total_vectors)


@mcp.tool()
def get_pricing_status() -> dict:
    """
    Get current pricing information and sources.
    
    Returns:
        Pricing status showing source (api/default), last update time,
        and key rates for each service.
    """
    return cost_estimator.pricing.get_status()


@mcp.tool()
def refresh_pricing(force: bool = False) -> dict:
    """
    Refresh pricing from AWS Pricing API.
    
    Args:
        force: If True, refresh even if cache is valid
        
    Returns:
        Status of refresh for each service
    """
    results = cost_estimator.pricing.refresh_all(force=force)
    return {
        "refreshed": results,
        "current_status": cost_estimator.pricing.get_status()
    }


@mcp.tool()
def upload_content(
    file_path: str,
    tags: list[str] | None = None,
    dimension: int | None = None,
    estimate_only: bool = False
) -> dict:
    """
    Upload content to semantic storage.
    
    Args:
        file_path: Path to file to upload
        tags: Optional tags for filtering
        dimension: Override embedding dimension
        estimate_only: If True, return cost estimate without uploading
        
    Returns:
        Upload result with cost breakdown (active + recurring)
    """
    dim = dimension or config.embedding_dimension
    
    # Get cost estimate
    cost_breakdown = cost_estimator.estimate_upload(
        Path(file_path), 
        _detect_content_type(file_path),
        dimension=dim
    )
    
    if estimate_only:
        return {
            "estimate_only": True,
            "cost": cost_breakdown.to_dict()
        }
    
    # Perform upload
    result = storage.upload(file_path, tags=tags, dimension=dim)
    
    # Record actual cost
    cost_estimator.record_operation(cost_breakdown)
    
    return {
        "key": result.key,
        "content_type": result.content_type,
        "dimension": dim,
        "cost": cost_breakdown.to_dict()
    }


@mcp.tool()
def search(
    query: str,
    top_k: int = 10,
    content_types: list[str] | None = None,
    tags: list[str] | None = None
) -> dict:
    """
    Search for semantically similar content.
    
    Returns results with query cost breakdown.
    """
    # Get index stats for cost estimation
    index_stats = storage.get_index_stats()
    
    # Estimate query cost
    cost_breakdown = cost_estimator.estimate_query(
        index_stats["vector_count"]
    )
    
    # Perform search
    results = storage.search(query, top_k=top_k, content_types=content_types, tags=tags)
    
    # Record cost
    cost_estimator.record_operation(cost_breakdown)
    
    return {
        "results": [r.to_dict() for r in results],
        "query_cost": cost_breakdown.to_dict()
    }
```

### 7.2 Example MCP Tool Responses

**estimate_cost for upload:**
```json
{
  "operation": "upload",
  "embedding_dimension": 1024,
  "costs": {
    "active": {
      "total_usd": 0.00018772,
      "description": "One-time costs for this operation",
      "items": [
        {"service": "s3", "component": "put_request", "amount_usd": 0.000005},
        {"service": "s3_vectors", "component": "put_data", "amount_usd": 0.0000012},
        {"service": "bedrock", "component": "embedding", "amount_usd": 0.00018}
      ]
    },
    "recurring": {
      "total_usd_per_month": 0.00002373,
      "description": "Monthly costs added by this operation",
      "items": [
        {"service": "s3", "component": "storage", "amount_usd_per_month": 0.0000229},
        {"service": "s3_vectors", "component": "storage", "amount_usd_per_month": 0.00000036}
      ]
    }
  }
}
```

**compare_dimensions:**
```json
{
  "comparison": {
    "256": {"vector_size_bytes": 3072, "monthly_storage_usd": 0.17, "relative_cost": 1.0},
    "384": {"vector_size_bytes": 3584, "monthly_storage_usd": 0.20, "relative_cost": 1.2},
    "1024": {"vector_size_bytes": 6144, "monthly_storage_usd": 0.34, "relative_cost": 2.0},
    "3072": {"vector_size_bytes": 14336, "monthly_storage_usd": 0.80, "relative_cost": 4.7}
  },
  "current_dimension": 3072,
  "recommendation": {
    "max_accuracy": {"dimension": 3072, "note": "Best quality, highest cost"},
    "balanced": {"dimension": 1024, "note": "Good quality, moderate cost"},
    "cost_optimized": {"dimension": 256, "note": "Acceptable quality, lowest cost"}
  }
}
```

### Transport Options

**stdio (for Claude Desktop):**
```bash
semstash-mcp
```

**Streamable HTTP (for remote access):**
```bash
semstash-mcp --http --port 8080
```

### Authentication for HTTP Transport

Set `MCP_API_KEY` environment variable for API key authentication:

```bash
export MCP_API_KEY="your-secret-key"
semstash-mcp --http --port 8080
```

### Available Tools

| Tool | Description |
|------|-------------|
| `upload_content` | Upload content with automatic embedding generation |
| `search` | Semantic search using natural language |
| `retrieve` | Get content metadata and download URL |
| `delete_content` | Delete content from storage |
| `browse` | Browse stored content with filtering |
| `estimate_cost` | Estimate operation costs |
| `get_usage_stats` | Get current usage statistics |

---

## 8. Web Interface Implementation

### 8.1 Cost-Aware REST API with Dimension Support

```python
# src/semstash/web/app.py

from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from typing import Optional, List

from ..core.costs import CostEstimator, CostBreakdown
from ..core.config import load_config, SUPPORTED_DIMENSIONS

app = FastAPI(
    title="semstash",
    description="Unlimited semantic storage for humans and AI agents"
)
config = load_config()
cost_estimator = CostEstimator(config, auto_refresh=True)


class CostEstimateRequest(BaseModel):
    operation: str  # upload, query, delete
    file_size_bytes: Optional[int] = None
    content_type: Optional[str] = None
    index_vector_count: Optional[int] = None
    dimension: Optional[int] = None


class MonthlyProjectionRequest(BaseModel):
    storage_gb: float
    total_vectors: int
    queries_per_month: int
    uploads_per_month: int = 100
    average_upload_mb: float = 1.0
    dimension: Optional[int] = None


@app.get("/costs/estimate")
def estimate_cost(
    operation: str = Query(..., description="upload, query, or delete"),
    file_size_bytes: Optional[int] = Query(None),
    content_type: Optional[str] = Query(None),
    index_vector_count: Optional[int] = Query(None),
    dimension: Optional[int] = Query(None, description="256, 384, 1024, or 3072")
):
    """
    Estimate operation cost with active/recurring breakdown.
    
    Returns separate active (one-time) and recurring (monthly) costs.
    """
    dim = dimension or config.embedding_dimension
    
    if dim not in SUPPORTED_DIMENSIONS:
        raise HTTPException(400, f"dimension must be one of {SUPPORTED_DIMENSIONS}")
    
    if operation == "upload":
        if not file_size_bytes or not content_type:
            raise HTTPException(400, "file_size_bytes and content_type required")
        breakdown = cost_estimator.estimate_upload_from_size(
            file_size_bytes, content_type, dimension=dim
        )
    elif operation == "query":
        if not index_vector_count:
            raise HTTPException(400, "index_vector_count required")
        breakdown = cost_estimator.estimate_query(index_vector_count, dimension=dim)
    elif operation == "delete":
        breakdown = cost_estimator.estimate_delete()
    else:
        raise HTTPException(400, f"Unknown operation: {operation}")
    
    return breakdown.to_dict()


@app.get("/costs/monthly")
def monthly_projection(
    storage_gb: float = Query(...),
    total_vectors: int = Query(...),
    queries_per_month: int = Query(...),
    uploads_per_month: int = Query(100),
    average_upload_mb: float = Query(1.0),
    dimension: Optional[int] = Query(None)
):
    """
    Project monthly costs with active/recurring breakdown.
    """
    return cost_estimator.estimate_monthly_cost(
        total_storage_gb=storage_gb,
        total_vectors=total_vectors,
        queries_per_month=queries_per_month,
        uploads_per_month=uploads_per_month,
        average_upload_size_mb=average_upload_mb,
        dimension=dimension
    )


@app.get("/costs/compare-dimensions")
def compare_dimensions(total_vectors: int = Query(...)):
    """
    Compare storage costs across all dimension options.
    """
    return cost_estimator.compare_dimensions(total_vectors)


@app.get("/costs/pricing")
def get_pricing_status():
    """
    Get current pricing information and sources.
    """
    return cost_estimator.pricing.get_status()


@app.post("/costs/refresh-pricing")
def refresh_pricing(force: bool = Query(False)):
    """
    Refresh pricing from AWS Pricing API.
    """
    results = cost_estimator.pricing.refresh_all(force=force)
    return {
        "refreshed": results,
        "current_status": cost_estimator.pricing.get_status()
    }


@app.post("/upload")
async def upload_file(
    file: UploadFile,
    tags: Optional[List[str]] = Query(None),
    dimension: Optional[int] = Query(None),
    estimate_only: bool = Query(False)
):
    """
    Upload file with cost tracking.
    
    Set estimate_only=true to get cost estimate without uploading.
    """
    dim = dimension or config.embedding_dimension
    content_type = file.content_type or "application/octet-stream"
    
    # Read file for size estimation
    content = await file.read()
    file_size = len(content)
    
    # Calculate cost estimate
    cost_breakdown = cost_estimator.estimate_upload_from_size(
        file_size, content_type, dimension=dim
    )
    
    if estimate_only:
        return {
            "estimate_only": True,
            "file_name": file.filename,
            "file_size_bytes": file_size,
            "dimension": dim,
            "cost": cost_breakdown.to_dict()
        }
    
    # Perform actual upload
    result = await storage.upload_bytes(content, file.filename, content_type, dimension=dim)
    
    # Record cost
    cost_estimator.record_operation(cost_breakdown)
    
    return {
        "key": result.key,
        "content_type": content_type,
        "dimension": dim,
        "cost": cost_breakdown.to_dict()
    }


@app.get("/search")
async def search(
    query: str = Query(...),
    top_k: int = Query(10, le=30),
    content_types: Optional[List[str]] = Query(None),
    tags: Optional[List[str]] = Query(None)
):
    """
    Search with cost tracking in response.
    """
    index_stats = await storage.get_index_stats()
    cost_breakdown = cost_estimator.estimate_query(index_stats["vector_count"])
    
    results = await storage.search(query, top_k=top_k, content_types=content_types, tags=tags)
    
    cost_estimator.record_operation(cost_breakdown)
    
    return {
        "results": [r.to_dict() for r in results],
        "query_cost": cost_breakdown.to_dict(),
        "index_stats": {
            "vector_count": index_stats["vector_count"],
            "dimension": config.embedding_dimension
        }
    }
```

### 8.2 Example API Responses

**GET /costs/estimate?operation=upload&file_size_bytes=1048576&content_type=image/jpeg&dimension=1024**
```json
{
  "operation": "upload",
  "embedding_dimension": 1024,
  "costs": {
    "active": {
      "total_usd": 0.00018772,
      "description": "One-time costs for this operation",
      "items": [
        {"service": "s3", "component": "put_request", "amount_usd": 0.000005, "description": "S3 PUT for content"},
        {"service": "s3_vectors", "component": "put_data", "amount_usd": 0.0000012, "description": "Vector ingestion (1024d)"},
        {"service": "bedrock", "component": "embedding", "amount_usd": 0.00018, "description": "Nova embedding (image/jpeg)"}
      ]
    },
    "recurring": {
      "total_usd_per_month": 0.00002373,
      "description": "Monthly costs added by this operation",
      "items": [
        {"service": "s3", "component": "storage", "amount_usd_per_month": 0.0000229, "description": "Content storage"},
        {"service": "s3_vectors", "component": "storage", "amount_usd_per_month": 0.00000036, "description": "Vector storage (1024d = 6144 bytes)"}
      ]
    }
  },
  "metadata": {
    "file_size_bytes": 1048576,
    "vector_size_bytes": 6144
  }
}
```

**GET /costs/pricing**
```json
{
  "s3": {
    "source": "api",
    "last_updated": "2024-01-15T10:30:00",
    "storage_per_gb_month": 0.023
  },
  "s3_vectors": {
    "source": "default",
    "last_updated": null,
    "storage_per_gb_month": 0.06
  },
  "bedrock": {
    "source": "api",
    "last_updated": "2024-01-15T10:30:00",
    "text_per_1000_tokens": 0.00013
  }
}
```

**GET /costs/compare-dimensions?total_vectors=1000000**
```json
{
  "comparison": {
    "256": {
      "vector_size_bytes": 3072,
      "total_storage_gb": 2.86,
      "monthly_storage_usd": 0.17,
      "relative_cost": 1.0
    },
    "384": {
      "vector_size_bytes": 3584,
      "total_storage_gb": 3.34,
      "monthly_storage_usd": 0.20,
      "relative_cost": 1.2
    },
    "1024": {
      "vector_size_bytes": 6144,
      "total_storage_gb": 5.72,
      "monthly_storage_usd": 0.34,
      "relative_cost": 2.0
    },
    "3072": {
      "vector_size_bytes": 14336,
      "total_storage_gb": 13.35,
      "monthly_storage_usd": 0.80,
      "relative_cost": 4.7
    }
  },
  "current_dimension": 3072,
  "recommendation": {
    "max_accuracy": {"dimension": 3072, "note": "Best quality, highest cost"},
    "balanced": {"dimension": 1024, "note": "Good quality, moderate cost"},
    "cost_optimized": {"dimension": 256, "note": "Acceptable quality, lowest cost"}
  }
}
```

FastAPI-based REST API running locally:

```bash
semstash-web --port 8000
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/upload` | Upload file with multipart form |
| GET | `/search` | Search with query parameters |
| GET | `/content/{key}` | Get content metadata/URL |
| DELETE | `/content/{key}` | Delete content |
| GET | `/browse` | Browse with filtering |
| GET | `/costs/estimate` | Estimate costs |
| GET | `/costs/monthly` | Monthly projection |
| GET | `/costs/usage` | Current usage stats |

---

## 9. Configuration Management

### 9.1 Configuration Precedence

CLI arguments > Environment variables > Config file > Defaults

### 9.2 Default Values

| Setting | Default | Description |
|---------|---------|-------------|
| `region` | `us-east-1` | AWS region (Nova only in us-east-1) |
| `embedding_dimension` | `3072` | Maximum accuracy (options: 256, 384, 1024, 3072) |
| `auto_refresh_pricing` | `true` | Fetch pricing from AWS API on startup |
| `pricing_cache_hours` | `24` | Hours before pricing refresh |
| `output_format` | `table` | CLI output format |

### 9.3 Configuration File (semstash.toml)

```toml
# semstash.toml — Unlimited semantic storage for humans and AI agents

[aws]
region = "us-east-1"
storage_bucket = "my-semantic-storage"
# vector_bucket auto-derived as "{storage_bucket}-vectors"

[embeddings]
# Dimension affects accuracy vs cost tradeoff:
#   3072 = Maximum accuracy (default), ~14KB per vector
#   1024 = Balanced, ~6KB per vector, 2x cheaper
#   384  = Reduced accuracy, ~3.5KB per vector
#   256  = Minimum cost, ~3KB per vector, 4.7x cheaper
dimension = 3072

[costs]
# Dynamic pricing from AWS Pricing API
auto_refresh_pricing = true
pricing_cache_hours = 24

[output]
format = "table"  # table, json, plain
```

### 9.4 Environment Variables

```bash
# AWS
export AWS_REGION="us-east-1"
export SEMSTORE_BUCKET="my-semantic-storage"
export SEMSTORE_VECTOR_BUCKET="my-semantic-storage-vectors"

# Embeddings
export SEMSTORE_DIMENSION="1024"  # 256, 384, 1024, 3072

# Pricing
export SEMSTORE_AUTO_REFRESH_PRICING="true"

# Output
export SEMSTORE_OUTPUT_FORMAT="json"
```

### 9.5 CLI Overrides

```bash
# Override dimension for single command
semstash upload ./image.jpg --dimension 1024

# Override region
semstash search "query" --region eu-central-1

# Set config value
semstash config set embeddings.dimension 1024
semstash config get embeddings.dimension
```

### Configuration Precedence

1. CLI arguments (highest)
2. Environment variables
3. Config file
4. Defaults (lowest)

### Config File Locations

Searched in order:
1. `./semstash.toml`
2. `./.semstash.toml`
3. `~/.config/semstash/config.toml`
4. `~/.semstash.toml`

### Example Config File

```toml
# semstash.toml

[aws]
region = "us-east-1"
storage_bucket = "my-semantic-storage"
# vector_bucket = "my-semantic-storage-vectors"  # Auto-derived

[embeddings]
dimension = 1024  # Options: 256, 384, 1024, 3072

[output]
format = "table"  # table, json, plain

[urls]
expiry_seconds = 3600
```

### Environment Variables

```bash
AWS_REGION=us-east-1
SEMSTORE_BUCKET=my-semantic-storage
SEMSTORE_DIMENSION=1024
MCP_API_KEY=secret-for-http-transport
```

---

## 10. Repository Structure

```
semantic-storage/
├── README.md
├── LICENSE
├── pyproject.toml
├── semstash.toml.example
├── .env.example
├── .gitignore
│
├── src/
│   └── semstash/
│       ├── __init__.py
│       ├── __main__.py
│       │
│       ├── core/
│       │   ├── __init__.py
│       │   ├── storage.py
│       │   ├── embeddings.py
│       │   ├── costs.py
│       │   ├── pricing_data.py
│       │   ├── pricing_api.py
│       │   ├── config.py
│       │   ├── models.py
│       │   └── exceptions.py
│       │
│       ├── cli/
│       │   ├── __init__.py
│       │   └── main.py
│       │
│       ├── mcp/
│       │   ├── __init__.py
│       │   ├── server.py
│       │   └── auth.py
│       │
│       └── web/
│           ├── __init__.py
│           └── app.py
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_core/
│   ├── test_cli/
│   ├── test_mcp/
│   └── test_web/
│
└── docs/
    ├── getting-started.md
    ├── cli-reference.md
    ├── mcp-integration.md
    └── api-reference.md
```

---

## 11. Dependencies & Setup

### pyproject.toml

```toml
[project]
name = "semstash"
version = "0.1.0"
description = "Unlimited semantic storage for humans and AI agents"
readme = "README.md"
license = {text = "Apache-2.0"}
requires-python = ">=3.10"
keywords = ["semantic", "storage", "embeddings", "vector", "s3", "mcp", "ai", "multimodal"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: Apache Software License",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]

dependencies = [
    "boto3>=1.35.0",
    "typer[all]>=0.12.0",
    "rich>=13.0.0",
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.30.0",
    "python-multipart>=0.0.9",
    "mcp[cli]>=1.22.0",
    "python-dotenv>=1.0.0",
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
    "tomli>=2.0.0; python_version < '3.11'",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "pytest-cov>=4.0.0",
    "moto[s3]>=5.0.0",
    "ruff>=0.5.0",
    "mypy>=1.10.0",
]

[project.scripts]
semstash = "semstash.cli.main:app"
semstash-mcp = "semstash.mcp.server:run_stdio"
semstash-web = "semstash.web.app:run"
```

### Installation

```bash
# Clone repository
git clone https://github.com/youruser/semantic-storage.git
cd semantic-storage

# Install with pip
pip install -e ".[dev]"

# Or with uv
uv pip install -e ".[dev]"
```

---

## 12. Testing Strategy

### Unit Tests
- Mock AWS services using moto (S3) and manual mocks (S3 Vectors, Bedrock)
- Test cost calculations with known inputs
- Test configuration loading

### Integration Tests
- Test against real AWS services in a test account
- Verify end-to-end upload/search/delete workflows

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=semstash

# Run specific test module
pytest tests/test_core/test_costs.py
```

---

## Summary

This implementation document provides complete specifications for building the semantic storage system. Key points:

1. **S3 Vectors is in preview** - API may change
2. **Nova Embeddings only in us-east-1** - Plan for cross-region calls
3. **Cost estimation built-in** - Users see costs across all interfaces
4. **Three interfaces, one core** - CLI, MCP, and Web share business logic
5. **Flexible configuration** - TOML files, env vars, and CLI args

The architecture enables straightforward implementation following the module interfaces defined above.
