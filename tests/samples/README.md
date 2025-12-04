# Test Sample Files

This directory contains sample files for integration tests that run against real AWS services.

## Running Integration Tests

Integration tests are skipped by default. To run them with real AWS:

```bash
pytest --use-aws
```

**Prerequisites:**
- Valid AWS credentials configured
- AWS region set (default: us-east-1)
- Sufficient permissions for S3, S3 Vectors, and Bedrock

## Included Sample Files

### Text Files

| File | Size | Description |
|------|------|-------------|
| `sample.txt` | 650B | Plain text about SemStash features |
| `sample.json` | 501B | JSON document with nested metadata |

### Image Files

| File | Size | Format | Description |
|------|------|--------|-------------|
| `sample.png` | 20KB | PNG | 100x100 programmatic gradient |
| `sample.jpg` | 9KB | JPEG | 200x200 photo from Picsum |
| `sample.gif` | 1.5MB | GIF | 320x320 animated GIF |
| `sample.webp` | 30KB | WebP | 550x368 photo from Google |

### Audio Files

| File | Size | Format | Description |
|------|------|--------|-------------|
| `sample.mp3` | 8.5MB | MP3 | Music track from SoundHelix (CC) |
| `sample.wav` | 7.1MB | WAV | 16-bit stereo 48kHz audio |

### Video Files

| File | Size | Format | Description |
|------|------|--------|-------------|
| `sample.mp4` | 10MB | MP4 | Sample video clip |
| `sample_large.mp4` | 30MB | MP4 | Large video for size limit testing (~40MB base64)

### Document Files

| File | Size | Format | Description |
|------|------|--------|-------------|
| `sample.pdf` | 35KB | PDF | PDF with text and embedded image |
| `sample.docx` | 56KB | DOCX | Word document with text and image |
| `sample.pptx` | 50KB | PPTX | PowerPoint with title, content, and image slides |
| `sample.xlsx` | 6KB | XLSX | Excel spreadsheet with product data |

## Supported Content Types

Amazon Nova Multimodal Embeddings supports:

### Text
- Plain text files (`.txt`)
- JSON documents (`.json`)
- Any text-based format

### Images
- **Formats**: PNG, JPEG, GIF, WebP
- **Max size**: 25MB (base64 encoded)
- **Resolution**: Up to 8K

### Audio
- **Formats**: MP3, WAV, FLAC, OGG, WebM, AMR, 3GP, AAC
- **Duration**: 1 second to 12 minutes
- **Max size**: 40MB

### Video
- **Formats**: MP4, WebM, MOV, MKV, FLV, AVI, MPEG, 3GP
- **Duration**: 1 second to 12 minutes
- **Max size**: 25MB
- **Resolution**: Up to 8K

### Documents
- **PDF**: Rendered to image using PyMuPDF, embedded with DOCUMENT_IMAGE detail level
- **DOCX**: Text extracted using python-docx, embedded as text
- **PPTX**: Text extracted using python-pptx, embedded as text
- **XLSX**: Converted to CSV text using openpyxl, embedded as text

## Regenerating Sample Files

For CI/CD or clean setup, regenerate all samples:

```bash
cd tests/samples

# Text files
cat > sample.txt << 'EOF'
This is sample text content for testing semantic search capabilities.
SemStash is a semantic storage library for text, images, audio, and video.
EOF

cat > sample.json << 'EOF'
{"name": "SemStash Test", "type": "sample", "tags": ["test", "semantic"]}
EOF

# PNG (programmatic)
python3 << 'PYEOF'
import struct, zlib
def create_png(w=100, h=100):
    sig = b'\x89PNG\r\n\x1a\n'
    ihdr = struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0)
    ihdr_crc = zlib.crc32(b'IHDR' + ihdr) & 0xffffffff
    ihdr_chunk = struct.pack('>I', 13) + b'IHDR' + ihdr + struct.pack('>I', ihdr_crc)
    raw = b''.join(b'\x00' + bytes([int(255*x/w), int(255*y/h), 128] for x in range(w)) for y in range(h))
    idat = zlib.compress(raw, 9)
    idat_crc = zlib.crc32(b'IDAT' + idat) & 0xffffffff
    idat_chunk = struct.pack('>I', len(idat)) + b'IDAT' + idat + struct.pack('>I', idat_crc)
    iend_crc = zlib.crc32(b'IEND') & 0xffffffff
    iend_chunk = struct.pack('>I', 0) + b'IEND' + struct.pack('>I', iend_crc)
    return sig + ihdr_chunk + idat_chunk + iend_chunk
open('sample.png', 'wb').write(create_png())
PYEOF

# Download media from free sources
curl -sL "https://picsum.photos/200/200.jpg" -o sample.jpg
curl -sL "https://media.giphy.com/media/3o7TKSjRrfIPjeiVyM/giphy.gif" -o sample.gif
curl -sL "https://www.gstatic.com/webp/gallery/1.webp" -o sample.webp
curl -sL "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3" -o sample.mp3
curl -sL "https://file-examples.com/storage/fe19e15eac6560f8c936c41/2017/11/file_example_WAV_1MG.wav" -o sample.wav
curl -sL "https://www.learningcontainer.com/wp-content/uploads/2020/05/sample-mp4-file.mp4" -o sample.mp4

# Create large video (>25MB base64) for size limit testing
ffmpeg -y -stream_loop 2 -i sample.mp4 -c copy sample_large.mp4
```

## Large File Testing

For testing size limit handling (>25MB):

```bash
# Large image (requires ImageMagick)
convert -size 4000x4000 xc:gradient sample_large.png

# Large video by concatenation
ffmpeg -stream_loop 10 -i sample.mp4 -c copy sample_large.mp4
```

## Notes

- Sample files are gitignored (except README.md)
- Tests skip gracefully if required samples are missing
- Audio/video sourced from free/CC-licensed content
- All formats are verified to work with Amazon Nova embeddings
