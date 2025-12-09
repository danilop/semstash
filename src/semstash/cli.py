"""Command-line interface for semstash.

Provides a rich CLI for semantic storage operations.
All content is addressed by paths (like a filesystem), with '/' as root.

Usage:
    semstash init my-stash
    semstash my-stash upload photo.jpg /                    # Upload to root
    semstash my-stash upload photo.jpg /images/             # Upload to /images/ folder
    semstash my-stash upload photo.jpg /images/sunset.jpg   # Upload with rename
    semstash my-stash upload ./docs/ /docs/                 # Upload directory
    semstash my-stash upload ./docs/ /docs/ -p "*.txt"      # Upload with pattern
    semstash my-stash query "sunset on beach"
    semstash my-stash query "sunset" --path /images/        # Filter by path
    semstash my-stash query "sunset" -d ./downloads/
    semstash my-stash get /images/photo.jpg
    semstash my-stash get /docs/readme.pdf -d ./local.pdf -m
    semstash my-stash delete /images/photo.jpg
    semstash my-stash browse /                              # Browse root
    semstash my-stash browse /images/                       # Browse folder
    semstash my-stash stats
    semstash my-stash check
    semstash my-stash sync
    semstash my-stash destroy --force
    semstash my-stash agent                                 # AI agent session
"""

import json
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from semstash.client import SemStash
from semstash.config import (
    DEFAULT_BROWSE_LIMIT,
    DEFAULT_DIMENSION,
    DEFAULT_REGION,
    DEFAULT_SEARCH_TOP_K,
)
from semstash.exceptions import (
    AlreadyExistsError,
    ContentExistsError,
    ContentNotFoundError,
    NotInitializedError,
    SemStashError,
    StorageError,
    UnsupportedContentTypeError,
)
from semstash.markdown import is_markdown_convertible, to_markdown
from semstash.utils import format_size

# Create app and console
app = typer.Typer(
    name="semstash",
    help="Unlimited semantic storage for humans and AI agents.",
    no_args_is_help=True,
)
console = Console()


# --- Helpers ---


def download_content(
    client: SemStash,
    path: str,
    destination: Path,
    content_type: str,
    as_markdown: bool = False,
    output: str = "text",
) -> Path:
    """Download content and optionally convert to Markdown.

    Returns the final path (either original or .md if converted).
    """
    dest = client.download(path, destination)

    if as_markdown and is_markdown_convertible(content_type):
        md_content = to_markdown(dest, content_type)
        md_path = dest.with_suffix(".md")
        md_path.write_text(md_content, encoding="utf-8")
        dest.unlink()  # Remove original
        if output == "text":
            console.print(f"[green]✓[/green] {path} → [bold]{md_path.name}[/bold]")
        return md_path

    if output == "text":
        console.print(f"[green]✓[/green] Downloaded [bold]{dest.name}[/bold]")
    return dest


# --- Output Formatting ---


def output_json(data: dict[str, Any]) -> None:
    """Output data as JSON."""
    console.print(json.dumps(data, indent=2, default=str))


def error_exit(message: str, code: int = 1) -> None:
    """Print error and exit."""
    console.print(f"[red]Error:[/red] {message}")
    raise typer.Exit(code=code)


# --- Shared Options ---


OutputOption = Annotated[
    str,
    typer.Option(
        "--output",
        "-o",
        help="Output format: text or json",
    ),
]

QuietOption = Annotated[
    bool,
    typer.Option(
        "--quiet",
        "-q",
        help="Minimal output",
    ),
]

RegionOption = Annotated[
    str,
    typer.Option(
        "--region",
        "-r",
        help="AWS region",
    ),
]

DimensionOption = Annotated[
    int,
    typer.Option(
        "--dimension",
        help="Embedding dimension (256, 384, 1024, 3072)",
    ),
]

# Stash argument for all commands except init
StashArgument = Annotated[str, typer.Argument(help="Stash name (S3 bucket)")]


# --- Commands ---


@app.command()
def init(
    stash: Annotated[str, typer.Argument(help="Stash name (S3 bucket to create)")],
    region: RegionOption = DEFAULT_REGION,
    dimension: DimensionOption = DEFAULT_DIMENSION,
    output: OutputOption = "text",
) -> None:
    """Create new semantic storage (fails if already exists).

    Creates a new S3 bucket and vector index for semantic storage.
    """
    try:
        client = SemStash(
            bucket=stash,
            region=region,
            dimension=dimension,
            auto_init=False,
        )
        result = client.init()

        if output == "json":
            output_json(
                {
                    "bucket": result.bucket,
                    "vector_bucket": result.vector_bucket,
                    "region": result.region,
                    "dimension": result.dimension,
                }
            )
        else:
            console.print(f"[green]✓[/green] Created bucket: [bold]{result.bucket}[/bold]")
            console.print(
                f"[green]✓[/green] Created vector index: "
                f"[bold]{result.vector_bucket}/default-index[/bold]"
            )
            console.print(
                f"[green]✓[/green] Region: {result.region}, Dimension: {result.dimension}"
            )

    except AlreadyExistsError as e:
        error_exit(str(e))
    except SemStashError as e:
        error_exit(str(e))


@app.command()
def upload(
    stash: StashArgument,
    files: Annotated[
        list[Path], typer.Argument(help="Files or directories to upload (supports glob: *.jpg)")
    ],
    target: Annotated[
        str,
        typer.Argument(
            help="Target path: '/' root, '/folder/' keeps name, '/path/name.ext' renames"
        ),
    ],
    pattern: Annotated[
        str, typer.Option("--pattern", "-p", help="Glob pattern for directory uploads")
    ] = "**/*",
    tags: Annotated[list[str] | None, typer.Option("--tag", "-t", help="Tags")] = None,
    region: RegionOption = DEFAULT_REGION,
    force: Annotated[bool, typer.Option("--force", "-f", help="Overwrite if exists")] = False,
    output: OutputOption = "text",
    quiet: QuietOption = False,
) -> None:
    """Upload files or directories to semantic storage.

    Stores files and generates embeddings for semantic search.
    Target path determines where files are stored:
    - '/' uploads to root (preserves filename)
    - '/folder/' uploads to folder (preserves filename)
    - '/path/name.ext' uploads with exact path (renames file)

    When uploading a directory, the directory structure is preserved.
    Use --pattern to filter which files are uploaded from directories.

    Use --force to overwrite existing content.

    Examples:
        semstash mystash upload photo.jpg /
        semstash mystash upload photo.jpg /images/
        semstash mystash upload photo.jpg /images/sunset.jpg
        semstash mystash upload *.jpg /photos/
        semstash mystash upload documents/*.pdf /docs/ --tag work
        semstash mystash upload ./docs/ /docs/
        semstash mystash upload ./images/ /photos/ --pattern "*.jpg"
    """
    # Check if any input is a directory
    has_directory = any(f.is_dir() for f in files if f.exists())

    # Validate: directories require target ending with /
    if has_directory and not target.endswith("/"):
        error_exit("Target must end with '/' when uploading directories.")

    # Validate: multiple inputs require target ending with /
    if len(files) > 1 and not target.endswith("/"):
        error_exit("Target must end with '/' when uploading multiple files.")

    # Validate all files/directories exist first
    for file_path in files:
        if not file_path.exists():
            error_exit(f"File or directory not found: {file_path}")

    try:
        client = SemStash(bucket=stash, region=region)
        results = []
        show_progress = output == "text" and not quiet

        # Separate directories and files
        directories = [f for f in files if f.is_dir()]
        regular_files = [f for f in files if f.is_file()]

        # Handle directory uploads
        if directories:
            for source_dir in directories:
                if show_progress:
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        console=console,
                        transient=True,
                    ) as progress:
                        progress.add_task(f"Uploading {source_dir}...", total=None)
                        dir_results = client.upload_directory(
                            source_dir=source_dir,
                            target_path=target,
                            pattern=pattern,
                            tags=tags,
                            force=force,
                        )
                        results.extend(dir_results)
                else:
                    dir_results = client.upload_directory(
                        source_dir=source_dir,
                        target_path=target,
                        pattern=pattern,
                        tags=tags,
                        force=force,
                    )
                    results.extend(dir_results)

        # Handle file uploads
        if regular_files:
            if show_progress and len(regular_files) == 1 and not directories:
                # Single file: use spinner
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                    transient=True,
                ) as progress:
                    progress.add_task(f"Uploading {regular_files[0].name}...", total=None)
                    result = client.upload(
                        file_path=regular_files[0],
                        target=target,
                        tags=tags if tags else None,
                        force=force,
                    )
                    results.append(result)
                console.print(f"[green]✓[/green] Uploaded [bold]{result.path}[/bold]")
            elif show_progress and (len(regular_files) > 1 or directories):
                # Multiple files: use progress bar
                with Progress(console=console) as progress:
                    task = progress.add_task("Uploading files...", total=len(regular_files))
                    for file_path in regular_files:
                        progress.update(task, description=f"Uploading {file_path.name}...")
                        result = client.upload(
                            file_path=file_path,
                            target=target,
                            tags=tags if tags else None,
                            force=force,
                        )
                        results.append(result)
                        progress.advance(task)
            else:
                # Quiet or JSON mode: no progress
                for file_path in regular_files:
                    result = client.upload(
                        file_path=file_path,
                        target=target,
                        tags=tags if tags else None,
                        force=force,
                    )
                    results.append(result)

        if output == "json":
            output_json(
                {
                    "uploaded": [
                        {
                            "path": r.path,
                            "key": r.key,
                            "content_type": r.content_type,
                            "file_size": r.file_size,
                            "dimension": r.dimension,
                        }
                        for r in results
                    ],
                    "count": len(results),
                }
            )
        elif quiet:
            for r in results:
                console.print(r.path)
        elif len(results) > 1:
            console.print(f"[green]✓[/green] Uploaded {len(results)} files")

    except NotInitializedError:
        error_exit(f"Stash '{stash}' not found. Run 'semstash init {stash}' first.")
    except ContentExistsError as e:
        error_exit(f"{e}\nUse --force to overwrite.")
    except UnsupportedContentTypeError as e:
        error_exit(str(e))
    except SemStashError as e:
        error_exit(str(e))


@app.command()
def query(
    stash: StashArgument,
    query_text: Annotated[str, typer.Argument(help="Query text")],
    top_k: Annotated[
        int | None,
        typer.Option("--top-k", "-k", help=f"Max results (default: {DEFAULT_SEARCH_TOP_K})"),
    ] = None,
    content_type: Annotated[str, typer.Option("--type", help="Filter by content type")] = "",
    tags: Annotated[list[str] | None, typer.Option("--tag", "-t", help="Filter by tag")] = None,
    path: Annotated[
        str | None, typer.Option("--path", "-P", help="Filter by path prefix (e.g., /docs/)")
    ] = None,
    download: Annotated[
        Path | None, typer.Option("--download", "-d", help="Download results to directory")
    ] = None,
    urls: Annotated[
        bool, typer.Option("--urls", "-u", help="Output presigned URLs only (one per line)")
    ] = False,
    expiry: Annotated[
        int | None, typer.Option("--expiry", "-e", help="URL expiry in seconds (default: 3600)")
    ] = None,
    markdown: Annotated[
        bool, typer.Option("--markdown", "-m", help="Convert documents to Markdown")
    ] = False,
    region: RegionOption = DEFAULT_REGION,
    output: OutputOption = "text",
) -> None:
    """Query for content using natural language.

    Returns semantically similar content ranked by relevance.
    Use --path to filter results by path prefix.
    Use -d to download results to a directory.
    Use -u to output presigned URLs only (one per line, for piping).
    Use -m to convert documents (PDF, DOCX, etc.) to Markdown.
    Use SEMSTASH_SEARCH_TOP_K env var to change the default.

    Examples:
        semstash mystash query "sunset on beach"
        semstash mystash query "readme" --path /docs/
        semstash mystash query "cat photos" -d ./downloads/
    """
    # Validate mutually exclusive options
    if download and urls:
        error_exit("Cannot use --download and --urls together.")
    try:
        client = SemStash(bucket=stash, region=region)
        show_progress = output == "text"

        if show_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True,
            ) as progress:
                progress.add_task("Searching...", total=None)
                results = client.query(
                    query_text=query_text,
                    top_k=top_k,
                    content_type=content_type or None,
                    tags=tags,
                    path=path,
                    url_expiry=expiry,
                )
        else:
            results = client.query(
                query_text=query_text,
                top_k=top_k,
                content_type=content_type or None,
                tags=tags,
                path=path,
                url_expiry=expiry,
            )

        # Handle download
        if download:
            download.mkdir(parents=True, exist_ok=True)
            for r in results:
                download_content(
                    client=client,
                    path=r.path,
                    destination=download,
                    content_type=r.content_type or "",
                    as_markdown=markdown,
                    output=output,
                )
            return

        # Handle --urls: output presigned URLs only (one per line)
        if urls:
            for r in results:
                if r.url:
                    print(r.url)
            return

        if output == "json":
            output_json(
                {
                    "query": query_text,
                    "results": [
                        {
                            "path": r.path,
                            "key": r.key,
                            "score": r.score,
                            "content_type": r.content_type,
                            "file_size": r.file_size,
                            "url": r.url,
                        }
                        for r in results
                    ],
                }
            )
        else:
            console.print(f'\nFound {len(results)} results for "[bold]{query_text}[/bold]"\n')

            if results:
                table = Table(show_header=True, header_style="bold cyan")
                table.add_column("Score", width=8)
                table.add_column("Path", min_width=20)
                table.add_column("Type", width=15)
                table.add_column("Size", width=10, justify="right")

                for r in results:
                    table.add_row(
                        f"{r.score:.2f}",
                        r.path,
                        r.content_type or "unknown",
                        format_size(r.file_size) if r.file_size else "-",
                    )

                console.print(table)
            else:
                console.print("  [dim]No results found[/dim]")

    except NotInitializedError:
        error_exit(f"Stash '{stash}' not found. Run 'semstash init {stash}' first.")
    except SemStashError as e:
        error_exit(str(e))


@app.command()
def get(
    stash: StashArgument,
    paths: Annotated[list[str], typer.Argument(help="Content paths (e.g., /photo.jpg)")],
    download: Annotated[
        Path | None,
        typer.Option("--download", "-d", help="Download to path (directory for multiple items)"),
    ] = None,
    expiry: Annotated[
        int | None, typer.Option("--expiry", "-e", help="URL expiry in seconds (default: 3600)")
    ] = None,
    markdown: Annotated[
        bool, typer.Option("--markdown", "-m", help="Convert document to Markdown")
    ] = False,
    region: RegionOption = DEFAULT_REGION,
    output: OutputOption = "text",
) -> None:
    """Get content metadata and download URL.

    Use -d to save content to a local file or directory.
    Use -m to convert documents (PDF, DOCX, etc.) to Markdown.

    Examples:
        semstash mystash get /photo.jpg
        semstash mystash get /images/photo.jpg -d ./local.jpg
        semstash mystash get /docs/a.pdf /docs/b.pdf -d ./downloads/
    """
    try:
        client = SemStash(bucket=stash, region=region)
        results = []

        for path in paths:
            try:
                result = client.get(path, url_expiry=expiry)
                results.append(result)
            except ContentNotFoundError:
                if len(paths) == 1:
                    error_exit(f"Content not found: {path}")
                else:
                    console.print(f"[yellow]⚠[/yellow] Not found: {path}")
                    continue

        if not results:
            error_exit("No content found")

        # Handle download
        if download:
            # For multiple paths, download must be a directory
            if len(paths) > 1:
                download.mkdir(parents=True, exist_ok=True)
            for result in results:
                download_content(
                    client=client,
                    path=result.path,
                    destination=download,
                    content_type=result.content_type,
                    as_markdown=markdown,
                    output=output,
                )
            return

        if output == "json":
            if len(results) == 1:
                r = results[0]
                output_json(
                    {
                        "path": r.path,
                        "key": r.key,
                        "content_type": r.content_type,
                        "file_size": r.file_size,
                        "created_at": r.created_at.isoformat(),
                        "url": r.url,
                    }
                )
            else:
                output_json(
                    {
                        "items": [
                            {
                                "path": r.path,
                                "key": r.key,
                                "content_type": r.content_type,
                                "file_size": r.file_size,
                                "created_at": r.created_at.isoformat(),
                                "url": r.url,
                            }
                            for r in results
                        ],
                        "count": len(results),
                    }
                )
        else:
            for result in results:
                console.print(f"\n[bold]{result.path}[/bold]")
                console.print(f"  Type: {result.content_type}")
                console.print(f"  Size: {format_size(result.file_size)}")
                console.print(f"  Created: {result.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
                console.print(f"\n  URL: [link]{result.url}[/link]")

    except NotInitializedError:
        error_exit(f"Stash '{stash}' not found. Run 'semstash init {stash}' first.")
    except SemStashError as e:
        error_exit(str(e))


@app.command()
def delete(
    stash: StashArgument,
    paths: Annotated[list[str], typer.Argument(help="Content paths (e.g., /photo.jpg)")],
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation")] = False,
    region: RegionOption = DEFAULT_REGION,
    output: OutputOption = "text",
    quiet: QuietOption = False,
) -> None:
    """Delete content from storage.

    Removes both the content and its embedding.

    Examples:
        semstash mystash delete /photo.jpg
        semstash mystash delete /images/photo.jpg
        semstash mystash delete /docs/a.pdf /docs/b.pdf -y
    """
    if not yes and output == "text":
        if len(paths) == 1:
            confirm = typer.confirm(f"Delete {paths[0]}?", default=False)
        else:
            confirm = typer.confirm(f"Delete {len(paths)} items?", default=False)
        if not confirm:
            raise typer.Abort()

    try:
        client = SemStash(bucket=stash, region=region)
        results = []

        for path in paths:
            result = client.delete(path)
            results.append(result)
            if output == "text" and not quiet:
                console.print(f"[green]✓[/green] Deleted [bold]{path}[/bold]")

        if output == "json":
            if len(results) == 1:
                output_json(
                    {"path": paths[0], "key": results[0].key, "deleted": results[0].deleted}
                )
            else:
                output_json(
                    {
                        "deleted": [
                            {"path": p, "key": r.key, "deleted": r.deleted}
                            for p, r in zip(paths, results, strict=True)
                        ],
                        "count": len(results),
                    }
                )

    except NotInitializedError:
        error_exit(f"Stash '{stash}' not found. Run 'semstash init {stash}' first.")
    except SemStashError as e:
        error_exit(str(e))


@app.command()
def browse(
    stash: StashArgument,
    path: Annotated[
        str,
        typer.Argument(help="Path to browse: '/' for root, '/folder/' for subfolder"),
    ],
    content_type: Annotated[str, typer.Option("--type", help="Content type filter")] = "",
    limit: Annotated[
        int | None,
        typer.Option("--limit", "-l", help=f"Max results (default: {DEFAULT_BROWSE_LIMIT})"),
    ] = None,
    region: RegionOption = DEFAULT_REGION,
    output: OutputOption = "text",
) -> None:
    """Browse stored content at a path.

    List content at the specified path with optional filtering by type.
    Use SEMSTASH_BROWSE_LIMIT env var to change the default.

    Examples:
        semstash mystash browse /
        semstash mystash browse /images/
        semstash mystash browse /docs/ --type application/pdf
    """
    try:
        client = SemStash(bucket=stash, region=region)
        result = client.browse(
            path=path,
            content_type=content_type or None,
            limit=limit,
        )

        if output == "json":
            output_json(
                {
                    "path": path,
                    "items": [
                        {
                            "path": item.path,
                            "key": item.key,
                            "content_type": item.content_type,
                            "file_size": item.file_size,
                            "created_at": item.created_at.isoformat(),
                        }
                        for item in result.items
                    ],
                    "total": result.total,
                    "has_more": result.next_token is not None,
                }
            )
        else:
            if not result.items:
                console.print(f"\n  [dim]No content found at {path}[/dim]")
                return

            table = Table(show_header=True, header_style="bold cyan")
            table.add_column("Path", min_width=25)
            table.add_column("Type", width=15)
            table.add_column("Size", width=10, justify="right")
            table.add_column("Created", width=12)

            for item in result.items:
                table.add_row(
                    item.path,
                    item.content_type,
                    format_size(item.file_size),
                    item.created_at.strftime("%Y-%m-%d"),
                )

            console.print()
            console.print(table)

            if result.next_token:
                console.print("\n  [dim]More results available...[/dim]")

    except NotInitializedError:
        error_exit(f"Stash '{stash}' not found. Run 'semstash init {stash}' first.")
    except SemStashError as e:
        error_exit(str(e))


@app.command()
def stats(
    stash: StashArgument,
    region: RegionOption = DEFAULT_REGION,
    output: OutputOption = "text",
) -> None:
    """Show storage statistics and AWS resources."""
    try:
        client = SemStash(bucket=stash, region=region)
        result = client.get_stats()

        if output == "json":
            output_json(
                {
                    "content_count": result.content_count,
                    "vector_count": result.vector_count,
                    "storage_bytes": result.storage_bytes,
                    "storage_gb": result.storage_gb,
                    "dimension": result.dimension,
                    "bucket": result.bucket,
                    "vector_bucket": result.vector_bucket,
                    "index_name": result.index_name,
                    "region": result.region,
                }
            )
        else:
            console.print("\n[bold]Storage Statistics[/bold]\n")
            console.print(f"  Content items: {result.content_count:,}")
            console.print(f"  Vectors: {result.vector_count:,}")
            console.print(f"  Storage: {format_size(result.storage_bytes)}")
            console.print(f"  Dimension: {result.dimension}")
            console.print("\n[bold]AWS Resources[/bold]\n")
            console.print(f"  Region: {result.region}")
            console.print(f"  S3 Bucket: {result.bucket}")
            console.print(f"  S3 Vectors Bucket: {result.vector_bucket}")
            console.print(f"  Vector Index: {result.index_name}")

    except NotInitializedError:
        error_exit(f"Stash '{stash}' not found. Run 'semstash init {stash}' first.")
    except SemStashError as e:
        error_exit(str(e))


@app.command()
def check(
    stash: StashArgument,
    region: RegionOption = DEFAULT_REGION,
    output: OutputOption = "text",
) -> None:
    """Check storage consistency.

    Compares S3 content with vector embeddings to identify:
    - Orphaned vectors: embeddings without corresponding S3 objects
    - Missing vectors: S3 objects without embeddings
    """
    try:
        client = SemStash(bucket=stash, region=region)
        result = client.check()

        if output == "json":
            output_json(
                {
                    "content_count": result.content_count,
                    "vector_count": result.vector_count,
                    "orphaned_vectors": result.orphaned_vectors,
                    "missing_vectors": result.missing_vectors,
                    "is_consistent": result.is_consistent,
                }
            )
        else:
            console.print("\n[bold]Storage Consistency Check[/bold]\n")
            console.print(f"  Content items: {result.content_count:,}")
            console.print(f"  Vectors: {result.vector_count:,}")

            if result.is_consistent:
                console.print("\n  [green]✓[/green] Storage is consistent!")
            else:
                if result.orphaned_vectors:
                    console.print(
                        f"\n  [yellow]⚠[/yellow] Orphaned vectors: {len(result.orphaned_vectors)}"
                    )
                    for key in result.orphaned_vectors[:5]:
                        console.print(f"    - {key}")
                    if len(result.orphaned_vectors) > 5:
                        console.print(f"    ... and {len(result.orphaned_vectors) - 5} more")

                if result.missing_vectors:
                    console.print(
                        f"\n  [yellow]⚠[/yellow] Missing vectors: {len(result.missing_vectors)}"
                    )
                    for key in result.missing_vectors[:5]:
                        console.print(f"    - {key}")
                    if len(result.missing_vectors) > 5:
                        console.print(f"    ... and {len(result.missing_vectors) - 5} more")

                console.print(f"\n  Run 'semstash {stash} sync' to fix inconsistencies.")

    except NotInitializedError:
        error_exit(f"Stash '{stash}' not found. Run 'semstash init {stash}' first.")
    except SemStashError as e:
        error_exit(str(e))


@app.command()
def sync(
    stash: StashArgument,
    delete_orphaned: Annotated[
        bool, typer.Option("--delete-orphaned", help="Delete orphaned vectors")
    ] = True,
    create_missing: Annotated[
        bool, typer.Option("--create-missing", help="Create missing embeddings")
    ] = True,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation")] = False,
    region: RegionOption = DEFAULT_REGION,
    output: OutputOption = "text",
    quiet: QuietOption = False,
) -> None:
    """Synchronize storage consistency.

    Fixes inconsistencies between S3 content and vector embeddings:
    - Deletes orphaned vectors (embeddings without S3 objects)
    - Creates missing embeddings for S3 objects
    """
    try:
        client = SemStash(bucket=stash, region=region)

        # First check what needs to be done
        check_result = client.check()

        if check_result.is_consistent:
            if output == "json":
                output_json(
                    {
                        "deleted_vectors": [],
                        "created_vectors": [],
                        "failed_keys": [],
                        "deleted_count": 0,
                        "created_count": 0,
                        "message": "Storage is already consistent",
                    }
                )
            elif not quiet:
                console.print("[green]✓[/green] Storage is already consistent!")
            return

        # Show what will be done and confirm
        if not yes and output == "text":
            console.print("\n[bold]Sync Preview[/bold]\n")
            if check_result.orphaned_vectors and delete_orphaned:
                console.print(
                    f"  Will delete {len(check_result.orphaned_vectors)} orphaned vectors"
                )
            if check_result.missing_vectors and create_missing:
                console.print(
                    f"  Will create {len(check_result.missing_vectors)} missing embeddings"
                )
            console.print()

            confirm = typer.confirm("Proceed with sync?", default=False)
            if not confirm:
                raise typer.Abort()

        if not quiet and output == "text":
            console.print("Synchronizing...", end=" ")

        result = client.sync(
            delete_orphaned=delete_orphaned,
            create_missing=create_missing,
        )

        if output == "json":
            output_json(
                {
                    "deleted_vectors": result.deleted_vectors,
                    "created_vectors": result.created_vectors,
                    "failed_keys": result.failed_keys,
                    "deleted_count": result.deleted_count,
                    "created_count": result.created_count,
                    "message": result.message,
                }
            )
        elif quiet:
            console.print(f"Deleted: {result.deleted_count}, Created: {result.created_count}")
        else:
            console.print("[green]done[/green]")
            console.print(f"\n  Deleted orphaned vectors: {result.deleted_count}")
            console.print(f"  Created missing embeddings: {result.created_count}")
            if result.failed_keys:
                console.print(f"  [yellow]Failed:[/yellow] {len(result.failed_keys)} keys")

    except NotInitializedError:
        error_exit(f"Stash '{stash}' not found. Run 'semstash init {stash}' first.")
    except SemStashError as e:
        error_exit(str(e))


@app.command()
def destroy(
    stash: StashArgument,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation")] = False,
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Force delete non-empty storage")
    ] = False,
    region: RegionOption = DEFAULT_REGION,
    output: OutputOption = "text",
) -> None:
    """Permanently destroy the semantic stash.

    WARNING: This operation is irreversible! It permanently deletes:
    - The S3 bucket with all content
    - The S3 Vectors bucket with all embeddings
    """
    try:
        client = SemStash(bucket=stash, region=region)

        # Show warning and get stats
        if not yes and output == "text":
            try:
                stats_result = client.get_stats()
                console.print("\n[bold red]WARNING: Destructive Operation[/bold red]\n")
                console.print(f"  Bucket: {stash}")
                console.print(f"  Content items: {stats_result.content_count:,}")
                console.print(f"  Vectors: {stats_result.vector_count:,}")
                console.print(f"  Storage: {format_size(stats_result.storage_bytes)}")
                console.print("\n  [red]This will permanently delete all data![/red]\n")
            except SemStashError:
                console.print("\n[bold red]WARNING: Destructive Operation[/bold red]\n")
                console.print("  [red]This will permanently delete all data![/red]\n")

            confirm = typer.confirm("Are you sure you want to destroy this stash?", default=False)
            if not confirm:
                raise typer.Abort()

        result = client.destroy(force=force)

        if output == "json":
            output_json(
                {
                    "bucket": result.bucket,
                    "vector_bucket": result.vector_bucket,
                    "content_deleted": result.content_deleted,
                    "vectors_deleted": result.vectors_deleted,
                    "bucket_deleted": result.bucket_deleted,
                    "vector_bucket_deleted": result.vector_bucket_deleted,
                    "destroyed": result.destroyed,
                    "message": result.message,
                }
            )
        else:
            if result.destroyed:
                console.print("[green]✓[/green] Stash destroyed successfully")
                console.print(f"  Content deleted: {result.content_deleted:,}")
                console.print(f"  Vectors deleted: {result.vectors_deleted:,}")
            else:
                console.print("[yellow]⚠[/yellow] Partial destruction")
                console.print(f"  {result.message}")

    except StorageError as e:
        error_exit(f"{e}\nUse --force to delete non-empty storage.")
    except NotInitializedError:
        error_exit(f"Stash '{stash}' not found. Run 'semstash init {stash}' first.")
    except SemStashError as e:
        error_exit(str(e))


@app.command()
def web(
    stash: Annotated[str, typer.Argument(help="Stash name")],
    host: Annotated[  # noqa: S104 - intentional for server
        str, typer.Option("--host", "-h", help="Host to bind to")
    ] = "0.0.0.0",  # nosec B104 - intentional for server
    port: Annotated[int, typer.Option("--port", "-p", help="Port to listen on")] = 8000,
) -> None:
    """Start the web interface and REST API.

    Examples:
        semstash web my-stash
        semstash web my-stash --port 8080
    """
    import os

    import uvicorn

    os.environ["SEMSTASH_BUCKET"] = stash
    from semstash.web import app as web_app

    console.print(f"Starting web server at http://{host}:{port}/ui/")
    uvicorn.run(web_app, host=host, port=port)


@app.command()
def mcp(
    stash: Annotated[str | None, typer.Argument(help="Stash name (or set SEMSTASH_BUCKET)")] = None,
) -> None:
    """Start the MCP (Model Context Protocol) server.

    Examples:
        semstash mcp my-stash
        SEMSTASH_BUCKET=my-stash semstash mcp
    """
    import os

    if stash:
        os.environ["SEMSTASH_BUCKET"] = stash

    if not os.environ.get("SEMSTASH_BUCKET"):
        error_exit("Stash required. Provide stash name or set SEMSTASH_BUCKET.")

    from semstash.mcp_server import main as mcp_main

    mcp_main()


@app.command()
def agent(
    stash: StashArgument,
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Bedrock model ID"),
    ] = "global.amazon.nova-2-lite-v1:0",
    region: RegionOption = DEFAULT_REGION,
) -> None:
    """Start an interactive AI agent session.

    The agent helps you interact with content in the stash:
    - Search and query for content
    - Get summaries of documents
    - Browse stored content
    - Delete content when requested

    Commands:
        /quit or /exit - Exit the agent
        /reset or /clear - Clear conversation history
        /help - Show available commands

    Examples:
        semstash my-stash agent
        semstash my-stash agent --model us.amazon.nova-pro-v1:0
    """
    import os
    import readline  # noqa: F401 - enables input history

    from semstash.agent import SemStashAgent

    # Set region if provided
    if region:
        os.environ["AWS_DEFAULT_REGION"] = region

    console.print("\n[bold cyan]SemStash Agent[/bold cyan]")
    console.print(f"Connected to stash: [bold]{stash}[/bold]")
    console.print(f"Model: {model}")
    console.print("\nType your message or /help for commands. Use /quit to exit.\n")

    try:
        with SemStashAgent(
            bucket=stash,
            model_id=model,
            streaming=True,
            region=region,
        ) as agent_instance:
            while True:
                try:
                    # Get user input (supports multi-line with continuation)
                    user_input = console.input("[bold green]You:[/bold green] ").strip()

                    if not user_input:
                        continue

                    # Handle commands
                    if user_input.startswith("/"):
                        cmd = user_input.lower()
                        if cmd in ("/quit", "/exit", "/q"):
                            console.print("\n[dim]Goodbye![/dim]")
                            break
                        elif cmd in ("/reset", "/clear"):
                            agent_instance.reset_conversation()
                            console.print("[dim]Conversation cleared.[/dim]\n")
                            continue
                        elif cmd == "/help":
                            console.print("\n[bold]Available Commands:[/bold]")
                            console.print("  /quit, /exit, /q  - Exit the agent")
                            console.print("  /reset, /clear    - Clear conversation history")
                            console.print("  /help             - Show this help message")
                            console.print()
                            continue
                        else:
                            console.print(f"[yellow]Unknown command: {user_input}[/yellow]")
                            console.print("Type /help for available commands.\n")
                            continue

                    # Stream agent response
                    console.print("[bold blue]Agent:[/bold blue] ", end="")

                    response_text = ""
                    try:
                        for event in agent_instance.chat_stream(user_input):
                            # Events are dicts - check for "data" key (text chunks)
                            if isinstance(event, dict) and "data" in event:
                                chunk = event["data"]
                                if isinstance(chunk, str):
                                    response_text += chunk
                                    print(chunk, end="", flush=True)
                    except Exception as stream_err:
                        # Fallback to non-streaming if streaming fails
                        if not response_text:
                            response_text = agent_instance.chat(user_input)
                            console.print(response_text, end="")
                        else:
                            console.print(f"\n[yellow]Stream interrupted: {stream_err}[/yellow]")

                    console.print("\n")

                except KeyboardInterrupt:
                    console.print("\n[dim]Use /quit to exit.[/dim]\n")
                    continue
                except EOFError:
                    console.print("\n[dim]Goodbye![/dim]")
                    break

    except NotInitializedError:
        error_exit(f"Stash '{stash}' not found. Run 'semstash init {stash}' first.")
    except SemStashError as e:
        error_exit(str(e))
    except Exception as e:
        error_exit(f"Agent error: {e}")


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
