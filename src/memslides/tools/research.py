import os
import re
import sys
from pathlib import Path
from typing import Any, Literal

# Ensure direct script execution imports the local repo package first.
_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from fastmcp import FastMCP

try:
    import arxiv
except ImportError as exc:  # pragma: no cover - optional dependency
    arxiv = None  # type: ignore[assignment]
    _ARXIV_IMPORT_ERROR = str(exc)
else:
    _ARXIV_IMPORT_ERROR = ""

try:
    from semanticscholar import SemanticScholar
    from semanticscholar.Author import Author
except ImportError as exc:  # pragma: no cover - optional dependency
    SemanticScholar = None  # type: ignore[assignment]
    Author = Any  # type: ignore[assignment]
    _SEMANTIC_SCHOLAR_IMPORT_ERROR = str(exc)
else:
    _SEMANTIC_SCHOLAR_IMPORT_ERROR = ""

from memslides.utils.log import set_logger

mcp = FastMCP(name="MemSlidesResearchTools")
PAGE_SIZE = int(os.getenv("MEMSLIDES_ARXIV_PAGE_SIZE", "5"))
client = arxiv.Client(page_size=PAGE_SIZE) if arxiv is not None else None
try:
    sch = SemanticScholar() if SemanticScholar is not None else None
except Exception as exc:  # pragma: no cover - defensive optional setup
    sch = None
    _SEMANTIC_SCHOLAR_IMPORT_ERROR = str(exc)


def _unavailable_payload(
    error: str,
    *,
    query: str = "",
    include_papers: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "available": False,
        "error": error,
    }
    if query:
        payload["query"] = query
    if include_papers:
        payload["total_results"] = 0
        payload["papers"] = []
    return payload


@mcp.tool()
def search_papers(
    query: str,
    max_results: int | None = 5,
) -> dict[str, Any]:
    """
    Search for academic papers using arXiv query syntax.
    Best Practice: Start with broad queries and progressively add constraints if too many results are returned.

    QUERY PARAMETERS (field prefixes):
    - ti: Title field, e.g., ti:"attention mechanism"
    - au: Author name, e.g., au:"Hinton"
    - abs: Abstract content, e.g., abs:"transformer"
    - cat: Subject category, e.g., cat:cs.AI

    Args:
        query: Search query using arXiv syntax (keywords, field searches, boolean operators)
        max_results: Maximum number of results to return (default: 5)

    Returns:
        List of papers, each containing: title, authors, abstract, published date and pdf_url
    """
    if arxiv is None or client is None:
        return _unavailable_payload(
            f"arxiv is unavailable: {_ARXIV_IMPORT_ERROR}",
            query=query,
            include_papers=True,
        )

    # Map sort_by string to arxiv SortCriterion
    search = arxiv.Search(
        query=query,
        max_results=PAGE_SIZE,
    )

    papers = [
        {
            "title": paper.title,
            "authors": [author.name for author in paper.authors],
            "summary": paper.summary,
            "published": paper.published.strftime("%Y-%m-%d"),
            "pdf_url": paper.pdf_url,
        }
        for paper in client.results(search)
    ]
    if max_results is not None:
        papers = papers[:max_results]
    return {"total_results": len(papers), "papers": papers}


@mcp.tool()
def get_paper_authors(arxiv_id: str) -> dict:
    """
    Get the authors of a paper by arxiv id.
    Args:
        arxiv_id: The id of the paper, must be in the format of ARXIV:2501.03936
    Returns:
        List of authors, containing their authorId and details.
    """
    if sch is None:
        return _unavailable_payload(
            f"Semantic Scholar is unavailable: {_SEMANTIC_SCHOLAR_IMPORT_ERROR}",
        )
    if not re.fullmatch(r"ARXIV:\d{4}\.\d{4,5}(v\d+)?", arxiv_id):
        return {"error": "Invalid arxiv_id format. It should be like ARXIV:2501.03936"}
    fields = ["name", "citationCount", "affiliations"]
    authors: list[Author] = list(sch.get_paper_authors(arxiv_id, fields=fields))
    return {"authors": [author._data for author in authors]}


@mcp.tool()
def get_scholar_details(
    author_id: str,
    paper_start_index: int = 0,
    sort_by: Literal["citationCount", "year"] | None = None,
) -> dict[str, Any]:
    """
    Get the details of a scholar by author ID.

    Args:
        author_id: The Semantic Scholar author ID of the scholar
        paper_start_index: Starting index for paper pagination (default: 0)
        sort_by: Sort papers by "citationCount" or "year" (default: None - no sorting)

    Returns:
        Dictionary containing scholar details including:
        - Basic info: name, hIndex, homepage, paperCount
        - papers: List of papers with title, year, citationCount, publicationVenue, arxivId
    """
    if sch is None:
        return _unavailable_payload(
            f"Semantic Scholar is unavailable: {_SEMANTIC_SCHOLAR_IMPORT_ERROR}",
        )
    fields = [
        "hIndex",
        "homepage",
        "paperCount",
        "papers",
        "papers.citationCount",
        "papers.externalIds",
        "papers.publicationDate",
        "papers.publicationVenue",
        "papers.title",
        "papers.year",
        "name",
    ]
    author = sch.get_author(author_id, fields=fields)
    author, papers = author._data, author._data.pop("papers")
    processed_papers = []
    for p in papers:
        p["arxivId"] = p.pop("externalIds", {}).get("ArXiv", "")
        p.pop("paperId", "")
        processed_papers.append({k: v for k, v in p.items() if v is not None})
    if sort_by == "citationCount":
        processed_papers.sort(key=lambda x: x.get("citationCount", 0), reverse=True)
    elif sort_by == "year":
        processed_papers.sort(key=lambda x: x.get("year", 0), reverse=True)
    author["papers"] = processed_papers[
        paper_start_index : paper_start_index + PAGE_SIZE
    ]
    return author


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: python -m memslides.tools.search_tools <workspace>"
    work_dir = Path(sys.argv[1])
    assert work_dir.exists(), f"Workspace {work_dir} does not exist."
    os.chdir(work_dir)
    set_logger(
        f"memslides-research-tools-{work_dir.stem}",
        work_dir / ".history" / "memslides_research_tools.log",
    )

    mcp.run(show_banner=False)
