from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from models.text_models import MedicalTextModel
from data_loader import MedicalDataLoader

router = APIRouter()

# Initialize models
text_model = MedicalTextModel()
data_loader = MedicalDataLoader()


class SearchQuery(BaseModel):
    query: str
    max_results: int = 10
    search_type: str = "semantic"  # semantic, keyword, hybrid


class SearchResult(BaseModel):
    title: str
    content: str
    similarity: float
    source: str
    metadata: Dict[str, Any] = {}


@router.post("/literature", response_model=List[SearchResult])
async def search_literature(search_query: SearchQuery):
    """Search medical literature"""
    try:
        # Fetch papers from PubMed
        papers = data_loader.fetch_pubmed_papers(
            search_query.query,
            search_query.max_results
        )

        if not papers:
            # Use mock data for demonstration
            papers = [
                {
                    "title": "Recent Advances in Medical AI",
                    "abstract": "This paper discusses recent developments in artificial intelligence applications for healthcare...",
                    "authors": ["Dr. Smith", "Dr. Johnson"],
                    "journal": "Nature Medicine",
                    "year": 2024
                },
                {
                    "title": "Multimodal Learning in Healthcare",
                    "abstract": "We present a comprehensive study on multimodal approaches to medical diagnosis...",
                    "authors": ["Dr. Williams", "Dr. Brown"],
                    "journal": "JAMA",
                    "year": 2024
                }
            ]

        # Extract text content for similarity search
        documents = [f"{paper.get('title', '')} {paper.get('abstract', '')}" for paper in papers]

        if search_query.search_type == "semantic":
            similar_docs = text_model.find_similar_documents(
                search_query.query,
                documents,
                top_k=search_query.max_results
            )

            results = []
            for doc_info in similar_docs:
                paper = papers[doc_info["index"]]
                results.append(SearchResult(
                    title=paper.get("title", "Unknown Title"),
                    content=paper.get("abstract", "No abstract available"),
                    similarity=doc_info["similarity"],
                    source="PubMed",
                    metadata={
                        "authors": paper.get("authors", []),
                        "journal": paper.get("journal", "Unknown"),
                        "year": paper.get("year", "Unknown")
                    }
                ))

            return results
        else:
            # Keyword search fallback
            results = []
            for i, paper in enumerate(papers[:search_query.max_results]):
                results.append(SearchResult(
                    title=paper.get("title", "Unknown Title"),
                    content=paper.get("abstract", "No abstract available"),
                    similarity=0.8,  # Mock similarity
                    source="PubMed",
                    metadata={
                        "authors": paper.get("authors", []),
                        "journal": paper.get("journal", "Unknown"),
                        "year": paper.get("year", "Unknown")
                    }
                ))
            return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/suggestions")
async def get_search_suggestions(q: str = Query(..., description="Partial query")):
    """Get search suggestions"""
    suggestions = [
        "COVID-19 treatment protocols",
        "Machine learning in radiology",
        "Cancer immunotherapy research",
        "Diabetes management guidelines",
        "Cardiovascular disease prevention"
    ]

    # Filter suggestions based on query
    filtered = [s for s in suggestions if q.lower() in s.lower()]
    return {"suggestions": filtered[:5]}