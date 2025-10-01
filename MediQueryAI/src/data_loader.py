import os
import json
import requests
import pandas as pd
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class MedicalDataLoader:
    """Handles loading data from various medical sources"""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"

        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def fetch_pubmed_papers(self, query: str, max_results: int = 100) -> List[Dict]:
        """Fetch papers from PubMed API"""
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

        # Search for paper IDs
        search_url = f"{base_url}esearch.fcgi"
        search_params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json"
        }

        try:
            response = requests.get(search_url, params=search_params)
            response.raise_for_status()
            search_data = response.json()

            paper_ids = search_data.get("esearchresult", {}).get("idlist", [])

            if not paper_ids:
                return []

            # Fetch paper details
            fetch_url = f"{base_url}efetch.fcgi"
            fetch_params = {
                "db": "pubmed",
                "id": ",".join(paper_ids),
                "retmode": "xml"
            }

            papers_response = requests.get(fetch_url, params=fetch_params)
            papers_response.raise_for_status()

            # Parse XML and extract relevant information
            papers = self._parse_pubmed_xml(papers_response.text)

            # Save to file
            output_file = self.raw_dir / f"pubmed_{query.replace(' ', '_')}.json"
            with open(output_file, 'w') as f:
                json.dump(papers, f, indent=2)

            return papers

        except Exception as e:
            logger.error(f"Error fetching PubMed papers: {e}")
            return []

    def _parse_pubmed_xml(self, xml_content: str) -> List[Dict]:
        """Parse PubMed XML response (simplified version)"""
        # This is a simplified parser - in production, use proper XML parsing
        papers = []
        # Add actual XML parsing logic here
        return papers

    def load_medical_images(self, image_dir: str) -> List[Dict]:
        """Load medical images from directory"""
        image_extensions = ['.jpg', '.jpeg', '.png', '.dcm', '.nii']
        images = []

        image_path = Path(image_dir)
        if not image_path.exists():
            return images

        for ext in image_extensions:
            for img_file in image_path.glob(f"*{ext}"):
                images.append({
                    "path": str(img_file),
                    "filename": img_file.name,
                    "size": img_file.stat().st_size,
                    "type": ext[1:]  # Remove dot
                })

        return images