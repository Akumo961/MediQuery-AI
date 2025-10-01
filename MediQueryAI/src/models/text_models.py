from sentence_transformers import SentenceTransformer
from transformers import pipeline
import numpy as np
from typing import List, Dict, Any
import os
from pathlib import Path


class MedicalTextModel:
    def __init__(self):
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        # Initialize QA pipeline
        self.qa_pipeline = pipeline(
            "question-answering",
            model="deepset/roberta-base-squad2",
            tokenizer="deepset/roberta-base-squad2"
        )

        self.documents = []

    def find_similar_documents(self, query: str, documents: List[str], top_k: int = 5) -> List[Dict]:
        """Find similar documents using semantic search"""
        if not documents:
            return []

        # Encode query and documents
        query_embedding = self.embedding_model.encode([query])
        doc_embeddings = self.embedding_model.encode(documents)

        # Calculate cosine similarity
        similarities = np.dot(doc_embeddings, query_embedding.T).flatten()

        # Get top k results
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "index": int(idx),
                "similarity": float(similarities[idx]),
                "content": documents[idx]
            })

        return results

    def answer_question(self, question: str, context: str) -> Dict[str, Any]:
        """Answer question based on context"""
        try:
            result = self.qa_pipeline({
                'question': question,
                'context': context[:1000]  # Limit context length
            })
            return {
                "answer": result['answer'],
                "confidence": result['score'],
                "start": result.get('start', 0),
                "end": result.get('end', 0)
            }
        except Exception as e:
            return {
                "answer": "I couldn't process this question with the provided context.",
                "confidence": 0.0,
                "start": 0,
                "end": 0
            }

    def summarize_text(self, text: str, max_length: int = 150) -> str:
        """Simple text summarization"""
        sentences = text.split('.')
        if len(sentences) <= 3:
            return text

        # Return first few sentences as summary
        summary = '. '.join(sentences[:3]) + '.'
        return summary