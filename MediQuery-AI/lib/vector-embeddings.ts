// Advanced vector search and embeddings system

export interface DocumentEmbedding {
  id: string
  content: string
  embedding: number[]
  metadata: {
    title: string
    source: string
    type: "research" | "clinical" | "guideline" | "textbook"
    date: string
    authors?: string[]
    keywords: string[]
  }
}

export interface SearchResult {
  document: DocumentEmbedding
  similarity: number
  relevantPassages: string[]
}

export class VectorEmbeddingService {
  private documents: DocumentEmbedding[] = []

  constructor() {
    this.initializeMockDocuments()
  }

  // Initialize with mock medical documents
  private initializeMockDocuments() {
    const mockDocuments = [
      {
        id: "doc-001",
        content:
          "Chest pain evaluation requires systematic assessment including history, physical examination, ECG, and cardiac biomarkers. The HEART score provides risk stratification for acute coronary syndrome. Patients with low HEART scores can be safely discharged with outpatient follow-up.",
        embedding: this.generateMockEmbedding(),
        metadata: {
          title: "Chest Pain Evaluation in Emergency Medicine",
          source: "Emergency Medicine Guidelines",
          type: "guideline" as const,
          date: "2023-08-15",
          authors: ["Emergency Medicine Society"],
          keywords: ["chest pain", "heart score", "emergency", "cardiac", "evaluation"],
        },
      },
      {
        id: "doc-002",
        content:
          "Diabetes management in elderly patients requires individualized glycemic targets. HbA1c goals should be less stringent (7.5-8.5%) to reduce hypoglycemia risk. Medication selection should consider renal function, cardiovascular comorbidities, and life expectancy.",
        embedding: this.generateMockEmbedding(),
        metadata: {
          title: "Diabetes Management in Geriatric Populations",
          source: "Geriatric Medicine Journal",
          type: "research" as const,
          date: "2023-09-20",
          authors: ["American Geriatrics Society"],
          keywords: ["diabetes", "elderly", "geriatric", "hba1c", "hypoglycemia"],
        },
      },
      {
        id: "doc-003",
        content:
          "Hypertension treatment in patients over 65 should target systolic BP <150 mmHg initially. ACE inhibitors or ARBs are first-line therapy. Thiazide diuretics provide cardiovascular protection. Monitor for orthostatic hypotension and electrolyte abnormalities.",
        embedding: this.generateMockEmbedding(),
        metadata: {
          title: "Hypertension Management in Older Adults",
          source: "Cardiology Clinical Practice",
          type: "clinical" as const,
          date: "2024-01-10",
          authors: ["American Heart Association"],
          keywords: ["hypertension", "elderly", "blood pressure", "ace inhibitors", "cardiovascular"],
        },
      },
      {
        id: "doc-004",
        content:
          "Pneumonia diagnosis requires clinical assessment, chest imaging, and laboratory studies. Community-acquired pneumonia severity can be assessed using CURB-65 or PSI scores. Antibiotic selection depends on severity, risk factors, and local resistance patterns.",
        embedding: this.generateMockEmbedding(),
        metadata: {
          title: "Community-Acquired Pneumonia Guidelines",
          source: "Infectious Disease Society",
          type: "guideline" as const,
          date: "2023-11-05",
          authors: ["IDSA Guidelines Committee"],
          keywords: ["pneumonia", "curb-65", "antibiotics", "respiratory", "infection"],
        },
      },
      {
        id: "doc-005",
        content:
          "Stroke recognition using FAST assessment (Face, Arms, Speech, Time) enables rapid identification. Acute ischemic stroke treatment with tPA is time-critical within 4.5 hours. Endovascular therapy may extend treatment window to 24 hours in selected patients.",
        embedding: this.generateMockEmbedding(),
        metadata: {
          title: "Acute Stroke Management Protocol",
          source: "Neurology Emergency Guidelines",
          type: "clinical" as const,
          date: "2023-12-15",
          authors: ["American Stroke Association"],
          keywords: ["stroke", "fast", "tpa", "endovascular", "neurology"],
        },
      },
    ]

    this.documents = mockDocuments
  }

  // Generate mock embedding vector (in production, use actual embedding models)
  private generateMockEmbedding(): number[] {
    return Array.from({ length: 384 }, () => Math.random() * 2 - 1)
  }

  // Calculate cosine similarity between two vectors
  private cosineSimilarity(a: number[], b: number[]): number {
    const dotProduct = a.reduce((sum, val, i) => sum + val * b[i], 0)
    const magnitudeA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0))
    const magnitudeB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0))
    return dotProduct / (magnitudeA * magnitudeB)
  }

  // Perform semantic search
  async semanticSearch(
    query: string,
    options: {
      limit?: number
      threshold?: number
      filters?: {
        type?: string[]
        dateRange?: { start: string; end: string }
        authors?: string[]
      }
    } = {},
  ): Promise<SearchResult[]> {
    const { limit = 10, threshold = 0.7, filters } = options

    // Generate query embedding (mock)
    const queryEmbedding = this.generateMockEmbedding()

    // Calculate similarities
    let results = this.documents.map((doc) => ({
      document: doc,
      similarity: this.cosineSimilarity(queryEmbedding, doc.embedding),
      relevantPassages: this.extractRelevantPassages(doc.content, query),
    }))

    // Apply filters
    if (filters) {
      results = results.filter((result) => {
        const { document } = result

        if (filters.type && !filters.type.includes(document.metadata.type)) {
          return false
        }

        if (
          filters.authors &&
          !filters.authors.some((author) =>
            document.metadata.authors?.some((docAuthor) => docAuthor.toLowerCase().includes(author.toLowerCase())),
          )
        ) {
          return false
        }

        if (filters.dateRange) {
          const docDate = new Date(document.metadata.date)
          const startDate = new Date(filters.dateRange.start)
          const endDate = new Date(filters.dateRange.end)
          if (docDate < startDate || docDate > endDate) {
            return false
          }
        }

        return true
      })
    }

    // Filter by similarity threshold and sort
    results = results
      .filter((result) => result.similarity >= threshold)
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, limit)

    return results
  }

  // Extract relevant passages from document content
  private extractRelevantPassages(content: string, query: string): string[] {
    const sentences = content.split(/[.!?]+/).filter((s) => s.trim().length > 0)
    const queryTerms = query.toLowerCase().split(/\s+/)

    return sentences.filter((sentence) => queryTerms.some((term) => sentence.toLowerCase().includes(term))).slice(0, 3) // Return top 3 relevant passages
  }

  // Add new document to the index
  async addDocument(document: Omit<DocumentEmbedding, "embedding">): Promise<void> {
    const embedding = this.generateMockEmbedding() // In production, generate actual embedding

    this.documents.push({
      ...document,
      embedding,
    })
  }

  // Get document statistics
  getIndexStats(): {
    totalDocuments: number
    documentsByType: Record<string, number>
    documentsBySource: Record<string, number>
  } {
    const stats = {
      totalDocuments: this.documents.length,
      documentsByType: {} as Record<string, number>,
      documentsBySource: {} as Record<string, number>,
    }

    this.documents.forEach((doc) => {
      stats.documentsByType[doc.metadata.type] = (stats.documentsByType[doc.metadata.type] || 0) + 1
      stats.documentsBySource[doc.metadata.source] = (stats.documentsBySource[doc.metadata.source] || 0) + 1
    })

    return stats
  }
}

// Singleton instance
export const vectorService = new VectorEmbeddingService()
