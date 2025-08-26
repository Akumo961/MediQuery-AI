import { type NextRequest, NextResponse } from "next/server"
import { vectorService } from "@/lib/vector-embeddings"

export async function POST(request: NextRequest) {
  try {
    const { query, filters, limit, threshold } = await request.json()

    const results = await vectorService.semanticSearch(query, {
      limit: limit || 10,
      threshold: threshold || 0.7,
      filters,
    })

    // Transform results for frontend
    const transformedResults = results.map((result) => ({
      id: result.document.id,
      title: result.document.metadata.title,
      content: result.document.content,
      similarity: result.similarity,
      source: result.document.metadata.source,
      metadata: result.document.metadata,
      relevantPassages: result.relevantPassages,
    }))

    return NextResponse.json({
      results: transformedResults,
      totalFound: results.length,
      searchQuery: query,
    })
  } catch (error) {
    console.error("Vector search error:", error)
    return NextResponse.json({ error: "Failed to perform semantic search" }, { status: 500 })
  }
}

export async function GET() {
  try {
    const stats = vectorService.getIndexStats()
    return NextResponse.json(stats)
  } catch (error) {
    console.error("Vector index stats error:", error)
    return NextResponse.json({ error: "Failed to get index statistics" }, { status: 500 })
  }
}
