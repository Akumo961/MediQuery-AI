import { type NextRequest, NextResponse } from "next/server"
import { PubMedAPI, ClinicalTrialsAPI } from "@/lib/external-apis"

export async function POST(request: NextRequest) {
  try {
    const { query, type } = await request.json()

    if (type === "pubmed") {
      const pubmedAPI = new PubMedAPI()
      const articles = await pubmedAPI.searchArticles(query, 10)

      const results = articles.map((article) => ({
        id: `pmid-${article.pmid}`,
        title: article.title,
        authors: article.authors,
        journal: article.journal,
        publishDate: article.publishDate,
        abstract: article.abstract,
        url: `https://pubmed.ncbi.nlm.nih.gov/${article.pmid}`,
        citationCount: article.citationCount || 0,
        relevanceScore: 0.85 + Math.random() * 0.15, // Mock relevance score
      }))

      return NextResponse.json({ results })
    } else if (type === "clinical") {
      const clinicalAPI = new ClinicalTrialsAPI()
      const trials = await clinicalAPI.searchTrials(query, 10)

      const results = trials.map((trial) => ({
        id: `nct-${trial.nctId}`,
        title: trial.title,
        authors: [trial.sponsor],
        journal: "ClinicalTrials.gov",
        publishDate: trial.startDate,
        abstract: `Phase ${trial.phase} clinical trial for ${trial.condition}. Intervention: ${trial.intervention}. Status: ${trial.status}. Enrollment: ${trial.enrollment} participants.`,
        url: `https://clinicaltrials.gov/ct2/show/${trial.nctId}`,
        citationCount: 0,
        relevanceScore: 0.8 + Math.random() * 0.2,
      }))

      return NextResponse.json({ results })
    }

    return NextResponse.json({ results: [] })
  } catch (error) {
    console.error("Research search error:", error)
    return NextResponse.json({ error: "Failed to search research database" }, { status: 500 })
  }
}
