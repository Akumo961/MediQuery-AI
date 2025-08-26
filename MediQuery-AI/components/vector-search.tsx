"use client"

import { useState } from "react"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent } from "@/components/ui/card"
import { Brain, Calendar, Users } from "lucide-react"
import { AdvancedSearch } from "./advanced-search"

interface SimilarityResult {
  id: string
  title: string
  content: string
  similarity: number
  source: string
  metadata: {
    type: "research" | "clinical" | "guideline" | "textbook"
    date: string
    authors?: string[]
    keywords: string[]
  }
  relevantPassages: string[]
}

export function VectorSearch() {
  const [results, setResults] = useState<SimilarityResult[]>([])
  const [isSearching, setIsSearching] = useState(false)
  const [searchQuery, setSearchQuery] = useState("")

  const handleAdvancedSearch = async (query: string, options: any) => {
    setIsSearching(true)
    setSearchQuery(query)

    try {
      const response = await fetch("/api/vector-search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, ...options }),
      })

      const data = await response.json()
      setResults(data.results)
    } catch (error) {
      console.error("Error in semantic search:", error)
    } finally {
      setIsSearching(false)
    }
  }

  const getTypeColor = (type: string) => {
    switch (type) {
      case "research":
        return "bg-blue-100 text-blue-800"
      case "clinical":
        return "bg-green-100 text-green-800"
      case "guideline":
        return "bg-purple-100 text-purple-800"
      case "textbook":
        return "bg-orange-100 text-orange-800"
      default:
        return "bg-gray-100 text-gray-800"
    }
  }

  return (
    <div className="space-y-6">
      <AdvancedSearch onSearch={handleAdvancedSearch} isSearching={isSearching} />

      {/* Search Results */}
      {results.length > 0 && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold">
              Semantic Search Results
              {searchQuery && <span className="text-sm font-normal text-gray-500 ml-2">for "{searchQuery}"</span>}
            </h3>
            <Badge variant="secondary">{results.length} matches found</Badge>
          </div>

          {results.map((result) => (
            <Card key={result.id} className="hover:shadow-md transition-shadow">
              <CardContent className="pt-6">
                <div className="space-y-4">
                  <div className="flex items-start justify-between">
                    <h4 className="font-semibold text-gray-900 leading-tight flex-1">{result.title}</h4>
                    <div className="flex items-center gap-2 ml-4">
                      <Badge className={getTypeColor(result.metadata.type)}>{result.metadata.type}</Badge>
                      <Badge variant="outline" className="text-xs">
                        {Math.round(result.similarity * 100)}% match
                      </Badge>
                    </div>
                  </div>

                  {/* Relevant Passages */}
                  {result.relevantPassages.length > 0 && (
                    <div className="bg-blue-50 p-3 rounded-lg">
                      <h5 className="text-sm font-medium text-blue-900 mb-2">Relevant Passages:</h5>
                      {result.relevantPassages.map((passage, idx) => (
                        <p key={idx} className="text-sm text-blue-800 italic">
                          "...{passage.trim()}..."
                        </p>
                      ))}
                    </div>
                  )}

                  <p className="text-gray-700 text-sm leading-relaxed">{result.content}</p>

                  {/* Keywords */}
                  <div className="flex flex-wrap gap-1">
                    {result.metadata.keywords.map((keyword, idx) => (
                      <Badge key={idx} variant="secondary" className="text-xs">
                        {keyword}
                      </Badge>
                    ))}
                  </div>

                  <div className="flex items-center justify-between pt-4 border-t text-xs text-gray-500">
                    <div className="flex items-center gap-4">
                      <span>Source: {result.source}</span>
                      <span className="flex items-center gap-1">
                        <Calendar className="h-3 w-3" />
                        {result.metadata.date}
                      </span>
                      {result.metadata.authors && (
                        <span className="flex items-center gap-1">
                          <Users className="h-3 w-3" />
                          {result.metadata.authors.slice(0, 2).join(", ")}
                        </span>
                      )}
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-16 bg-gray-200 rounded-full h-1">
                        <div
                          className="bg-cyan-600 h-1 rounded-full transition-all duration-300"
                          style={{ width: `${result.similarity * 100}%` }}
                        />
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {/* Search Tips */}
      <Card className="bg-blue-50 border-blue-200">
        <CardContent className="pt-6">
          <h4 className="font-medium text-blue-900 mb-2 flex items-center gap-2">
            <Brain className="h-4 w-4" />
            Advanced Search Tips
          </h4>
          <ul className="text-sm text-blue-800 space-y-1">
            <li>• Use natural language to describe medical concepts or symptoms</li>
            <li>• Filter by document type (research, clinical, guidelines) for targeted results</li>
            <li>• Adjust similarity threshold to control result precision vs. recall</li>
            <li>• Search by author names or publication date ranges</li>
            <li>• Results show relevant passages highlighted from source documents</li>
          </ul>
        </CardContent>
      </Card>
    </div>
  )
}
