"use client"

import type React from "react"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { Search, ExternalLink, Calendar, Users, Loader2 } from "lucide-react"

interface ResearchResult {
  id: string
  title: string
  authors: string[]
  journal: string
  publishDate: string
  abstract: string
  url: string
  citationCount: number
  relevanceScore: number
}

export function ResearchDatabase() {
  const [query, setQuery] = useState("")
  const [results, setResults] = useState<ResearchResult[]>([])
  const [isSearching, setIsSearching] = useState(false)
  const [searchType, setSearchType] = useState<"pubmed" | "clinical">("pubmed")

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!query.trim()) return

    setIsSearching(true)

    try {
      const response = await fetch("/api/research-search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, type: searchType }),
      })

      const data = await response.json()
      setResults(data.results)
    } catch (error) {
      console.error("Error searching:", error)
    } finally {
      setIsSearching(false)
    }
  }

  return (
    <div className="space-y-6">
      {/* Search Interface */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Search className="h-5 w-5" />
            Research Database Search
          </CardTitle>
          <CardDescription>Search PubMed, ClinicalTrials.gov, and other medical databases</CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSearch} className="space-y-4">
            <div className="flex gap-2 mb-4">
              <Button
                type="button"
                variant={searchType === "pubmed" ? "default" : "outline"}
                onClick={() => setSearchType("pubmed")}
                size="sm"
              >
                PubMed
              </Button>
              <Button
                type="button"
                variant={searchType === "clinical" ? "default" : "outline"}
                onClick={() => setSearchType("clinical")}
                size="sm"
              >
                Clinical Trials
              </Button>
            </div>

            <div className="flex gap-2">
              <Input
                placeholder="Search for medical research, drug studies, clinical trials..."
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                className="flex-1"
              />
              <Button type="submit" disabled={isSearching}>
                {isSearching ? <Loader2 className="h-4 w-4 animate-spin" /> : <Search className="h-4 w-4" />}
              </Button>
            </div>
          </form>
        </CardContent>
      </Card>

      {/* Search Results */}
      {results.length > 0 && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold">Search Results</h3>
            <Badge variant="secondary">{results.length} results found</Badge>
          </div>

          {results.map((result) => (
            <Card key={result.id} className="hover:shadow-md transition-shadow">
              <CardContent className="pt-6">
                <div className="space-y-4">
                  <div>
                    <h4 className="font-semibold text-gray-900 mb-2 leading-tight">{result.title}</h4>
                    <div className="flex items-center gap-4 text-sm text-gray-600 mb-3">
                      <span className="flex items-center gap-1">
                        <Users className="h-3 w-3" />
                        {result.authors.slice(0, 3).join(", ")}
                        {result.authors.length > 3 && ` +${result.authors.length - 3} more`}
                      </span>
                      <span className="flex items-center gap-1">
                        <Calendar className="h-3 w-3" />
                        {result.publishDate}
                      </span>
                    </div>
                  </div>

                  <p className="text-gray-700 text-sm leading-relaxed">{result.abstract}</p>

                  <div className="flex items-center justify-between pt-4 border-t">
                    <div className="flex items-center gap-4">
                      <Badge variant="outline">{result.journal}</Badge>
                      <span className="text-xs text-gray-500">{result.citationCount} citations</span>
                      <Badge variant="secondary" className="text-xs">
                        {Math.round(result.relevanceScore * 100)}% relevant
                      </Badge>
                    </div>
                    <Button variant="outline" size="sm" asChild>
                      <a href={result.url} target="_blank" rel="noopener noreferrer">
                        <ExternalLink className="h-3 w-3 mr-1" />
                        View Full Text
                      </a>
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  )
}
