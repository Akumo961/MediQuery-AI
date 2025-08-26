"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Checkbox } from "@/components/ui/checkbox"
import { Slider } from "@/components/ui/slider"
import { Settings, Search, Filter } from "lucide-react"

interface AdvancedSearchProps {
  onSearch: (query: string, filters: any) => void
  isSearching: boolean
}

export function AdvancedSearch({ onSearch, isSearching }: AdvancedSearchProps) {
  const [query, setQuery] = useState("")
  const [showFilters, setShowFilters] = useState(false)
  const [filters, setFilters] = useState({
    types: [] as string[],
    dateRange: { start: "", end: "" },
    authors: "",
    similarity: [0.7],
  })

  const handleSearch = () => {
    const searchFilters = {
      type: filters.types.length > 0 ? filters.types : undefined,
      dateRange: filters.dateRange.start && filters.dateRange.end ? filters.dateRange : undefined,
      authors: filters.authors ? [filters.authors] : undefined,
    }

    onSearch(query, {
      filters: searchFilters,
      threshold: filters.similarity[0],
      limit: 20,
    })
  }

  const handleTypeChange = (type: string, checked: boolean) => {
    setFilters((prev) => ({
      ...prev,
      types: checked ? [...prev.types, type] : prev.types.filter((t) => t !== type),
    }))
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Search className="h-5 w-5" />
          Advanced Semantic Search
        </CardTitle>
        <CardDescription>Search medical literature using AI-powered semantic understanding</CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Main Search */}
        <div className="space-y-2">
          <Label htmlFor="search-query">Search Query</Label>
          <div className="flex gap-2">
            <Input
              id="search-query"
              placeholder="Describe symptoms, conditions, or medical concepts..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              className="flex-1"
            />
            <Button variant="outline" onClick={() => setShowFilters(!showFilters)} className="px-3">
              <Filter className="h-4 w-4" />
            </Button>
            <Button onClick={handleSearch} disabled={!query.trim() || isSearching}>
              {isSearching ? "Searching..." : "Search"}
            </Button>
          </div>
        </div>

        {/* Advanced Filters */}
        {showFilters && (
          <div className="space-y-4 p-4 bg-gray-50 rounded-lg">
            <div className="flex items-center gap-2 mb-4">
              <Settings className="h-4 w-4" />
              <span className="font-medium">Search Filters</span>
            </div>

            {/* Document Types */}
            <div className="space-y-2">
              <Label>Document Types</Label>
              <div className="grid grid-cols-2 gap-2">
                {["research", "clinical", "guideline", "textbook"].map((type) => (
                  <div key={type} className="flex items-center space-x-2">
                    <Checkbox
                      id={type}
                      checked={filters.types.includes(type)}
                      onCheckedChange={(checked) => handleTypeChange(type, checked as boolean)}
                    />
                    <Label htmlFor={type} className="capitalize text-sm">
                      {type}
                    </Label>
                  </div>
                ))}
              </div>
            </div>

            {/* Date Range */}
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="start-date">Start Date</Label>
                <Input
                  id="start-date"
                  type="date"
                  value={filters.dateRange.start}
                  onChange={(e) =>
                    setFilters((prev) => ({
                      ...prev,
                      dateRange: { ...prev.dateRange, start: e.target.value },
                    }))
                  }
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="end-date">End Date</Label>
                <Input
                  id="end-date"
                  type="date"
                  value={filters.dateRange.end}
                  onChange={(e) =>
                    setFilters((prev) => ({
                      ...prev,
                      dateRange: { ...prev.dateRange, end: e.target.value },
                    }))
                  }
                />
              </div>
            </div>

            {/* Authors */}
            <div className="space-y-2">
              <Label htmlFor="authors">Author Filter</Label>
              <Input
                id="authors"
                placeholder="Search by author name..."
                value={filters.authors}
                onChange={(e) => setFilters((prev) => ({ ...prev, authors: e.target.value }))}
              />
            </div>

            {/* Similarity Threshold */}
            <div className="space-y-2">
              <Label>Similarity Threshold: {Math.round(filters.similarity[0] * 100)}%</Label>
              <Slider
                value={filters.similarity}
                onValueChange={(value) => setFilters((prev) => ({ ...prev, similarity: value }))}
                max={1}
                min={0.5}
                step={0.05}
                className="w-full"
              />
              <div className="flex justify-between text-xs text-gray-500">
                <span>50% (More Results)</span>
                <span>100% (Exact Match)</span>
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
