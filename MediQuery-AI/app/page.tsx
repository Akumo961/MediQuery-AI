"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { Search, FileText, ImageIcon, Database, Brain, Stethoscope, Activity } from "lucide-react"
import { DocumentQA } from "@/components/document-qa"
import { MedicalImaging } from "@/components/medical-imaging"
import { ResearchDatabase } from "@/components/research-database"
import { VectorSearch } from "@/components/vector-search"

export default function MediQueryAI() {
  const [activeFeature, setActiveFeature] = useState<string | null>(null)

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-card">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="flex items-center justify-center w-10 h-10 bg-primary rounded-lg">
                <Stethoscope className="w-6 h-6 text-primary-foreground" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-foreground font-heading">MediQuery AI</h1>
                <p className="text-sm text-muted-foreground">Healthcare Research Assistant</p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Badge variant="secondary" className="bg-secondary text-secondary-foreground">
                <Activity className="w-3 h-3 mr-1" />
                Active
              </Badge>
              {activeFeature && (
                <Button variant="outline" onClick={() => setActiveFeature(null)}>
                  Back to Dashboard
                </Button>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8">
        {!activeFeature ? (
          <>
            {/* Search Section */}
            <div className="mb-8">
              <div className="max-w-2xl mx-auto text-center mb-6">
                <h2 className="text-3xl font-bold text-foreground mb-3 font-heading">AI-Powered Medical Research</h2>
                <p className="text-muted-foreground text-lg">
                  Search medical literature, analyze clinical images, and get instant answers to your healthcare
                  research questions.
                </p>
              </div>

              <div className="max-w-3xl mx-auto">
                <div className="relative">
                  <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 text-muted-foreground w-5 h-5" />
                  <Input
                    placeholder="Ask a medical question, search literature, or describe symptoms..."
                    className="pl-12 pr-4 py-6 text-lg bg-card border-border focus:ring-primary"
                  />
                  <Button className="absolute right-2 top-1/2 transform -translate-y-1/2 bg-primary hover:bg-primary/90">
                    <Brain className="w-4 h-4 mr-2" />
                    Analyze
                  </Button>
                </div>
              </div>
            </div>

            {/* Feature Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
              <Card className="border-border hover:shadow-lg transition-shadow">
                <CardHeader>
                  <div className="flex items-center gap-3">
                    <div className="p-2 bg-primary/10 rounded-lg">
                      <FileText className="w-6 h-6 text-primary" />
                    </div>
                    <div>
                      <CardTitle className="font-heading">Document Analysis</CardTitle>
                      <CardDescription>Search and analyze medical literature</CardDescription>
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground mb-4">
                    Extract insights from research papers, clinical trials, and medical documents using advanced NLP.
                  </p>
                  <Button
                    variant="outline"
                    className="w-full bg-transparent"
                    onClick={() => setActiveFeature("document-qa")}
                  >
                    Upload Documents
                  </Button>
                </CardContent>
              </Card>

              <Card className="border-border hover:shadow-lg transition-shadow">
                <CardHeader>
                  <div className="flex items-center gap-3">
                    <div className="p-2 bg-secondary/10 rounded-lg">
                      <ImageIcon className="w-6 h-6 text-secondary" />
                    </div>
                    <div>
                      <CardTitle className="font-heading">Medical Imaging</CardTitle>
                      <CardDescription>AI-powered image analysis</CardDescription>
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground mb-4">
                    Analyze X-rays, MRIs, CT scans, and other medical images for anomaly detection and diagnosis
                    support.
                  </p>
                  <Button
                    variant="outline"
                    className="w-full bg-transparent"
                    onClick={() => setActiveFeature("medical-imaging")}
                  >
                    Upload Images
                  </Button>
                </CardContent>
              </Card>

              <Card className="border-border hover:shadow-lg transition-shadow">
                <CardHeader>
                  <div className="flex items-center gap-3">
                    <div className="p-2 bg-accent/10 rounded-lg">
                      <Database className="w-6 h-6 text-accent" />
                    </div>
                    <div>
                      <CardTitle className="font-heading">Research Database</CardTitle>
                      <CardDescription>Access medical databases</CardDescription>
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground mb-4">
                    Search PubMed, ClinicalTrials.gov, and other medical databases with semantic similarity matching.
                  </p>
                  <Button
                    variant="outline"
                    className="w-full bg-transparent"
                    onClick={() => setActiveFeature("research-database")}
                  >
                    Search Database
                  </Button>
                </CardContent>
              </Card>
            </div>

            {/* Quick Actions */}
            <div className="bg-card border border-border rounded-lg p-6">
              <h3 className="text-xl font-semibold mb-4 font-heading">Quick Actions</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <Button
                  variant="ghost"
                  className="h-auto p-4 flex flex-col items-center gap-2"
                  onClick={() => setActiveFeature("document-qa")}
                >
                  <FileText className="w-8 h-8 text-primary" />
                  <span className="text-sm">Literature Review</span>
                </Button>
                <Button
                  variant="ghost"
                  className="h-auto p-4 flex flex-col items-center gap-2"
                  onClick={() => setActiveFeature("medical-imaging")}
                >
                  <ImageIcon className="w-8 h-8 text-secondary" />
                  <span className="text-sm">Image Analysis</span>
                </Button>
                <Button
                  variant="ghost"
                  className="h-auto p-4 flex flex-col items-center gap-2"
                  onClick={() => setActiveFeature("vector-search")}
                >
                  <Brain className="w-8 h-8 text-accent" />
                  <span className="text-sm">Semantic Search</span>
                </Button>
                <Button
                  variant="ghost"
                  className="h-auto p-4 flex flex-col items-center gap-2"
                  onClick={() => setActiveFeature("research-database")}
                >
                  <Database className="w-8 h-8 text-primary" />
                  <span className="text-sm">Clinical Trials</span>
                </Button>
              </div>
            </div>
          </>
        ) : (
          /* Added feature-specific content rendering */
          <div>
            {activeFeature === "document-qa" && <DocumentQA />}
            {activeFeature === "medical-imaging" && <MedicalImaging />}
            {activeFeature === "research-database" && <ResearchDatabase />}
            {activeFeature === "vector-search" && <VectorSearch />}
          </div>
        )}
      </main>
    </div>
  )
}
