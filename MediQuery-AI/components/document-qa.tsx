"use client"

import type React from "react"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Textarea } from "@/components/ui/textarea"
import { Upload, FileText, MessageSquare, Loader2 } from "lucide-react"

interface QAResult {
  question: string
  answer: string
  confidence: number
  sources: string[]
}

export function DocumentQA() {
  const [file, setFile] = useState<File | null>(null)
  const [question, setQuestion] = useState("")
  const [results, setResults] = useState<QAResult[]>([])
  const [isProcessing, setIsProcessing] = useState(false)
  const [isUploading, setIsUploading] = useState(false)

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files?.[0]
    if (!selectedFile) return

    setIsUploading(true)
    setFile(selectedFile)

    // Simulate file processing
    setTimeout(() => {
      setIsUploading(false)
    }, 2000)
  }

  const handleQuestionSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!question.trim() || !file) return

    setIsProcessing(true)

    try {
      const response = await fetch("/api/document-qa", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question,
          fileName: file.name,
        }),
      })

      const result = await response.json()

      setResults((prev) => [result, ...prev])
      setQuestion("")
    } catch (error) {
      console.error("Error processing question:", error)
    } finally {
      setIsProcessing(false)
    }
  }

  return (
    <div className="space-y-6">
      {/* File Upload */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Upload className="h-5 w-5" />
            Document Upload
          </CardTitle>
          <CardDescription>Upload medical documents, research papers, or clinical reports for analysis</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center">
            <input
              type="file"
              accept=".pdf,.doc,.docx,.txt"
              onChange={handleFileUpload}
              className="hidden"
              id="file-upload"
            />
            <label htmlFor="file-upload" className="cursor-pointer">
              {isUploading ? (
                <div className="flex items-center justify-center gap-2">
                  <Loader2 className="h-6 w-6 animate-spin" />
                  <span>Processing document...</span>
                </div>
              ) : file ? (
                <div className="flex items-center justify-center gap-2 text-emerald-600">
                  <FileText className="h-6 w-6" />
                  <span>{file.name}</span>
                </div>
              ) : (
                <div>
                  <Upload className="h-12 w-12 mx-auto mb-4 text-gray-400" />
                  <p className="text-lg font-medium">Click to upload document</p>
                  <p className="text-sm text-gray-500">PDF, DOC, DOCX, TXT up to 10MB</p>
                </div>
              )}
            </label>
          </div>
        </CardContent>
      </Card>

      {/* Question Input */}
      {file && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <MessageSquare className="h-5 w-5" />
              Ask Questions
            </CardTitle>
            <CardDescription>Ask specific questions about the uploaded document</CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleQuestionSubmit} className="space-y-4">
              <Textarea
                placeholder="What are the key findings in this study? What are the side effects mentioned? What is the recommended dosage?"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                className="min-h-[100px]"
              />
              <Button type="submit" disabled={!question.trim() || isProcessing} className="w-full">
                {isProcessing ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  "Ask Question"
                )}
              </Button>
            </form>
          </CardContent>
        </Card>
      )}

      {/* Results */}
      {results.length > 0 && (
        <div className="space-y-4">
          <h3 className="text-lg font-semibold">Q&A Results</h3>
          {results.map((result, index) => (
            <Card key={index}>
              <CardContent className="pt-6">
                <div className="space-y-4">
                  <div>
                    <h4 className="font-medium text-gray-900 mb-2">Question:</h4>
                    <p className="text-gray-700">{result.question}</p>
                  </div>
                  <div>
                    <h4 className="font-medium text-gray-900 mb-2">Answer:</h4>
                    <p className="text-gray-700">{result.answer}</p>
                  </div>
                  <div className="flex items-center justify-between text-sm text-gray-500">
                    <span>Confidence: {Math.round(result.confidence * 100)}%</span>
                    <span>Sources: {result.sources.join(", ")}</span>
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
