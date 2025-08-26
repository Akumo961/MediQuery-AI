"use client"

import type React from "react"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Camera, Upload, Loader2, Eye, AlertTriangle } from "lucide-react"

interface AnalysisResult {
  findings: string[]
  confidence: number
  recommendations: string[]
  urgency: "low" | "medium" | "high"
}

export function MedicalImaging() {
  const [image, setImage] = useState<string | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [results, setResults] = useState<AnalysisResult | null>(null)

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    const reader = new FileReader()
    reader.onload = (e) => {
      setImage(e.target?.result as string)
      setResults(null)
    }
    reader.readAsDataURL(file)
  }

  const analyzeImage = async () => {
    if (!image) return

    setIsAnalyzing(true)

    try {
      const response = await fetch("/api/medical-imaging", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image }),
      })

      const result = await response.json()
      setResults(result)
    } catch (error) {
      console.error("Error analyzing image:", error)
    } finally {
      setIsAnalyzing(false)
    }
  }

  const getUrgencyColor = (urgency: string) => {
    switch (urgency) {
      case "high":
        return "bg-red-100 text-red-800"
      case "medium":
        return "bg-yellow-100 text-yellow-800"
      case "low":
        return "bg-green-100 text-green-800"
      default:
        return "bg-gray-100 text-gray-800"
    }
  }

  return (
    <div className="space-y-6">
      {/* Image Upload */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Camera className="h-5 w-5" />
            Medical Image Analysis
          </CardTitle>
          <CardDescription>
            Upload X-rays, CT scans, MRIs, or other medical images for AI-powered analysis
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center">
              <input type="file" accept="image/*" onChange={handleImageUpload} className="hidden" id="image-upload" />
              <label htmlFor="image-upload" className="cursor-pointer">
                {image ? (
                  <div className="space-y-4">
                    <img
                      src={image || "/placeholder.svg"}
                      alt="Medical scan"
                      className="max-w-full max-h-64 mx-auto rounded-lg"
                    />
                    <p className="text-sm text-gray-500">Click to change image</p>
                  </div>
                ) : (
                  <div>
                    <Upload className="h-12 w-12 mx-auto mb-4 text-gray-400" />
                    <p className="text-lg font-medium">Upload Medical Image</p>
                    <p className="text-sm text-gray-500">JPEG, PNG, DICOM up to 50MB</p>
                  </div>
                )}
              </label>
            </div>

            {image && (
              <Button onClick={analyzeImage} disabled={isAnalyzing} className="w-full">
                {isAnalyzing ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Analyzing Image...
                  </>
                ) : (
                  <>
                    <Eye className="mr-2 h-4 w-4" />
                    Analyze Image
                  </>
                )}
              </Button>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Analysis Results */}
      {results && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <span>Analysis Results</span>
              <Badge className={getUrgencyColor(results.urgency)}>{results.urgency.toUpperCase()} PRIORITY</Badge>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Findings */}
            <div>
              <h4 className="font-medium text-gray-900 mb-3 flex items-center gap-2">
                <AlertTriangle className="h-4 w-4" />
                Key Findings
              </h4>
              <ul className="space-y-2">
                {results.findings.map((finding, index) => (
                  <li key={index} className="flex items-start gap-2">
                    <span className="w-2 h-2 bg-cyan-600 rounded-full mt-2 flex-shrink-0" />
                    <span className="text-gray-700">{finding}</span>
                  </li>
                ))}
              </ul>
            </div>

            {/* Recommendations */}
            <div>
              <h4 className="font-medium text-gray-900 mb-3">Recommendations</h4>
              <ul className="space-y-2">
                {results.recommendations.map((rec, index) => (
                  <li key={index} className="flex items-start gap-2">
                    <span className="w-2 h-2 bg-emerald-600 rounded-full mt-2 flex-shrink-0" />
                    <span className="text-gray-700">{rec}</span>
                  </li>
                ))}
              </ul>
            </div>

            {/* Confidence */}
            <div className="bg-gray-50 p-4 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium">Analysis Confidence</span>
                <span className="text-sm font-bold">{Math.round(results.confidence * 100)}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className="bg-cyan-600 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${results.confidence * 100}%` }}
                />
              </div>
            </div>

            <div className="text-xs text-gray-500 bg-yellow-50 p-3 rounded-lg">
              <strong>Disclaimer:</strong> This AI analysis is for research and educational purposes only. Always
              consult with qualified medical professionals for diagnosis and treatment decisions.
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
