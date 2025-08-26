import { type NextRequest, NextResponse } from "next/server"

export async function POST(request: NextRequest) {
  try {
    const { question, fileName } = await request.json()

    // Simulate AI processing with realistic medical responses
    const mockResponses = [
      {
        question,
        answer:
          "Based on the document analysis, the key findings indicate that the treatment shows significant efficacy in reducing symptoms by 65% compared to placebo. The study demonstrates a favorable safety profile with minimal adverse events reported.",
        confidence: 0.87,
        sources: ["Section 3.2", "Table 4", "Discussion"],
      },
      {
        question,
        answer:
          "The document suggests that the recommended dosage is 10mg twice daily for adults, with dose adjustments needed for patients with renal impairment. Monitoring of liver function is recommended during the first 3 months of treatment.",
        confidence: 0.92,
        sources: ["Dosing Guidelines", "Safety Monitoring"],
      },
      {
        question,
        answer:
          "According to the clinical trial data, common side effects include mild headache (12% of patients), nausea (8%), and dizziness (5%). Serious adverse events were rare, occurring in less than 1% of participants.",
        confidence: 0.89,
        sources: ["Adverse Events Table", "Safety Analysis"],
      },
    ]

    // Return a random mock response
    const response = mockResponses[Math.floor(Math.random() * mockResponses.length)]

    // Simulate processing delay
    await new Promise((resolve) => setTimeout(resolve, 2000))

    return NextResponse.json(response)
  } catch (error) {
    console.error("Document QA error:", error)
    return NextResponse.json({ error: "Failed to process document question" }, { status: 500 })
  }
}
