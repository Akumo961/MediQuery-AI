import { type NextRequest, NextResponse } from "next/server"
import { MedicalImageAPI } from "@/lib/external-apis"

export async function POST(request: NextRequest) {
  try {
    const { image } = await request.json()

    const imageAPI = new MedicalImageAPI()
    const analysis = await imageAPI.analyzeImage(image)

    return NextResponse.json(analysis)
  } catch (error) {
    console.error("Medical imaging error:", error)
    return NextResponse.json({ error: "Failed to analyze medical image" }, { status: 500 })
  }
}
