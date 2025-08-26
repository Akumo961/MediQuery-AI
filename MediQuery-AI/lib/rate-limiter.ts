// Rate limiting utility for external API calls

interface RateLimitConfig {
  windowMs: number
  maxRequests: number
}

class RateLimiter {
  private requests: Map<string, number[]> = new Map()

  constructor(private config: RateLimitConfig) {}

  isAllowed(identifier: string): boolean {
    const now = Date.now()
    const windowStart = now - this.config.windowMs

    // Get existing requests for this identifier
    const userRequests = this.requests.get(identifier) || []

    // Filter out requests outside the current window
    const recentRequests = userRequests.filter((timestamp) => timestamp > windowStart)

    // Check if under the limit
    if (recentRequests.length >= this.config.maxRequests) {
      return false
    }

    // Add current request
    recentRequests.push(now)
    this.requests.set(identifier, recentRequests)

    return true
  }

  getRemainingRequests(identifier: string): number {
    const now = Date.now()
    const windowStart = now - this.config.windowMs
    const userRequests = this.requests.get(identifier) || []
    const recentRequests = userRequests.filter((timestamp) => timestamp > windowStart)

    return Math.max(0, this.config.maxRequests - recentRequests.length)
  }
}

// Export rate limiters for different APIs
export const pubmedLimiter = new RateLimiter({
  windowMs: 60 * 1000, // 1 minute
  maxRequests: 10, // 10 requests per minute
})

export const clinicalTrialsLimiter = new RateLimiter({
  windowMs: 60 * 1000, // 1 minute
  maxRequests: 20, // 20 requests per minute
})

export const imagingLimiter = new RateLimiter({
  windowMs: 60 * 1000, // 1 minute
  maxRequests: 5, // 5 image analyses per minute
})
