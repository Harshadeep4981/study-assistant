const DEFAULT_BASE_URL = 'http://localhost:8000'

export class ApiError extends Error {
  constructor(message, status = null, payload = null, options = {}) {
    super(message, options)
    this.name = 'ApiError'
    this.status = status
    this.payload = payload
  }
}

export function getApiBaseUrl() {
  return (import.meta.env.VITE_API_BASE_URL || DEFAULT_BASE_URL).replace(/\/$/, '')
}

function rethrowCancellation(error, signal) {
  if (error?.name === 'AbortError') throw error
  if (signal?.aborted) throw new DOMException('The request was aborted.', 'AbortError')
}

export async function request(path, { body, signal, headers, ...options } = {}) {
  let response
  try {
    response = await fetch(`${getApiBaseUrl()}${path}`, {
      ...options,
      body,
      signal,
      headers: { ...headers },
    })
  } catch (error) {
    rethrowCancellation(error, signal)
    throw new ApiError(
      'Unable to connect to the Study Assistant server. Check your connection and try again.',
      null,
      null,
      { cause: error },
    )
  }

  let payload
  try {
    payload = await response.json()
  } catch (error) {
    rethrowCancellation(error, signal)
    payload = null
  }

  if (!response.ok) {
    const detail = payload?.detail
    const message = response.status >= 500
      ? 'The Study Assistant server could not complete the request. Please try again.'
      : typeof detail === 'string'
        ? detail
        : `Request failed (${response.status})`
    throw new ApiError(message, response.status, payload)
  }

  return payload
}

export function postJson(path, value, { signal } = {}) {
  return request(path, {
    method: 'POST',
    body: JSON.stringify(value),
    signal,
    headers: { 'Content-Type': 'application/json' },
  })
}
