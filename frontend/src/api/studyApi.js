import { postJson, request } from './client.js'
export function ingestText(text, { signal } = {}) { return postJson('/api/documents/ingest', { text }, { signal }) }
export function ingestFile(file, { signal } = {}) { const body = new FormData(); body.append('file', file); return request('/api/documents/ingest', { method: 'POST', body, signal }) }
export function generateFlashcards(text, count = 5, { signal } = {}) { return postJson('/api/generate/flashcards', { text, count }, { signal }) }
export function generateQuiz(text, count = 5, { signal } = {}) { return postJson('/api/generate/quiz', { text, count }, { signal }) }
