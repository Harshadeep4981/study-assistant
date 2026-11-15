import { cleanup, fireEvent, render, renderHook, screen, act, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import App from '../App.jsx'
import { ApiError, request } from '../api/client.js'
import { generateFlashcards, generateQuiz, ingestFile, ingestText } from '../api/studyApi.js'
import { isSupportedFile, useStudyComposer } from '../state/useStudyComposer.js'

afterEach(cleanup)

function uploadInput() {
  return screen.getByRole('tabpanel').querySelector('input[type="file"]')
}

function ingestionResponse(overrides = {}) {
  return {
    extracted_text: 'Normalized material from backend.',
    source_type: 'pasted_text',
    metadata: { filename: null, character_count: 32, word_count: 4 },
    warnings: [],
    ...overrides,
  }
}

function flashcardResponse(cards = [
  { question: 'What is photosynthesis?', answer: 'The conversion of light energy into chemical energy.' },
  { question: 'Where is chlorophyll found?', answer: 'In chloroplasts.' },
]) {
  return { flashcards: cards, count: cards.length }
}

function quizQuestion(question = 'Which planet is known as the Red Planet?', overrides = {}) {
  return {
    question,
    options: ['Mercury', 'Venus', 'Earth', 'Mars'],
    correct_answer: 'Mars',
    explanation: 'Mars appears red because iron minerals in its soil oxidize.',
    ...overrides,
  }
}

function quizResponse(questions = [
  quizQuestion(),
  quizQuestion('Which planet is closest to the Sun?', { correct_answer: 'Mercury', explanation: 'Mercury orbits closest to the Sun.' }),
]) {
  return { quiz: questions, count: questions.length }
}

function jsonResponse(payload, { ok = true, status = 200 } = {}) {
  return { ok, status, json: async () => payload }
}

async function createStudySetInApp(user, notes = 'Photosynthesis study notes') {
  await user.type(screen.getByRole('textbox', { name: /study notes or topic/i }), notes)
  await user.click(screen.getByRole('button', { name: /create study set/i }))
  await screen.findByRole('heading', { name: 'Ready to make it stick?' })
}

async function openFlashcardsInApp(user) {
  await user.click(screen.getByRole('button', { name: /start learning/i }))
}

async function openQuizInApp(user) {
  await user.click(screen.getByRole('button', { name: /take a quiz/i }))
}

describe('Studywise creation screen', () => {
  it('renders a centered creation screen without the old mode selector or preview', () => {
    render(<App />)
    expect(screen.getByRole('heading', { name: 'Turn what you study into what you know.' })).toBeInTheDocument()
    expect(screen.queryByRole('textbox', { name: /study set title/i })).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: /create study set/i })).toBeInTheDocument()
    expect(screen.queryByRole('group', { name: /choose a study mode/i })).not.toBeInTheDocument()
    expect(screen.queryByText(/your study set will appear here/i)).not.toBeInTheDocument()
  })

  it('switches input tabs while preserving pasted notes', async () => {
    const user = userEvent.setup()
    render(<App />)
    const pasteTab = screen.getByRole('tab', { name: 'Paste notes' })
    const uploadTab = screen.getByRole('tab', { name: 'Upload document' })
    await user.type(screen.getByRole('textbox', { name: /study notes or topic/i }), 'Cell biology notes')
    await user.click(uploadTab)
    expect(uploadTab).toHaveAttribute('aria-selected', 'true')
    await user.click(pasteTab)
    expect(screen.getByRole('textbox', { name: /study notes or topic/i })).toHaveValue('Cell biology notes')
  })

  it('preserves the selected file when switching to paste and back', async () => {
    const user = userEvent.setup()
    render(<App />)
    await user.click(screen.getByRole('tab', { name: 'Upload document' }))
    await user.upload(uploadInput(), new File(['lesson'], 'lesson.txt', { type: 'text/plain' }))
    await user.click(screen.getByRole('tab', { name: 'Paste notes' }))
    await user.click(screen.getByRole('tab', { name: 'Upload document' }))
    expect(screen.getByText('lesson.txt')).toBeInTheDocument()
  })

  it('rejects unsupported file extensions with useful validation', () => {
    render(<App />)
    fireEvent.click(screen.getByRole('tab', { name: 'Upload document' }))
    fireEvent.change(uploadInput(), { target: { files: [new File(['image'], 'photo.png', { type: 'image/png' })] } })
    expect(screen.getByRole('status')).toHaveTextContent(/not supported.*txt or pdf/i)
    expect(screen.queryByText('photo.png')).not.toBeInTheDocument()
  })

  it('shows the 100-page PDF limit before a document is selected', async () => {
    const user = userEvent.setup()
    render(<App />)
    await user.click(screen.getByRole('tab', { name: 'Upload document' }))
    expect(screen.getByText(/pdfs up to 100 pages/i)).toBeInTheDocument()
  })

  it('removes a selected file', async () => {
    const user = userEvent.setup()
    render(<App />)
    await user.click(screen.getByRole('tab', { name: 'Upload document' }))
    await user.upload(uploadInput(), new File(['notes'], 'biology.txt', { type: 'text/plain' }))
    await user.click(screen.getByRole('button', { name: /remove biology.txt/i }))
    expect(screen.getByText(/drop your document here/i)).toBeInTheDocument()
  })

  it('replaces a selected file and shows the new filename and size', async () => {
    const user = userEvent.setup()
    render(<App />)
    await user.click(screen.getByRole('tab', { name: 'Upload document' }))
    await user.upload(uploadInput(), new File(['old'], 'old.txt', { type: 'text/plain' }))
    await user.click(screen.getByRole('button', { name: 'Replace' }))
    await user.upload(uploadInput(), new File(['new content'], 'new.pdf', { type: 'application/pdf' }))
    expect(screen.getByText('new.pdf')).toBeInTheDocument()
    expect(screen.getByText('11 B')).toBeInTheDocument()
    expect(screen.queryByText('old.txt')).not.toBeInTheDocument()
  })

  it('updates the character count and can clear pasted notes', async () => {
    const user = userEvent.setup()
    render(<App />)
    const textarea = screen.getByRole('textbox', { name: /study notes or topic/i })
    await user.type(textarea, 'Biology')
    expect(screen.getByText('7 characters')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: /clear notes/i }))
    expect(textarea).toHaveValue('')
    expect(screen.getByText('0 characters')).toBeInTheDocument()
  })

  it('uses only the active source while preserving the inactive source', () => {
    const { result } = renderHook(() => useStudyComposer())
    const file = new File(['file notes'], 'notes.txt', { type: 'text/plain' })
    act(() => { result.current.updateText('pasted notes'); result.current.chooseFile(file) })
    expect(result.current.activeSource).toEqual({ type: 'text', text: 'pasted notes' })
    act(() => result.current.changeInputMode('upload'))
    expect(result.current.activeSource).toEqual({ type: 'file', file })
    act(() => result.current.changeInputMode('paste'))
    expect(result.current.activeSource).toEqual({ type: 'text', text: 'pasted notes' })
  })

  it('supports keyboard navigation between input tabs', async () => {
    const user = userEvent.setup()
    render(<App />)
    const pasteTab = screen.getByRole('tab', { name: 'Paste notes' })
    const uploadTab = screen.getByRole('tab', { name: 'Upload document' })
    pasteTab.focus()
    await user.keyboard('{ArrowRight}')
    expect(uploadTab).toHaveFocus()
    expect(uploadTab).toHaveAttribute('aria-selected', 'true')
    await user.keyboard('{ArrowLeft}')
    expect(pasteTab).toHaveFocus()
    expect(pasteTab).toHaveAttribute('aria-selected', 'true')
  })

  it('allows keyboard activation of the browse-files control', async () => {
    const user = userEvent.setup()
    render(<App />)
    await user.click(screen.getByRole('tab', { name: 'Upload document' }))
    const input = uploadInput()
    const clickSpy = vi.spyOn(input, 'click')
    const browseButton = screen.getByRole('button', { name: 'Browse files' })
    browseButton.focus()
    expect(browseButton).toHaveFocus()
    await user.keyboard('{Enter}')
    expect(clickSpy).toHaveBeenCalledOnce()
  })

  it('validates the active input source', async () => {
    const user = userEvent.setup()
    render(<App />)
    await user.click(screen.getByRole('button', { name: /create study set/i }))
    expect(screen.getByRole('status')).toHaveTextContent(/add notes or a topic/i)
  })
})

describe('study set ingestion and workspace', () => {
  it('sends only pasted text using the backend JSON contract and shows the returned workspace', async () => {
    const user = userEvent.setup()
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ingestionResponse() })
    global.fetch = fetchMock
    render(<App />)
    await user.type(screen.getByRole('textbox', { name: /study notes or topic/i }), 'Raw pasted notes')
    await user.click(screen.getByRole('button', { name: /create study set/i }))

    expect(fetchMock).toHaveBeenCalledOnce()
    const [url, options] = fetchMock.mock.calls[0]
    expect(url).toBe('http://localhost:8000/api/documents/ingest')
    expect(JSON.parse(options.body)).toEqual({ text: 'Raw pasted notes' })
    expect(options.body).not.toContain('file')
    expect(await screen.findByRole('heading', { name: 'Ready to make it stick?' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Your study material' })).toBeInTheDocument()
    expect(screen.getByText(/material processed/i)).toBeInTheDocument()
    expect(screen.getByText(/pasted notes/i)).toBeInTheDocument()
    expect(screen.getByText(/32 characters/i)).toBeInTheDocument()
  })

  it('sends an uploaded document using multipart form data and stores backend filename and source type', async () => {
    const user = userEvent.setup()
    const file = new File(['plain study notes'], 'lesson.txt', { type: 'text/plain' })
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ingestionResponse({
      extracted_text: 'Backend extracted file text.', source_type: 'txt',
      metadata: { filename: 'lesson.txt', character_count: 29, word_count: 4 },
    }) })
    global.fetch = fetchMock
    render(<App />)
    await user.click(screen.getByRole('tab', { name: 'Upload document' }))
    await user.upload(uploadInput(), file)
    await user.click(screen.getByRole('button', { name: /create study set/i }))

    const [url, options] = fetchMock.mock.calls[0]
    expect(url).toBe('http://localhost:8000/api/documents/ingest')
    expect(options.body).toBeInstanceOf(FormData)
    expect(options.body.get('file')).toBe(file)
    expect(options.body.has('text')).toBe(false)
    expect(options.headers['Content-Type']).toBeUndefined()
    expect(await screen.findByText('lesson.txt')).toBeInTheDocument()
    expect(screen.getByText(/source · TXT/i)).toBeInTheDocument()
    expect(screen.getByText(/material processed/i)).toBeInTheDocument()
  })

  it('keeps the creation screen visible and shows backend validation errors', async () => {
    const user = userEvent.setup()
    global.fetch = vi.fn().mockResolvedValue({
      ok: false, status: 400,
      json: async () => ({ detail: 'Text input cannot be empty or contain only whitespace.' }),
    })
    render(<App />)
    await user.type(screen.getByRole('textbox', { name: /study notes or topic/i }), 'Notes')
    await user.click(screen.getByRole('button', { name: /create study set/i }))
    expect(await screen.findByRole('status')).toHaveTextContent(/text input cannot be empty/i)
    expect(screen.getByRole('heading', { name: 'Turn what you study into what you know.' })).toBeInTheDocument()
    expect(screen.queryByRole('heading', { name: 'Ready to make it stick?' })).not.toBeInTheDocument()
  })

  it('displays the backend PDF page-limit error clearly', async () => {
    const user = userEvent.setup()
    global.fetch = vi.fn().mockResolvedValue(jsonResponse({
      detail: 'The PDF contains 101 pages, which exceeds the maximum allowed limit of 100 pages.',
    }, { ok: false, status: 413 }))
    render(<App />)
    await user.click(screen.getByRole('tab', { name: 'Upload document' }))
    await user.upload(uploadInput(), new File(['pdf'], 'lecture.pdf', { type: 'application/pdf' }))
    await user.click(screen.getByRole('button', { name: /create study set/i }))
    expect(await screen.findByRole('status')).toHaveTextContent(/101 pages.*maximum allowed limit of 100 pages/i)
  })

  it('shows a safe network error and does not create a workspace', async () => {
    const user = userEvent.setup()
    global.fetch = vi.fn().mockRejectedValue(new TypeError('socket internals'))
    render(<App />)
    await user.type(screen.getByRole('textbox', { name: /study notes or topic/i }), 'Notes')
    await user.click(screen.getByRole('button', { name: /create study set/i }))
    expect(await screen.findByRole('status')).toHaveTextContent(/unable to connect to the study assistant server/i)
    expect(screen.queryByRole('heading', { name: 'Ready to make it stick?' })).not.toBeInTheDocument()
    expect(screen.queryByText(/socket internals/i)).not.toBeInTheDocument()
  })

  it('shows a loading state and prevents duplicate submissions', async () => {
    const user = userEvent.setup()
    let resolveFetch
    const fetchMock = vi.fn(() => new Promise((resolve) => { resolveFetch = resolve }))
    global.fetch = fetchMock
    render(<App />)
    await user.type(screen.getByRole('textbox', { name: /study notes or topic/i }), 'Notes')
    const submit = screen.getByRole('button', { name: /create study set/i })
    await user.click(submit)
    expect(screen.getByRole('button', { name: /preparing your study material/i })).toBeDisabled()
    expect(screen.getByRole('textbox', { name: /study notes or topic/i })).toBeDisabled()
    expect(screen.getByRole('tab', { name: /upload document/i })).toBeDisabled()
    expect(fetchMock).toHaveBeenCalledOnce()
    resolveFetch({ ok: true, json: async () => ingestionResponse() })
    expect(await screen.findByRole('heading', { name: 'Ready to make it stick?' })).toBeInTheDocument()
  })

  it('starts a new study set and clears the prior active study set and form content', async () => {
    const user = userEvent.setup()
    global.fetch = vi.fn().mockResolvedValue({ ok: true, json: async () => ingestionResponse() })
    render(<App />)
    await user.type(screen.getByRole('textbox', { name: /study notes or topic/i }), 'Old notes')
    await user.click(screen.getByRole('button', { name: /create study set/i }))
    await screen.findByRole('heading', { name: 'Ready to make it stick?' })
    await user.click(screen.getByRole('button', { name: /start new study set/i }))

    expect(screen.getByRole('heading', { name: 'Turn what you study into what you know.' })).toBeInTheDocument()
    expect(screen.getByRole('textbox', { name: /study notes or topic/i })).toHaveValue('')
    expect(screen.getByText('0 characters')).toBeInTheDocument()
  })
})

describe('AI flashcard learning flow', () => {
  it('sends the active extracted text to the existing API contract and displays validated cards', async () => {
    const user = userEvent.setup()
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(jsonResponse(ingestionResponse({ extracted_text: 'Backend-normalized study text.' })))
      .mockResolvedValueOnce(jsonResponse(flashcardResponse()))
    global.fetch = fetchMock
    render(<App />)
    await createStudySetInApp(user)
    expect(fetchMock).toHaveBeenCalledOnce()
    await openFlashcardsInApp(user)

    expect(await screen.findByRole('heading', { name: 'Flashcards' })).toBeInTheDocument()
    expect(screen.getByText('What is photosynthesis?')).toBeInTheDocument()
    expect(fetchMock).toHaveBeenCalledTimes(2)
    const [url, options] = fetchMock.mock.calls[1]
    expect(url).toBe('http://localhost:8000/api/generate/flashcards')
    expect(JSON.parse(options.body)).toEqual({ text: 'Backend-normalized study text.', count: 5 })
  })

  it('shows the generation loading state and prevents duplicate hook requests', async () => {
    let resolveGeneration
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(jsonResponse(ingestionResponse()))
      .mockImplementationOnce(() => new Promise((resolve) => { resolveGeneration = resolve }))
    global.fetch = fetchMock
    const { result } = renderHook(() => useStudyComposer())
    act(() => result.current.updateText('Notes for flashcard testing'))
    await act(async () => { await result.current.createStudySet() })

    let firstRequest
    let duplicateRequest
    act(() => {
      firstRequest = result.current.openFlashcards()
      duplicateRequest = result.current.openFlashcards()
    })
    await waitFor(() => expect(result.current.isGeneratingFlashcards).toBe(true))
    expect(fetchMock).toHaveBeenCalledTimes(2)
    resolveGeneration(jsonResponse(flashcardResponse()))
    await act(async () => { await firstRequest })
    await expect(duplicateRequest).resolves.toBe(false)
    expect(result.current.activeStudySet.flashcards).toHaveLength(2)
  })

  it('renders the dedicated flashcard loading message until generation completes', async () => {
    const user = userEvent.setup()
    let resolveGeneration
    global.fetch = vi.fn()
      .mockResolvedValueOnce(jsonResponse(ingestionResponse()))
      .mockImplementationOnce(() => new Promise((resolve) => { resolveGeneration = resolve }))
    render(<App />)
    await createStudySetInApp(user)
    await openFlashcardsInApp(user)
    expect(await screen.findByRole('status')).toHaveTextContent(/creating your flashcards/i)
    expect(screen.queryByText('What is photosynthesis?')).not.toBeInTheDocument()
    resolveGeneration(jsonResponse(flashcardResponse()))
    expect(await screen.findByText('What is photosynthesis?')).toBeInTheDocument()
  })

  it('shows an API failure and retries successfully without fake cards', async () => {
    const user = userEvent.setup()
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(jsonResponse(ingestionResponse()))
      .mockRejectedValueOnce(new TypeError('private network detail'))
      .mockResolvedValueOnce(jsonResponse(flashcardResponse()))
    global.fetch = fetchMock
    render(<App />)
    await createStudySetInApp(user)
    await openFlashcardsInApp(user)
    expect(await screen.findByRole('alert')).toHaveTextContent(/unable to connect to the study assistant server/i)
    expect(screen.queryByText('What is photosynthesis?')).not.toBeInTheDocument()
    expect(screen.queryByText(/private network detail/i)).not.toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: /retry/i }))
    expect(await screen.findByText('What is photosynthesis?')).toBeInTheDocument()
    expect(fetchMock).toHaveBeenCalledTimes(3)
  })

  it.each([
    ['malformed result', { flashcards: [{ question: 'Question only' }], count: 1 }, /missing a question or answer/i],
    ['empty result', { flashcards: [], count: 0 }, /no flashcards were created/i],
    ['missing result list', { count: 0 }, /unexpected format/i],
  ])('rejects a %s without showing a successful study card', async (_label, payload, message) => {
    const user = userEvent.setup()
    global.fetch = vi.fn()
      .mockResolvedValueOnce(jsonResponse(ingestionResponse()))
      .mockResolvedValueOnce(jsonResponse(payload))
    render(<App />)
    await createStudySetInApp(user)
    await openFlashcardsInApp(user)
    expect(await screen.findByRole('alert')).toHaveTextContent(message)
    expect(screen.queryByRole('progressbar')).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: /retry/i })).toBeInTheDocument()
  })

  it('flips between question and answer and is operable from the keyboard', async () => {
    const user = userEvent.setup()
    global.fetch = vi.fn()
      .mockResolvedValueOnce(jsonResponse(ingestionResponse()))
      .mockResolvedValueOnce(jsonResponse(flashcardResponse()))
    render(<App />)
    await createStudySetInApp(user)
    await openFlashcardsInApp(user)
    const flipButton = await screen.findByRole('button', { name: 'Show answer for card 1' })
    flipButton.focus()
    await user.keyboard('{Enter}')
    expect(await screen.findByText('The conversion of light energy into chemical energy.')).toBeInTheDocument()
    expect(screen.getByText('Answer')).toBeInTheDocument()
    await user.keyboard(' ')
    expect(screen.getByText('What is photosynthesis?')).toBeInTheDocument()
    expect(screen.getByText('Question')).toBeInTheDocument()
  })

  it('navigates cards, enforces boundaries, and reports position progress', async () => {
    const user = userEvent.setup()
    const cards = [
      { question: 'Question one', answer: 'Answer one' },
      { question: 'Question two', answer: 'Answer two' },
      { question: 'Question three', answer: 'Answer three' },
    ]
    global.fetch = vi.fn()
      .mockResolvedValueOnce(jsonResponse(ingestionResponse()))
      .mockResolvedValueOnce(jsonResponse(flashcardResponse(cards)))
    render(<App />)
    await createStudySetInApp(user)
    await openFlashcardsInApp(user)
    await screen.findByText('Question one')
    const previous = screen.getByRole('button', { name: /previous/i })
    const next = screen.getByRole('button', { name: /^next/i })
    expect(previous).toBeDisabled()
    expect(screen.getByRole('progressbar')).toHaveAttribute('aria-valuenow', '1')
    await user.click(next)
    expect(screen.getByText('Question two')).toBeInTheDocument()
    expect(previous).toBeEnabled()
    expect(screen.getByRole('progressbar')).toHaveAttribute('aria-valuenow', '2')
    await user.click(next)
    expect(screen.getByText('Question three')).toBeInTheDocument()
    expect(next).toBeDisabled()
    expect(screen.getByRole('progressbar')).toHaveAttribute('aria-valuetext', '3 of 3 cards')
  })

  it('stores Known and Need Practice feedback and supports a filtered practice review', async () => {
    const user = userEvent.setup()
    global.fetch = vi.fn()
      .mockResolvedValueOnce(jsonResponse(ingestionResponse()))
      .mockResolvedValueOnce(jsonResponse(flashcardResponse()))
    render(<App />)
    await createStudySetInApp(user)
    await openFlashcardsInApp(user)
    await screen.findByText('What is photosynthesis?')
    const knowIt = screen.getByRole('button', { name: /know it/i })
    const needPractice = screen.getByRole('button', { name: /^need practice$/i })
    expect(knowIt).toHaveAttribute('aria-pressed', 'false')
    await user.click(knowIt)
    expect(knowIt).toHaveAttribute('aria-pressed', 'true')
    expect(screen.getByText(/1 known.*0 to practice/i)).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: /^next/i }))
    await user.click(screen.getByRole('button', { name: /^need practice$/i }))
    expect(needPractice).toHaveAttribute('aria-pressed', 'true')
    expect(screen.getByText(/1 known.*1 to practice/i)).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: /review need practice \(1\)/i }))
    expect(screen.getByRole('progressbar')).toHaveAttribute('aria-valuetext', '1 of 1 cards')
    expect(screen.getByText('Where is chlorophyll found?')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /^need practice$/i })).toHaveAttribute('aria-pressed', 'true')
    expect(screen.getByRole('button', { name: /back to all cards/i })).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: /back to all cards/i }))
    expect(screen.getByRole('progressbar')).toHaveAttribute('aria-valuetext', '2 of 2 cards')
    expect(screen.getByText('Where is chlorophyll found?')).toBeInTheDocument()
  })

  it('preserves the current card and cached response when returning and reopening', async () => {
    const user = userEvent.setup()
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(jsonResponse(ingestionResponse()))
      .mockResolvedValueOnce(jsonResponse(flashcardResponse()))
    global.fetch = fetchMock
    render(<App />)
    await createStudySetInApp(user)
    await openFlashcardsInApp(user)
    await screen.findByText('What is photosynthesis?')
    await user.click(screen.getByRole('button', { name: /^know it$/i }))
    await user.click(screen.getByRole('button', { name: /^next/i }))
    await user.click(screen.getByRole('button', { name: /back to workspace/i }))
    await user.click(screen.getByRole('button', { name: /start learning/i }))
    expect(await screen.findByText('Where is chlorophyll found?')).toBeInTheDocument()
    expect(screen.getByRole('progressbar')).toHaveAttribute('aria-valuetext', '2 of 2 cards')
    await user.click(screen.getByRole('button', { name: /^previous/i }))
    expect(screen.getByRole('button', { name: /^know it$/i })).toHaveAttribute('aria-pressed', 'true')
    expect(fetchMock).toHaveBeenCalledTimes(2)
  })

  it('clears cached cards and review state when starting a new Study Set', async () => {
    const user = userEvent.setup()
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(jsonResponse(ingestionResponse({ extracted_text: 'First set text.' })))
      .mockResolvedValueOnce(jsonResponse(flashcardResponse([{ question: 'Old question', answer: 'Old answer' }])))
      .mockResolvedValueOnce(jsonResponse(ingestionResponse({ extracted_text: 'Second set text.' })))
      .mockResolvedValueOnce(jsonResponse(flashcardResponse([{ question: 'New question', answer: 'New answer' }])))
    global.fetch = fetchMock
    render(<App />)
    await createStudySetInApp(user, 'First notes')
    await openFlashcardsInApp(user)
    await screen.findByText('Old question')
    await user.click(screen.getByRole('button', { name: /back to workspace/i }))
    await user.click(screen.getByRole('button', { name: /start new study set/i }))
    await createStudySetInApp(user, 'Second notes')
    await openFlashcardsInApp(user)
    expect(await screen.findByText('New question')).toBeInTheDocument()
    expect(screen.queryByText('Old question')).not.toBeInTheDocument()
    expect(fetchMock).toHaveBeenCalledTimes(4)
  })

  it('ignores a late response from a discarded Study Set', async () => {
    const user = userEvent.setup()
    let resolveOldGeneration
    let generationCall = 0
    const fetchMock = vi.fn((url) => {
      if (url.endsWith('/api/documents/ingest')) {
        const extracted_text = fetchMock.mock.calls.filter(([calledUrl]) => calledUrl.endsWith('/api/documents/ingest')).length === 1 ? 'Old extracted text.' : 'New extracted text.'
        return Promise.resolve(jsonResponse(ingestionResponse({ extracted_text })))
      }
      generationCall += 1
      if (generationCall === 1) return new Promise((resolve) => { resolveOldGeneration = resolve })
      return Promise.resolve(jsonResponse(flashcardResponse([{ question: 'New session card', answer: 'New session answer' }])))
    })
    global.fetch = fetchMock
    render(<App />)
    await createStudySetInApp(user, 'Old notes')
    await openFlashcardsInApp(user)
    expect(screen.getByRole('status')).toHaveTextContent(/creating your flashcards/i)
    await user.click(screen.getByRole('button', { name: /back to workspace/i }))
    await user.click(screen.getByRole('button', { name: /start new study set/i }))
    await createStudySetInApp(user, 'New notes')
    await openFlashcardsInApp(user)
    expect(await screen.findByText('New session card')).toBeInTheDocument()
    resolveOldGeneration(jsonResponse(flashcardResponse([{ question: 'Stale old card', answer: 'Old session answer' }])))
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)) })
    expect(screen.getByText('New session card')).toBeInTheDocument()
    expect(screen.queryByText('Stale old card')).not.toBeInTheDocument()
  })
})

describe('AI quiz flow', () => {
  it('uses the existing quiz endpoint contract and displays the actual generated question total', async () => {
    const user = userEvent.setup()
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(jsonResponse(ingestionResponse({ extracted_text: 'Backend-normalized quiz text.' })))
      .mockResolvedValueOnce(jsonResponse(quizResponse()))
    global.fetch = fetchMock
    render(<App />)
    await createStudySetInApp(user)
    expect(fetchMock).toHaveBeenCalledOnce()
    await openQuizInApp(user)

    expect(await screen.findByRole('heading', { name: 'Quiz' })).toBeInTheDocument()
    expect(screen.getByText('Which planet is known as the Red Planet?')).toBeInTheDocument()
    expect(screen.getByRole('progressbar')).toHaveAttribute('aria-valuemax', '2')
    expect(screen.queryByText('Correct answer')).not.toBeInTheDocument()
    const [url, options] = fetchMock.mock.calls[1]
    expect(url).toBe('http://localhost:8000/api/generate/quiz')
    expect(JSON.parse(options.body)).toEqual({ text: 'Backend-normalized quiz text.', count: 5 })
  })

  it('shows quiz generation loading and prevents duplicate hook requests', async () => {
    let resolveQuiz
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(jsonResponse(ingestionResponse()))
      .mockImplementationOnce(() => new Promise((resolve) => { resolveQuiz = resolve }))
    global.fetch = fetchMock
    const { result } = renderHook(() => useStudyComposer())
    act(() => result.current.updateText('Notes for quiz testing'))
    await act(async () => { await result.current.createStudySet() })

    let firstRequest
    let duplicateRequest
    act(() => {
      firstRequest = result.current.openQuiz()
      duplicateRequest = result.current.openQuiz()
    })
    await waitFor(() => expect(result.current.isGeneratingQuiz).toBe(true))
    expect(fetchMock).toHaveBeenCalledTimes(2)
    resolveQuiz(jsonResponse(quizResponse()))
    await act(async () => { await firstRequest })
    await expect(duplicateRequest).resolves.toBe(false)
    expect(result.current.activeStudySet.quiz).toHaveLength(2)
  })

  it('renders a loading message until questions arrive', async () => {
    const user = userEvent.setup()
    let resolveQuiz
    global.fetch = vi.fn()
      .mockResolvedValueOnce(jsonResponse(ingestionResponse()))
      .mockImplementationOnce(() => new Promise((resolve) => { resolveQuiz = resolve }))
    render(<App />)
    await createStudySetInApp(user)
    await openQuizInApp(user)
    expect(await screen.findByRole('status')).toHaveTextContent(/building your quiz/i)
    expect(screen.queryByRole('radio')).not.toBeInTheDocument()
    resolveQuiz(jsonResponse(quizResponse()))
    expect(await screen.findByRole('radio', { name: /mercury/i })).toBeInTheDocument()
  })

  it('shows a safe API error and retries generation successfully', async () => {
    const user = userEvent.setup()
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(jsonResponse(ingestionResponse()))
      .mockRejectedValueOnce(new TypeError('internal socket details'))
      .mockResolvedValueOnce(jsonResponse(quizResponse()))
    global.fetch = fetchMock
    render(<App />)
    await createStudySetInApp(user)
    await openQuizInApp(user)
    expect(await screen.findByRole('alert')).toHaveTextContent(/unable to connect to the study assistant server/i)
    expect(screen.queryByRole('radio')).not.toBeInTheDocument()
    expect(screen.queryByText(/internal socket details/i)).not.toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: /retry/i }))
    expect(await screen.findByText('Which planet is known as the Red Planet?')).toBeInTheDocument()
    expect(fetchMock).toHaveBeenCalledTimes(3)
  })

  it.each([
    ['empty questions', { quiz: [], count: 0 }, /no quiz questions were created/i],
    ['missing question array', { count: 0 }, /unexpected format/i],
    ['missing question text', { quiz: [quizQuestion('', {})], count: 1 }, /invalid question/i],
    ['an invalid option count', { quiz: [quizQuestion('Question?', { options: ['A', 'B', 'C'] })], count: 1 }, /invalid answer options/i],
    ['duplicate options', { quiz: [quizQuestion('Question?', { options: ['A', 'A', 'C', 'D'], correct_answer: 'A' })], count: 1 }, /invalid answer options/i],
    ['a missing correct answer', { quiz: [quizQuestion('Question?', { correct_answer: '' })], count: 1 }, /invalid correct answer/i],
    ['a correct answer outside the options', { quiz: [quizQuestion('Question?', { correct_answer: 'Jupiter' })], count: 1 }, /invalid correct answer/i],
    ['a missing explanation', { quiz: [quizQuestion('Question?', { explanation: '' })], count: 1 }, /invalid explanation/i],
    ['a mismatched response count', { quiz: [quizQuestion()], count: 2 }, /incomplete quiz/i],
  ])('rejects %s and does not show a question', async (_label, payload, message) => {
    const user = userEvent.setup()
    global.fetch = vi.fn()
      .mockResolvedValueOnce(jsonResponse(ingestionResponse()))
      .mockResolvedValueOnce(jsonResponse(payload))
    render(<App />)
    await createStudySetInApp(user)
    await openQuizInApp(user)
    expect(await screen.findByRole('alert')).toHaveTextContent(message)
    expect(screen.queryByRole('radio')).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: /retry/i })).toBeInTheDocument()
  })

  it('allows selecting and submitting an answer, then locks the submitted choice', async () => {
    const user = userEvent.setup()
    global.fetch = vi.fn()
      .mockResolvedValueOnce(jsonResponse(ingestionResponse()))
      .mockResolvedValueOnce(jsonResponse(quizResponse([quizQuestion()])))
    render(<App />)
    await createStudySetInApp(user)
    await openQuizInApp(user)
    const submit = await screen.findByRole('button', { name: /submit answer/i })
    expect(submit).toBeDisabled()
    const answer = screen.getByRole('radio', { name: /mars/i })
    await user.click(answer)
    expect(answer).toBeChecked()
    expect(submit).toBeEnabled()
    await user.click(submit)
    expect(await screen.findByRole('status')).toHaveTextContent(/that's correct/i)
    expect(screen.getByText(/mars appears red because iron minerals/i)).toBeInTheDocument()
    expect(answer).toBeDisabled()
    expect(screen.getByText('Correct answer')).toBeInTheDocument()
  })

  it('shows exact incorrect-answer feedback, correct option, and backend explanation', async () => {
    const user = userEvent.setup()
    global.fetch = vi.fn()
      .mockResolvedValueOnce(jsonResponse(ingestionResponse()))
      .mockResolvedValueOnce(jsonResponse(quizResponse([quizQuestion()])))
    render(<App />)
    await createStudySetInApp(user)
    await openQuizInApp(user)
    const selected = await screen.findByRole('radio', { name: /venus/i })
    await user.click(selected)
    await user.click(screen.getByRole('button', { name: /submit answer/i }))
    expect(await screen.findByRole('status')).toHaveTextContent(/not quite/i)
    expect(screen.getByRole('status')).toHaveTextContent(/correct answer: mars/i)
    expect(screen.getByText(/mars appears red because iron minerals/i)).toBeInTheDocument()
    expect(selected).toBeDisabled()
  })

  it('preserves question selection, submitted answers, and progress when navigating questions', async () => {
    const user = userEvent.setup()
    global.fetch = vi.fn()
      .mockResolvedValueOnce(jsonResponse(ingestionResponse()))
      .mockResolvedValueOnce(jsonResponse(quizResponse()))
    render(<App />)
    await createStudySetInApp(user)
    await openQuizInApp(user)
    const mars = await screen.findByRole('radio', { name: /mars/i })
    await user.click(mars)
    await user.click(screen.getByRole('button', { name: /submit answer/i }))
    await user.click(screen.getByRole('button', { name: /^continue/i }))
    expect(screen.getByText('Which planet is closest to the Sun?')).toBeInTheDocument()
    expect(screen.getByRole('progressbar')).toHaveAttribute('aria-valuetext', 'Question 2 of 2')
    await user.click(screen.getByRole('radio', { name: /venus/i }))
    await user.click(screen.getByRole('button', { name: /previous/i }))
    expect(screen.getByRole('radio', { name: /mars/i })).toBeDisabled()
    expect(screen.getByRole('radio', { name: /mars/i })).toBeChecked()
    await user.click(screen.getByRole('button', { name: /^continue/i }))
    expect(screen.getByRole('radio', { name: /venus/i })).toBeChecked()
    expect(screen.getByRole('button', { name: /submit answer/i })).toBeEnabled()
  })

  it('calculates the final attempt score and reviews incorrect answers', async () => {
    const user = userEvent.setup()
    global.fetch = vi.fn()
      .mockResolvedValueOnce(jsonResponse(ingestionResponse()))
      .mockResolvedValueOnce(jsonResponse(quizResponse()))
    render(<App />)
    await createStudySetInApp(user)
    await openQuizInApp(user)
    await user.click(await screen.findByRole('radio', { name: /venus/i }))
    await user.click(screen.getByRole('button', { name: /submit answer/i }))
    await user.click(screen.getByRole('button', { name: /^continue/i }))
    await user.click(screen.getByRole('radio', { name: /mercury/i }))
    await user.click(screen.getByRole('button', { name: /submit answer/i }))
    await user.click(screen.getByRole('button', { name: /view results/i }))

    expect(await screen.findByRole('heading', { name: /quiz complete/i })).toBeInTheDocument()
    expect(screen.getByText(/1 of 2 correct/)).toBeInTheDocument()
    expect(screen.getByText('50%')).toBeInTheDocument()
    expect(screen.getByText(/1 incorrect answer/)).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: /review incorrect answers/i }))
    expect(await screen.findByRole('heading', { name: /review incorrect answers/i })).toBeInTheDocument()
    expect(screen.getByText('Which planet is known as the Red Planet?')).toBeInTheDocument()
    expect(screen.getByText('Correct answer')).toBeInTheDocument()
    expect(screen.getByText('Your answer:').parentElement).toHaveTextContent('Your answer: Venus')
    expect(screen.getByText(/mars appears red because iron minerals/i)).toBeInTheDocument()
  })

  it('retries only incorrect questions and preserves the original result', async () => {
    const user = userEvent.setup()
    global.fetch = vi.fn()
      .mockResolvedValueOnce(jsonResponse(ingestionResponse()))
      .mockResolvedValueOnce(jsonResponse(quizResponse()))
    render(<App />)
    await createStudySetInApp(user)
    await openQuizInApp(user)
    await user.click(await screen.findByRole('radio', { name: /venus/i }))
    await user.click(screen.getByRole('button', { name: /submit answer/i }))
    await user.click(screen.getByRole('button', { name: /^continue/i }))
    await user.click(screen.getByRole('radio', { name: /mercury/i }))
    await user.click(screen.getByRole('button', { name: /submit answer/i }))
    await user.click(screen.getByRole('button', { name: /view results/i }))
    await user.click(screen.getByRole('button', { name: /retry incorrect questions/i }))

    expect(await screen.findByText('Question 1 of 1')).toBeInTheDocument()
    expect(screen.getByText('Which planet is known as the Red Planet?')).toBeInTheDocument()
    await user.click(screen.getByRole('radio', { name: /mars/i }))
    await user.click(screen.getByRole('button', { name: /submit answer/i }))
    await user.click(screen.getByRole('button', { name: /view results/i }))
    expect(await screen.findByText(/1 of 1 correct/)).toBeInTheDocument()
    expect(screen.getByText('100%')).toBeInTheDocument()
    expect(screen.getByText('Previous attempt')).toBeInTheDocument()
    expect(screen.getByText('1 of 2 correct · 50%')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: /review previous mistakes/i }))
    expect(screen.getByText('Your answer:').parentElement).toHaveTextContent('Your answer: Venus')
  })

  it('preserves an in-progress attempt and cached questions when reopening Quiz', async () => {
    const user = userEvent.setup()
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(jsonResponse(ingestionResponse()))
      .mockResolvedValueOnce(jsonResponse(quizResponse()))
    global.fetch = fetchMock
    render(<App />)
    await createStudySetInApp(user)
    await openQuizInApp(user)
    await user.click(await screen.findByRole('radio', { name: /mars/i }))
    await user.click(screen.getByRole('button', { name: /submit answer/i }))
    await user.click(screen.getByRole('button', { name: /^continue/i }))
    await user.click(screen.getByRole('radio', { name: /venus/i }))
    await user.click(screen.getByRole('button', { name: /back to workspace/i }))
    await openQuizInApp(user)
    expect(await screen.findByText('Which planet is closest to the Sun?')).toBeInTheDocument()
    expect(screen.getByRole('radio', { name: /venus/i })).toBeChecked()
    expect(fetchMock).toHaveBeenCalledTimes(2)
  })

  it('clears the previous quiz and attempt state when starting a new Study Set', async () => {
    const user = userEvent.setup()
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(jsonResponse(ingestionResponse({ extracted_text: 'First text' })))
      .mockResolvedValueOnce(jsonResponse(quizResponse([quizQuestion('Old quiz question?')])))
      .mockResolvedValueOnce(jsonResponse(ingestionResponse({ extracted_text: 'Second text' })))
      .mockResolvedValueOnce(jsonResponse(quizResponse([quizQuestion('New quiz question?')])))
    global.fetch = fetchMock
    render(<App />)
    await createStudySetInApp(user, 'First notes')
    await openQuizInApp(user)
    await screen.findByText('Old quiz question?')
    await user.click(screen.getByRole('button', { name: /back to workspace/i }))
    await user.click(screen.getByRole('button', { name: /start new study set/i }))
    await createStudySetInApp(user, 'Second notes')
    await openQuizInApp(user)
    expect(await screen.findByText('New quiz question?')).toBeInTheDocument()
    expect(screen.queryByText('Old quiz question?')).not.toBeInTheDocument()
    expect(fetchMock).toHaveBeenCalledTimes(4)
  })

  it('ignores a late response from a discarded Study Set', async () => {
    const user = userEvent.setup()
    let resolveOldQuiz
    let generationCall = 0
    const fetchMock = vi.fn((url) => {
      if (url.endsWith('/api/documents/ingest')) {
        const count = fetchMock.mock.calls.filter(([calledUrl]) => calledUrl.endsWith('/api/documents/ingest')).length
        return Promise.resolve(jsonResponse(ingestionResponse({ extracted_text: count === 1 ? 'Old set text' : 'New set text' })))
      }
      generationCall += 1
      if (generationCall === 1) return new Promise((resolve) => { resolveOldQuiz = resolve })
      return Promise.resolve(jsonResponse(quizResponse([quizQuestion('New session question?')])))
    })
    global.fetch = fetchMock
    render(<App />)
    await createStudySetInApp(user, 'Old notes')
    await openQuizInApp(user)
    expect(screen.getByRole('status')).toHaveTextContent(/building your quiz/i)
    await user.click(screen.getByRole('button', { name: /back to workspace/i }))
    await user.click(screen.getByRole('button', { name: /start new study set/i }))
    await createStudySetInApp(user, 'New notes')
    await openQuizInApp(user)
    expect(await screen.findByText('New session question?')).toBeInTheDocument()
    resolveOldQuiz(jsonResponse(quizResponse([quizQuestion('Stale old question?')])))
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)) })
    expect(screen.getByText('New session question?')).toBeInTheDocument()
    expect(screen.queryByText('Stale old question?')).not.toBeInTheDocument()
  })
})

describe('file type compatibility', () => {
  it('accepts TXT with text/plain', () => {
    expect(isSupportedFile(new File(['notes'], 'notes.txt', { type: 'text/plain' }))).toBe(true)
  })

  it('accepts PDF with application/pdf', () => {
    expect(isSupportedFile(new File(['pdf'], 'notes.pdf', { type: 'application/pdf' }))).toBe(true)
  })

  it('accepts TXT with application/octet-stream', () => {
    expect(isSupportedFile(new File(['notes'], 'notes.txt', { type: 'application/octet-stream' }))).toBe(true)
  })

  it('accepts PDF with an empty MIME type', () => {
    expect(isSupportedFile(new File(['pdf'], 'notes.pdf', { type: '' }))).toBe(true)
  })

  it('rejects unsupported extensions', () => {
    expect(isSupportedFile(new File(['doc'], 'notes.docx', { type: 'application/octet-stream' }))).toBe(false)
  })
})

describe('API client errors and endpoint contracts', () => {
  it('returns JSON for a successful response', async () => {
    global.fetch = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ ok: true }) })
    await expect(request('/api/test')).resolves.toEqual({ ok: true })
  })

  it('preserves HTTP status and response payload on failure', async () => {
    const payload = { detail: 'Invalid input', code: 'bad_request' }
    global.fetch = vi.fn().mockResolvedValue({ ok: false, status: 422, json: async () => payload })
    await expect(request('/api/test')).rejects.toMatchObject({ name: 'ApiError', status: 422, message: 'Invalid input', payload })
  })

  it('normalizes network failures and preserves the cause', async () => {
    const networkError = new TypeError('socket details')
    global.fetch = vi.fn().mockRejectedValue(networkError)
    await expect(request('/api/test')).rejects.toMatchObject({
      name: 'ApiError',
      status: null,
      message: 'Unable to connect to the Study Assistant server. Check your connection and try again.',
      cause: networkError,
    })
  })
  it('uses a safe message for server errors while retaining status and payload', async () => {
    const payload = { detail: 'Internal traceback and filesystem details' }
    global.fetch = vi.fn().mockResolvedValue({ ok: false, status: 500, json: async () => payload })
    await expect(request('/api/test')).rejects.toMatchObject({
      name: 'ApiError',
      status: 500,
      message: 'The Study Assistant server could not complete the request. Please try again.',
      payload,
    })
  })

  it('does not swallow cancellation while reading a response body', async () => {
    const controller = new AbortController()
    const abortError = new DOMException('The body read was aborted.', 'AbortError')
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: () => controller.signal.aborted
        ? Promise.reject(abortError)
        : new Promise((_resolve, reject) => {
          controller.signal.addEventListener('abort', () => reject(abortError), { once: true })
        }),
    })
    const pendingRequest = request('/api/test', { signal: controller.signal })
    controller.abort()
    await expect(pendingRequest).rejects.toBe(abortError)
  })

  it('propagates AbortController cancellation without converting it to ApiError', async () => {
    const controller = new AbortController()
    const abortError = new DOMException('The operation was aborted.', 'AbortError')
    global.fetch = vi.fn((_url, { signal }) => new Promise((_resolve, reject) => {
      signal.addEventListener('abort', () => reject(abortError), { once: true })
    }))
    const pendingRequest = request('/api/test', { signal: controller.signal })
    controller.abort()
    await expect(pendingRequest).rejects.toBe(abortError)
    await expect(pendingRequest).rejects.not.toBeInstanceOf(ApiError)
  })

  it('maps methods to expected paths and request formats', async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ({}) })
    global.fetch = fetchMock
    await ingestText('notes')
    await generateFlashcards('notes', 3)
    await generateQuiz('notes', 2)
    const file = new File(['x'], 'notes.txt', { type: 'text/plain' })
    await ingestFile(file)
    const calls = fetchMock.mock.calls
    expect(calls.map(([url]) => url)).toEqual([
      'http://localhost:8000/api/documents/ingest',
      'http://localhost:8000/api/generate/flashcards',
      'http://localhost:8000/api/generate/quiz',
      'http://localhost:8000/api/documents/ingest',
    ])
    expect(JSON.parse(calls[0][1].body)).toEqual({ text: 'notes' })
    expect(calls[0][1].headers['Content-Type']).toBe('application/json')
    expect(JSON.parse(calls[1][1].body)).toEqual({ text: 'notes', count: 3 })
    expect(JSON.parse(calls[2][1].body)).toEqual({ text: 'notes', count: 2 })
    expect(calls[3][1].body).toBeInstanceOf(FormData)
    expect(calls[3][1].body.get('file')).toBe(file)
    expect(calls[3][1].headers).toEqual({})
  })
})
