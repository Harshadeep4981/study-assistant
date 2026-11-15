import { useRef, useState } from 'react'
import { generateFlashcards, generateQuiz, ingestFile, ingestText } from '../api/studyApi.js'

export function isSupportedFile(file) {
  return Boolean(file && /\.(txt|pdf)$/i.test(file.name))
}

export function validateFlashcardResponse(response, studySetId) {
  if (!Array.isArray(response?.flashcards)) {
    throw new Error('The server returned flashcards in an unexpected format. Please retry.')
  }
  if (response.flashcards.length === 0) {
    throw new Error('No flashcards were created from this material. Try again or use different notes.')
  }
  if (!Number.isInteger(response.count) || response.count !== response.flashcards.length) {
    throw new Error('The server returned an incomplete flashcard set. Please retry.')
  }

  return response.flashcards.map((card, index) => {
    if (typeof card?.question !== 'string' || !card.question.trim()
      || typeof card?.answer !== 'string' || !card.answer.trim()) {
      throw new Error(`Flashcard ${index + 1} is missing a question or answer. Please retry.`)
    }
    return {
      id: `${studySetId}-card-${index + 1}`,
      question: card.question.trim(),
      answer: card.answer.trim(),
    }
  })
}

export function validateQuizResponse(response, studySetId) {
  if (!Array.isArray(response?.quiz)) {
    throw new Error('The server returned quiz questions in an unexpected format. Please retry.')
  }
  if (response.quiz.length === 0) {
    throw new Error('No quiz questions were created from this material. Try again or use different notes.')
  }
  if (!Number.isInteger(response.count) || response.count !== response.quiz.length) {
    throw new Error('The server returned an incomplete quiz. Please retry.')
  }

  return response.quiz.map((item, index) => {
    const invalid = (field) => new Error(`Quiz question ${index + 1} has an invalid ${field}. Please retry.`)
    if (typeof item?.question !== 'string' || !item.question.trim()) throw invalid('question')
    if (!Array.isArray(item.options) || item.options.length !== 4) throw invalid('answer options')
    const options = item.options.map((option) => typeof option === 'string' ? option.trim() : '')
    if (options.some((option) => !option) || new Set(options).size !== 4) throw invalid('answer options')
    if (typeof item.correct_answer !== 'string' || !item.correct_answer.trim()) throw invalid('correct answer')
    const correctAnswer = item.correct_answer.trim()
    if (!options.includes(correctAnswer)) throw invalid('correct answer')
    if (typeof item.explanation !== 'string' || !item.explanation.trim()) throw invalid('explanation')

    return {
      id: `${studySetId}-question-${index + 1}`,
      question: item.question.trim(),
      options,
      correctAnswer,
      explanation: item.explanation.trim(),
    }
  })
}

export function useStudyComposer() {
  const [inputMode, setInputMode] = useState('paste')
  const [text, setText] = useState('')
  const [file, setFile] = useState(null)
  const [message, setMessage] = useState('')
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [activeStudySet, setActiveStudySet] = useState(null)
  const [screen, setScreen] = useState('creation')
  const [isGeneratingFlashcards, setIsGeneratingFlashcards] = useState(false)
  const [flashcardError, setFlashcardError] = useState('')
  const [isGeneratingQuiz, setIsGeneratingQuiz] = useState(false)
  const [quizError, setQuizError] = useState('')
  const submissionLock = useRef(false)
  const activeStudySetRef = useRef(null)
  const generationRef = useRef({ token: 0, controller: null, locked: false })
  const quizGenerationRef = useRef({ token: 0, controller: null, locked: false })

  function commitStudySet(nextStudySet) {
    activeStudySetRef.current = nextStudySet
    setActiveStudySet(nextStudySet)
  }

  function changeInputMode(nextMode) {
    setInputMode(nextMode)
    setMessage('')
  }

  function updateText(value) {
    setText(value)
    setMessage('')
  }

  function chooseFile(nextFile) {
    if (!nextFile) return false
    if (!isSupportedFile(nextFile)) {
      setMessage('That file type is not supported. Choose a TXT or PDF document.')
      return false
    }
    setFile(nextFile)
    setMessage('')
    return true
  }

  function clearFile() {
    setFile(null)
    setMessage('')
  }

  const activeSource = inputMode === 'paste'
    ? (text.trim() ? { type: 'text', text: text.trim() } : null)
    : (file ? { type: 'file', file } : null)

  function validate() {
    if (!activeSource) {
      setMessage(inputMode === 'paste' ? 'Add notes or a topic to continue.' : 'Choose a TXT or PDF document to continue.')
      return false
    }
    return true
  }

  async function createStudySet() {
    if (submissionLock.current || !validate()) return false
    submissionLock.current = true
    setIsSubmitting(true)
    setMessage('')

    try {
      const response = activeSource.type === 'text'
        ? await ingestText(activeSource.text)
        : await ingestFile(activeSource.file)

      if (typeof response?.extracted_text !== 'string' || !response.extracted_text.trim()) {
        throw new Error('The server did not return any extracted study material. Please try again.')
      }

      const sourceName = activeSource.type === 'file'
        ? (response.metadata?.filename || activeSource.file.name)
        : null
      commitStudySet({
        id: globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(36).slice(2)}`,
        extractedText: response.extracted_text,
        sourceType: response.source_type,
        sourceName,
        createdAt: new Date().toISOString(),
        flashcards: [],
        flashcardStatuses: {},
        flashcardCurrentId: null,
        flashcardReviewMode: 'all',
        quiz: [],
        quizAttempt: null,
        quizAttempts: [],
        quizReview: { resultId: null, questionIds: [], index: 0 },
        warnings: Array.isArray(response.warnings) ? response.warnings : [],
        metadata: response.metadata || {},
      })
      setFlashcardError('')
      setQuizError('')
      setScreen('workspace')
      return true
    } catch (error) {
      if (error?.name === 'AbortError') throw error
      setMessage(error?.message || 'We could not process your study material. Please try again.')
      return false
    } finally {
      submissionLock.current = false
      setIsSubmitting(false)
    }
  }

  async function openFlashcards() {
    const studySet = activeStudySetRef.current
    if (!studySet) return false
    setScreen('flashcards')
    setFlashcardError('')
    if (studySet.flashcards.length > 0) return true

    const generation = generationRef.current
    if (generation.locked) return false
    const token = generation.token + 1
    generationRef.current = { token, controller: new AbortController(), locked: true }
    const controller = generationRef.current.controller
    setIsGeneratingFlashcards(true)

    try {
      const response = await generateFlashcards(studySet.extractedText, 5, { signal: controller.signal })
      const cards = validateFlashcardResponse(response, studySet.id)
      if (generationRef.current.token !== token || activeStudySetRef.current?.id !== studySet.id) return false

      commitStudySet({
        ...activeStudySetRef.current,
        flashcards: cards,
        flashcardStatuses: {},
        flashcardCurrentId: cards[0].id,
        flashcardReviewMode: 'all',
      })
      return true
    } catch (error) {
      if (error?.name === 'AbortError') return false
      if (generationRef.current.token === token && activeStudySetRef.current?.id === studySet.id) {
        setFlashcardError(error?.message || 'We could not create flashcards. Please try again.')
      }
      return false
    } finally {
      if (generationRef.current.token === token) {
        generationRef.current = { ...generationRef.current, controller: null, locked: false }
        setIsGeneratingFlashcards(false)
      }
    }
  }

  function returnToWorkspace() {
    setScreen('workspace')
  }

  function setFlashcardCurrentId(cardId) {
    const current = activeStudySetRef.current
    if (!current || !current.flashcards.some((card) => card.id === cardId)) return
    commitStudySet({ ...current, flashcardCurrentId: cardId })
  }

  function setFlashcardReviewMode(mode) {
    const current = activeStudySetRef.current
    if (!current || !['all', 'practice'].includes(mode)) return
    const visibleCards = mode === 'practice'
      ? current.flashcards.filter((card) => current.flashcardStatuses[card.id] === 'practice')
      : current.flashcards
    const currentStillVisible = visibleCards.some((card) => card.id === current.flashcardCurrentId)
    commitStudySet({
      ...current,
      flashcardReviewMode: mode,
      flashcardCurrentId: currentStillVisible ? current.flashcardCurrentId : (visibleCards[0]?.id ?? null),
    })
  }

  function markFlashcard(cardId, status) {
    const current = activeStudySetRef.current
    if (!current || !['known', 'practice'].includes(status)) return
    const flashcardStatuses = { ...current.flashcardStatuses, [cardId]: status }
    let flashcardCurrentId = current.flashcardCurrentId

    if (current.flashcardReviewMode === 'practice' && cardId === current.flashcardCurrentId && status !== 'practice') {
      const remainingPractice = current.flashcards.filter((card) => flashcardStatuses[card.id] === 'practice')
      const oldPosition = current.flashcards
        .filter((card) => current.flashcardStatuses[card.id] === 'practice')
        .findIndex((card) => card.id === cardId)
      flashcardCurrentId = remainingPractice[Math.min(Math.max(oldPosition, 0), remainingPractice.length - 1)]?.id ?? null
    }

    commitStudySet({ ...current, flashcardStatuses, flashcardCurrentId })
  }

  function makeQuizAttempt(studySetId, questionIds, mode = 'all') {
    return {
      id: `${studySetId}-quiz-attempt-${Date.now()}-${Math.random().toString(36).slice(2)}`,
      mode,
      questionIds,
      currentIndex: 0,
      selectedAnswers: {},
      submittedAnswers: {},
      correctCount: 0,
      isComplete: false,
    }
  }

  async function openQuiz() {
    const studySet = activeStudySetRef.current
    if (!studySet) return false
    setQuizError('')
    if (studySet.quiz.length > 0) {
      setScreen(studySet.quizAttempt?.isComplete ? 'quiz-results' : 'quiz')
      return true
    }

    setScreen('quiz')
    const generation = quizGenerationRef.current
    if (generation.locked) return false
    const token = generation.token + 1
    quizGenerationRef.current = { token, controller: new AbortController(), locked: true }
    const controller = quizGenerationRef.current.controller
    setIsGeneratingQuiz(true)

    try {
      const response = await generateQuiz(studySet.extractedText, 5, { signal: controller.signal })
      const questions = validateQuizResponse(response, studySet.id)
      if (quizGenerationRef.current.token !== token || activeStudySetRef.current?.id !== studySet.id) return false
      const quizAttempt = makeQuizAttempt(studySet.id, questions.map((question) => question.id))
      commitStudySet({ ...activeStudySetRef.current, quiz: questions, quizAttempt, quizAttempts: [], quizReview: { resultId: null, questionIds: [], index: 0 } })
      return true
    } catch (error) {
      if (error?.name === 'AbortError') return false
      if (quizGenerationRef.current.token === token && activeStudySetRef.current?.id === studySet.id) {
        setQuizError(error?.message || 'We could not create your quiz. Please try again.')
      }
      return false
    } finally {
      if (quizGenerationRef.current.token === token) {
        quizGenerationRef.current = { ...quizGenerationRef.current, controller: null, locked: false }
        setIsGeneratingQuiz(false)
      }
    }
  }

  function chooseQuizAnswer(questionId, answer) {
    const current = activeStudySetRef.current
    const attempt = current?.quizAttempt
    const question = current?.quiz.find((item) => item.id === questionId)
    if (!current || !attempt || attempt.isComplete || !question?.options.includes(answer) || attempt.submittedAnswers[questionId]) return
    commitStudySet({
      ...current,
      quizAttempt: { ...attempt, selectedAnswers: { ...attempt.selectedAnswers, [questionId]: answer } },
    })
  }

  function submitQuizAnswer() {
    const current = activeStudySetRef.current
    const attempt = current?.quizAttempt
    const questionId = attempt?.questionIds[attempt.currentIndex]
    const question = current?.quiz.find((item) => item.id === questionId)
    const selectedAnswer = questionId ? attempt?.selectedAnswers[questionId] : null
    if (!current || !attempt || !question || !selectedAnswer || attempt.submittedAnswers[questionId]) return false
    const isCorrect = selectedAnswer === question.correctAnswer
    commitStudySet({
      ...current,
      quizAttempt: {
        ...attempt,
        submittedAnswers: { ...attempt.submittedAnswers, [questionId]: { answer: selectedAnswer, isCorrect } },
        correctCount: attempt.correctCount + Number(isCorrect),
      },
    })
    return true
  }

  function moveQuizQuestion(index) {
    const current = activeStudySetRef.current
    const attempt = current?.quizAttempt
    if (!current || !attempt || attempt.isComplete || index < 0 || index >= attempt.questionIds.length) return
    const currentQuestionId = attempt.questionIds[attempt.currentIndex]
    if (!attempt.submittedAnswers[currentQuestionId] && index > attempt.currentIndex) return
    commitStudySet({ ...current, quizAttempt: { ...attempt, currentIndex: index } })
  }

  function continueQuiz() {
    const current = activeStudySetRef.current
    const attempt = current?.quizAttempt
    if (!current || !attempt) return false
    const currentQuestionId = attempt.questionIds[attempt.currentIndex]
    if (!attempt.submittedAnswers[currentQuestionId]) return false
    if (attempt.currentIndex < attempt.questionIds.length - 1) {
      moveQuizQuestion(attempt.currentIndex + 1)
      return true
    }

    const incorrectQuestionIds = attempt.questionIds.filter((questionId) => !attempt.submittedAnswers[questionId]?.isCorrect)
    const result = {
      id: attempt.id,
      mode: attempt.mode,
      total: attempt.questionIds.length,
      correctCount: attempt.correctCount,
      incorrectQuestionIds,
      answers: attempt.submittedAnswers,
      percentage: Math.round((attempt.correctCount / attempt.questionIds.length) * 100),
      completedAt: new Date().toISOString(),
    }
    const completedAttempt = { ...attempt, isComplete: true }
    commitStudySet({ ...current, quizAttempt: completedAttempt, quizAttempts: [...current.quizAttempts, result] })
    setScreen('quiz-results')
    return true
  }

  function retryIncorrectQuizQuestions() {
    const current = activeStudySetRef.current
    const latestResult = current?.quizAttempts.at(-1)
    if (!current || !latestResult?.incorrectQuestionIds.length) return false
    const quizAttempt = makeQuizAttempt(current.id, latestResult.incorrectQuestionIds, 'retry')
    commitStudySet({ ...current, quizAttempt })
    setScreen('quiz')
    return true
  }

  function reviewIncorrectQuizQuestions(resultId) {
    const current = activeStudySetRef.current
    const result = current?.quizAttempts.find((attempt) => attempt.id === (resultId || current.quizAttempts.at(-1)?.id))
    if (!current || !result?.incorrectQuestionIds.length) return false
    commitStudySet({
      ...current,
      quizReview: { resultId: result.id, questionIds: result.incorrectQuestionIds, index: 0 },
    })
    setScreen('quiz-review')
    return true
  }

  function moveQuizReview(index) {
    const current = activeStudySetRef.current
    if (!current || index < 0 || index >= current.quizReview.questionIds.length) return
    commitStudySet({ ...current, quizReview: { ...current.quizReview, index } })
  }

  function returnToQuizResults() {
    setScreen('quiz-results')
  }

  function startNewStudySet() {
    const generation = generationRef.current
    generation.controller?.abort()
    generationRef.current = { token: generation.token + 1, controller: null, locked: false }
    const quizGeneration = quizGenerationRef.current
    quizGeneration.controller?.abort()
    quizGenerationRef.current = { token: quizGeneration.token + 1, controller: null, locked: false }
    activeStudySetRef.current = null
    setActiveStudySet(null)
    setInputMode('paste')
    setText('')
    setFile(null)
    setMessage('')
    setFlashcardError('')
    setIsGeneratingFlashcards(false)
    setQuizError('')
    setIsGeneratingQuiz(false)
    setScreen('creation')
  }

  return {
    inputMode, changeInputMode, text, updateText, file,
    chooseFile, clearFile, activeSource, message, validate, createStudySet,
    isSubmitting, activeStudySet, screen, openFlashcards, returnToWorkspace,
    isGeneratingFlashcards, flashcardError, setFlashcardCurrentId,
    setFlashcardReviewMode, markFlashcard, isGeneratingQuiz, quizError,
    openQuiz, chooseQuizAnswer, submitQuizAnswer, moveQuizQuestion, continueQuiz,
    retryIncorrectQuizQuestions, reviewIncorrectQuizQuestions, moveQuizReview,
    returnToQuizResults, startNewStudySet,
  }
}
