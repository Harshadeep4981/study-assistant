import { useState } from 'react'
import { ArrowRight, LoaderCircle } from 'lucide-react'
import AppHeader from './components/layout/AppHeader.jsx'
import StudyStage from './components/layout/StudyStage.jsx'
import LearningIllustration from './components/brand/LearningIllustration.jsx'
import MaterialInput from './features/document-input/components/MaterialInput.jsx'
import FlashcardPage from './features/flashcards/components/FlashcardPage.jsx'
import QuizPage from './features/quiz/components/QuizPage.jsx'
import QuizResults from './features/quiz/components/QuizResults.jsx'
import QuizReviewPage from './features/quiz/components/QuizReviewPage.jsx'
import StudyWorkspace from './features/study-set/components/StudyWorkspace.jsx'
import { useStudyComposer } from './state/useStudyComposer.js'

const MOCK_PREVIEW_SET = {
  id: 'preview-set-1',
  extractedText: 'Mitochondria generate ATP through cellular respiration...',
  sourceType: 'pasted_text',
  sourceName: 'Cell Biology Notes',
  createdAt: '2026-11-15T10:30:00.000Z',
  warnings: [],
  metadata: { character_count: 2450 },
  flashcards: [
    {
      id: 'preview-set-1-card-1',
      question: 'What is the primary function of mitochondria in eukaryotic cells?',
      answer: 'To generate the majority of ATP through aerobic cellular respiration and oxidative phosphorylation.',
    },
    {
      id: 'preview-set-1-card-2',
      question: 'What molecule acts as the universal chemical energy currency of the cell?',
      answer: 'Adenosine triphosphate (ATP).',
    },
  ],
  flashcardStatuses: { 'preview-set-1-card-1': 'known' },
  flashcardCurrentId: 'preview-set-1-card-1',
  flashcardReviewMode: 'all',
  quiz: [
    {
      id: 'preview-set-1-question-1',
      question: 'Which metabolic stage yields the greatest net quantity of ATP during aerobic respiration?',
      options: ['Glycolysis', 'Pyruvate oxidation', 'Oxidative phosphorylation', 'Citric acid cycle'],
      correctAnswer: 'Oxidative phosphorylation',
      explanation: 'Oxidative phosphorylation via the electron transport chain and ATP synthase generates approximately 28 to 34 ATP molecules per glucose.',
    },
    {
      id: 'preview-set-1-question-2',
      question: 'In eukaryotic cells, where does the citric acid cycle (Krebs cycle) take place?',
      options: ['Cytosol', 'Mitochondrial matrix', 'Inner mitochondrial membrane', 'Nucleolus'],
      correctAnswer: 'Mitochondrial matrix',
      explanation: 'The enzymes catalyzing the citric acid cycle reactions reside in the aqueous mitochondrial matrix.',
    },
  ],
  quizAttempt: {
    id: 'preview-set-1-attempt-1',
    mode: 'all',
    questionIds: ['preview-set-1-question-1', 'preview-set-1-question-2'],
    currentIndex: 0,
    selectedAnswers: { 'preview-set-1-question-1': 'Oxidative phosphorylation' },
    submittedAnswers: {},
    correctCount: 0,
    isComplete: false,
  },
  quizAttempts: [
    {
      id: 'preview-set-1-attempt-1',
      mode: 'all',
      total: 2,
      correctCount: 1,
      incorrectQuestionIds: ['preview-set-1-question-2'],
      answers: {
        'preview-set-1-question-1': { answer: 'Oxidative phosphorylation', isCorrect: true },
        'preview-set-1-question-2': { answer: 'Cytosol', isCorrect: false },
      },
      percentage: 50,
      completedAt: '2026-11-15T11:00:00.000Z',
    },
  ],
  quizReview: {
    resultId: 'preview-set-1-attempt-1',
    questionIds: ['preview-set-1-question-2'],
    index: 0,
  },
}

export default function App() {
  const composer = useStudyComposer()
  const previewParam = typeof window !== 'undefined'
    ? new URLSearchParams(window.location.search).get('preview')
    : null

  const [previewState, setPreviewState] = useState(() => {
    if (!previewParam) return null
    return {
      screen: previewParam,
      studySet: { ...MOCK_PREVIEW_SET },
    }
  })

  // Determine active screen and study set
  const isPreview = Boolean(previewState)
  const activeScreen = isPreview ? previewState.screen : (composer.activeStudySet ? composer.screen : 'creation')
  const activeStudySet = isPreview ? previewState.studySet : composer.activeStudySet

  return (
    <div id="home" className={`min-h-screen ${activeStudySet && activeScreen !== 'creation' ? 'study-app-shell' : 'bg-ivory text-ink'}`}>
      <AppHeader
        onWorkspaceClick={
          isPreview
            ? () => setPreviewState((prev) => ({ ...prev, screen: 'workspace' }))
            : (composer.screen === 'flashcards' || composer.screen.startsWith('quiz') ? composer.returnToWorkspace : undefined)
        }
      />
      {activeStudySet && activeScreen !== 'creation' ? <StudyStage key={activeScreen}>
        {activeScreen === 'flashcards' ? (
          <FlashcardPage
            studySet={activeStudySet}
            isLoading={composer.isGeneratingFlashcards}
            error={composer.flashcardError}
            onRetry={composer.openFlashcards}
            onBack={isPreview ? () => setPreviewState((prev) => ({ ...prev, screen: 'workspace' })) : composer.returnToWorkspace}
            onSetCurrentCard={composer.setFlashcardCurrentId}
            onSetReviewMode={composer.setFlashcardReviewMode}
            onMarkCard={composer.markFlashcard}
          />
        ) : activeScreen === 'quiz' ? (
          <QuizPage
            studySet={activeStudySet}
            isLoading={composer.isGeneratingQuiz}
            error={composer.quizError}
            onRetry={composer.openQuiz}
            onBack={isPreview ? () => setPreviewState((prev) => ({ ...prev, screen: 'workspace' })) : composer.returnToWorkspace}
            onChooseAnswer={composer.chooseQuizAnswer}
            onSubmitAnswer={composer.submitQuizAnswer}
            onMoveQuestion={composer.moveQuizQuestion}
            onContinue={composer.continueQuiz}
          />
        ) : activeScreen === 'quiz-results' ? (
          <QuizResults
            studySet={activeStudySet}
            onBack={isPreview ? () => setPreviewState((prev) => ({ ...prev, screen: 'workspace' })) : composer.returnToWorkspace}
            onReviewIncorrect={isPreview ? () => setPreviewState((prev) => ({ ...prev, screen: 'quiz-review' })) : composer.reviewIncorrectQuizQuestions}
            onRetryIncorrect={isPreview ? () => setPreviewState((prev) => ({ ...prev, screen: 'quiz' })) : composer.retryIncorrectQuizQuestions}
          />
        ) : activeScreen === 'quiz-review' ? (
          <QuizReviewPage
            studySet={activeStudySet}
            onBackToResults={isPreview ? () => setPreviewState((prev) => ({ ...prev, screen: 'quiz-results' })) : composer.returnToQuizResults}
            onMove={composer.moveQuizReview}
            onRetryIncorrect={isPreview ? () => setPreviewState((prev) => ({ ...prev, screen: 'quiz' })) : composer.retryIncorrectQuizQuestions}
          />
        ) : (
          <StudyWorkspace
            studySet={activeStudySet}
            onStartNew={isPreview ? () => setPreviewState((prev) => ({ ...prev, screen: 'creation' })) : composer.startNewStudySet}
            onStartFlashcards={isPreview ? () => setPreviewState((prev) => ({ ...prev, screen: 'flashcards' })) : composer.openFlashcards}
            onStartQuiz={isPreview ? () => setPreviewState((prev) => ({ ...prev, screen: 'quiz' })) : composer.openQuiz}
          />
        )}
      </StudyStage> : (
        <main id="workspace" className="pb-14 sm:pb-20">
          <section className="creation-hero">
            <div className="mx-auto grid max-w-6xl items-center gap-3 px-5 pb-20 pt-11 sm:px-8 sm:pb-24 sm:pt-14 lg:grid-cols-[1.05fr_0.95fr] lg:gap-12 lg:py-14">
              <div className="relative z-10 max-w-2xl">
                <p className="flex items-center gap-2 text-[11px] font-bold uppercase tracking-[0.2em] text-sage">
                  <span className="h-px w-7 bg-gold" /> A quieter way to learn
                </p>
                <h1 className="font-display mt-5 text-[2.65rem] font-medium leading-[1.08] tracking-tight text-ivory sm:text-5xl lg:text-[3.65rem]">
                  Turn what you study into <span className="hero-highlight">what you know.</span>
                </h1>
                <p className="mt-5 max-w-xl text-sm leading-7 text-sage sm:text-base sm:leading-8">
                  Bring your notes and documents together. Discover key concepts, practise what matters, and make learning stick.
                </p>
                <div className="mt-7 flex flex-col gap-3 sm:flex-row">
                  <a href="#create-study-set" className="btn btn--primary">
                    Create your study set <ArrowRight size={16} aria-hidden="true" />
                  </a>
                  <a href="#create-study-set" className="btn btn--secondary">See how it works</a>
                </div>
              </div>
              <div className="-mb-9 mt-2 w-full sm:mt-0 lg:mb-0">
                <LearningIllustration />
              </div>
            </div>
          </section>

          <div id="create-study-set" className="relative z-10 mx-auto -mt-7 max-w-5xl px-4 sm:-mt-10 sm:px-8">
            <section className="study-paper rounded-2xl p-5 sm:rounded-3xl sm:p-8">
              <div className="mb-6 flex flex-col gap-4 border-b border-line pb-5 sm:flex-row sm:items-end sm:justify-between">
                <div>
                  <p className="text-[11px] font-bold uppercase tracking-[0.18em] text-gold-ink">Start with what you have</p>
                  <h2 className="font-display mt-1 text-2xl text-deep sm:text-3xl">Build your study set</h2>
                </div>
                <p className="max-w-xs text-xs leading-5 text-muted">Your material stays yours. Only the selected source is sent for processing.</p>
              </div>
              <MaterialInput
                inputMode={composer.inputMode}
                onInputModeChange={composer.changeInputMode}
                text={composer.text}
                onTextChange={composer.updateText}
                file={composer.file}
                onFileSelect={composer.chooseFile}
                onFileRemove={composer.clearFile}
                message={composer.message}
                disabled={composer.isSubmitting}
              >
                <div className="mt-2 flex flex-col gap-3 border-t border-line pt-5 sm:flex-row sm:items-center sm:gap-4">
                  <button
                    type="button"
                    onClick={composer.createStudySet}
                    disabled={composer.isSubmitting}
                    aria-busy={composer.isSubmitting}
                    className="btn btn--primary w-full sm:w-auto"
                  >
                    {composer.isSubmitting ? (
                      <>
                        <LoaderCircle size={18} className="animate-spin text-gold" aria-hidden="true" />
                        Preparing your study material...
                      </>
                    ) : (
                      <>
                        Create Study Set <ArrowRight size={17} aria-hidden="true" />
                      </>
                    )}
                  </button>
                  {!composer.isSubmitting && (
                    <p className="text-center text-xs font-medium text-muted sm:text-left">Ready in a moment. Your notes and selected file remain separate.</p>
                  )}
                </div>
              </MaterialInput>
            </section>
          </div>
        </main>
      )}
      <footer className="app-footer"><span className="font-display">studywise</span><span>A quieter way to learn.</span></footer>
    </div>
  )
}
