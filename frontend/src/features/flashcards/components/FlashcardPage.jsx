import { useEffect, useState } from 'react'
import { ArrowLeft, BookOpenCheck, LoaderCircle, RotateCcw } from 'lucide-react'
import FlashcardControls from './FlashcardControls.jsx'
import FlashcardProgress from './FlashcardProgress.jsx'
import FlashcardResults from './FlashcardResults.jsx'
import FlashcardView from './FlashcardView.jsx'

export default function FlashcardPage({
  studySet, isLoading, error, onRetry, onBack,
  onSetCurrentCard, onSetReviewMode, onMarkCard,
}) {
  const [isRevealed, setIsRevealed] = useState(false)
  const cards = studySet.flashcardReviewMode === 'practice'
    ? studySet.flashcards.filter((card) => studySet.flashcardStatuses[card.id] === 'practice')
    : studySet.flashcards
  const currentIndex = cards.findIndex((card) => card.id === studySet.flashcardCurrentId)
  const safeIndex = Math.max(0, currentIndex)
  const currentCard = cards[safeIndex] || null
  const knownCount = studySet.flashcards.filter((card) => studySet.flashcardStatuses[card.id] === 'known').length
  const practiceCount = studySet.flashcards.filter((card) => studySet.flashcardStatuses[card.id] === 'practice').length

  useEffect(() => setIsRevealed(false), [currentCard?.id])

  function moveTo(index) {
    const next = cards[index]
    if (!next) return
    setIsRevealed(false)
    onSetCurrentCard(next.id)
  }

  function beginPracticeReview() {
    onSetReviewMode('practice')
    const firstPracticeCard = studySet.flashcards.find((card) => studySet.flashcardStatuses[card.id] === 'practice')
    if (firstPracticeCard) onSetCurrentCard(firstPracticeCard.id)
  }

  function returnToAllCards() {
    onSetReviewMode('all')
  }

  return (
    <main className="mx-auto max-w-5xl px-4 pb-14 pt-6 sm:px-7 sm:pb-20 sm:pt-8">
      <div className="flex flex-col gap-4 border-b border-line pb-5 sm:flex-row sm:items-center sm:justify-between">
        <button
          type="button"
          onClick={onBack}
          className="inline-flex min-h-10 w-fit items-center gap-2 rounded-full border border-line bg-paper px-4 py-2 text-sm font-bold text-forest transition hover:border-forest hover:bg-forest hover:text-ivory focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-gold-ink"
        >
          <ArrowLeft size={17} aria-hidden="true" /> Back to Workspace
        </button>
        <p className="flex items-center gap-2 truncate text-sm font-semibold text-muted">
          <BookOpenCheck size={17} className="shrink-0 text-gold-ink" aria-hidden="true" />
          <span className="truncate">Studywise</span>
        </p>
      </div>

      {isLoading ? (
        <section className="mt-8 flex min-h-72 flex-col items-center justify-center rounded-2xl border-2 border-forest bg-forest px-5 py-12 text-center text-sage" role="status" aria-live="polite">
          <LoaderCircle size={28} className="animate-spin text-gold" aria-hidden="true" />
          <h1 className="mt-5 text-xl font-bold text-sage">Creating your flashcards...</h1>
          <p className="mt-2 max-w-md text-sm font-medium text-sage">We’re turning your study material into question and answer cards.</p>
        </section>
      ) : error ? (
        <section className="mt-8 rounded-2xl border-2 border-forest bg-forest px-5 py-10 text-center text-sage sm:px-8" aria-labelledby="flashcard-error-title">
          <h1 id="flashcard-error-title" className="text-xl font-bold text-sage">Flashcards could not be created</h1>
          <p role="alert" className="mx-auto mt-3 max-w-xl text-sm font-medium text-sage">{error}</p>
          <div className="mt-6 flex flex-col justify-center gap-3 sm:flex-row">
            <button
              type="button"
              onClick={onRetry}
              className="inline-flex min-h-11 items-center justify-center gap-2 rounded-xl border-2 border-gold bg-deep px-5 text-sm font-bold text-gold transition hover:bg-gold hover:text-deep focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-gold"
            >
              <RotateCcw size={16} aria-hidden="true" /> Retry
            </button>
            <button
              type="button"
              onClick={onBack}
              className="min-h-11 rounded-xl border-2 border-forest bg-sage px-5 text-sm font-bold text-deep transition hover:bg-forest hover:text-sage focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-gold"
            >
              Back to Workspace
            </button>
          </div>
        </section>
      ) : currentCard ? (
        <section className="mx-auto mt-7 max-w-3xl space-y-5 sm:mt-9 sm:space-y-6" aria-labelledby="flashcards-heading">
          <div className="text-center">
            <p className="text-[11px] font-bold uppercase tracking-[0.18em] text-gold-ink">Active recall <span className="px-1 text-gold">/</span> Studywise</p>
            <h1 id="flashcards-heading" className="font-display mt-1 text-3xl font-medium tracking-tight text-deep sm:text-4xl">Flashcards</h1>
          </div>

          <FlashcardResults
            mode={studySet.flashcardReviewMode}
            practiceCount={practiceCount}
            onShowAll={returnToAllCards}
            onReviewPractice={beginPracticeReview}
          />

          <FlashcardProgress current={safeIndex + 1} total={cards.length} knownCount={knownCount} practiceCount={practiceCount} />
          <FlashcardView card={currentCard} cardNumber={safeIndex + 1} isRevealed={isRevealed} onFlip={() => setIsRevealed((value) => !value)} />
          <FlashcardControls
            canGoPrevious={safeIndex > 0}
            canGoNext={safeIndex < cards.length - 1}
            onPrevious={() => moveTo(safeIndex - 1)}
            onNext={() => moveTo(safeIndex + 1)}
            status={studySet.flashcardStatuses[currentCard.id]}
            onMarkKnown={() => onMarkCard(currentCard.id, 'known')}
            onMarkPractice={() => onMarkCard(currentCard.id, 'practice')}
          />
        </section>
      ) : studySet.flashcardReviewMode === 'practice' ? (
        <section className="mt-8 rounded-2xl border-2 border-forest bg-forest px-5 py-10 text-center text-sage sm:px-8" aria-labelledby="practice-empty-title">
          <h1 id="practice-empty-title" className="text-xl font-bold text-sage">No cards to practice right now</h1>
          <p className="mt-2 text-sm font-medium text-sage">Mark cards as Need Practice while reviewing the full set to add them here.</p>
          <button
            type="button"
            onClick={returnToAllCards}
            className="mt-5 min-h-11 rounded-xl border-2 border-gold bg-deep px-5 text-sm font-bold text-gold transition hover:bg-gold hover:text-deep focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-gold"
          >
            Back to all cards
          </button>
        </section>
      ) : null}
    </main>
  )
}
