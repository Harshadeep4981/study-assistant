import { RotateCcw } from 'lucide-react'

export default function FlashcardView({ card, cardNumber, isRevealed, onFlip }) {
  return (
    <button
      type="button"
      onClick={onFlip}
      aria-label={`${isRevealed ? 'Show question' : 'Show answer'} for card ${cardNumber}`}
      aria-pressed={isRevealed}
      className="group flashcard-scene w-full rounded-3xl text-center focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-gold-ink"
    >
      <span className={`flashcard-rotator ${isRevealed ? 'is-flipped' : ''}`}>
        <span aria-hidden={isRevealed} className="flashcard-face study-card-front">
          <span className="rounded-full border border-gold bg-deep px-3.5 py-1 text-xs font-bold uppercase tracking-widest text-gold">Question</span>
          <span className="font-display mt-6 max-w-2xl whitespace-pre-wrap text-xl leading-8 text-ivory sm:text-2xl sm:leading-9">{card.question}</span>
          <span className="mt-7 inline-flex items-center gap-2 text-sm font-semibold text-gold"><RotateCcw size={15} aria-hidden="true" /> Click or press Enter to reveal answer</span>
        </span>
        <span aria-hidden={!isRevealed} className="flashcard-face flashcard-face-back study-card-back">
          <span className="rounded-full bg-ivory px-3.5 py-1 text-xs font-bold uppercase tracking-widest text-deep">Answer</span>
          <span className="font-display mt-6 max-w-2xl whitespace-pre-wrap text-xl leading-8 text-deep sm:text-2xl sm:leading-9">{card.answer}</span>
          <span className="mt-7 inline-flex items-center gap-2 text-sm font-semibold text-forest"><RotateCcw size={15} aria-hidden="true" /> Click or press Enter to show question</span>
        </span>
      </span>
    </button>
  )
}
