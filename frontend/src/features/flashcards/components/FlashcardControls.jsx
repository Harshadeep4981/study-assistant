import { ArrowLeft, ArrowRight, Check, RotateCcw } from 'lucide-react'

function masteryClass(selected, type) {
  const base = 'inline-flex min-h-11 items-center justify-center gap-2 rounded-full border px-4 text-sm font-bold transition focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-gold-ink'
  if (type === 'known') {
    return `${base} btn btn--ok`
  }
  return `${base} btn btn--practice ${selected ? 'ring-2 ring-amber-300' : ''}`
}

export default function FlashcardControls({
  canGoPrevious, canGoNext, onPrevious, onNext,
  status, onMarkKnown, onMarkPractice,
}) {
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3">
        <button
          type="button"
          disabled={!canGoPrevious}
          onClick={onPrevious}
          className="btn btn--secondary min-h-11 px-4 text-sm"
        >
          <ArrowLeft size={16} aria-hidden="true" /> Previous
        </button>
        <button
          type="button"
          disabled={!canGoNext}
          onClick={onNext}
          className="btn btn--primary min-h-11 px-4 text-sm"
        >
          Next <ArrowRight size={16} aria-hidden="true" />
        </button>
      </div>
      <div className="rounded-2xl bg-paper p-4 text-ink sm:p-5">
        <p className="text-center text-sm font-bold text-ink">How well did you know this?</p>
        <div className="mt-3 grid gap-2.5 sm:grid-cols-2">
          <button
            type="button"
            aria-pressed={status === 'known'}
            onClick={onMarkKnown}
            className={masteryClass(status === 'known', 'known')}
          >
            <Check size={16} aria-hidden="true" /> Know it
          </button>
          <button
            type="button"
            aria-pressed={status === 'practice'}
            onClick={onMarkPractice}
            className={masteryClass(status === 'practice', 'practice')}
          >
            <RotateCcw size={15} aria-hidden="true" /> Need Practice
          </button>
        </div>
      </div>
    </div>
  )
}
