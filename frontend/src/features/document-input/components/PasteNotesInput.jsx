import { useRef } from 'react'

export default function PasteNotesInput({ text, onTextChange, disabled = false }) {
  const textAreaRef = useRef(null)
  return (
    <div className="space-y-2.5">
      <div className="flex items-center justify-between gap-3">
        <label htmlFor="study-notes" className="text-sm font-bold text-ink">Study notes or topic</label>
        <span className="text-xs tabular-nums font-semibold text-gold-ink" aria-live="polite">{text.length.toLocaleString()} characters</span>
      </div>
      <textarea
        ref={textAreaRef}
        id="study-notes"
        value={text}
        disabled={disabled}
        onChange={(event) => onTextChange(event.target.value)}
        placeholder="Paste your notes, add a reading excerpt, or enter a topic you want to study..."
        className="min-h-36 w-full resize-y rounded-xl border border-line bg-ivory px-4 py-3.5 text-sm leading-6 text-ink placeholder:text-muted focus:border-gold-ink focus:outline-none sm:min-h-40 disabled:cursor-not-allowed"
      />
      {text && (
        <div className="flex justify-end">
          <button
            type="button"
            disabled={disabled}
            onClick={() => onTextChange('')}
            className="rounded-full border border-line px-3 py-1.5 text-xs font-bold text-forest transition hover:border-forest hover:bg-forest hover:text-ivory focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-gold-ink disabled:cursor-not-allowed"
          >
            Clear notes
          </button>
        </div>
      )}
    </div>
  )
}
