export default function FlashcardProgress({ current, total, knownCount, practiceCount }) {
  const percent = total > 0 ? Math.round((current / total) * 100) : 0

  return (
    <section aria-label="Flashcard progress" className="space-y-2.5">
      <div className="flex items-center justify-between gap-4 text-sm">
        <p className="font-bold text-deep">Card {current} of {total}</p>
        <p className="text-xs font-semibold text-forest">{knownCount} known <span aria-hidden="true">·</span> {practiceCount} to practice</p>
      </div>
      <div
        role="progressbar"
        aria-label="Flashcard position"
        aria-valuemin={0}
        aria-valuemax={total}
        aria-valuenow={current}
        aria-valuetext={`${current} of ${total} cards`}
        className="h-2.5 overflow-hidden rounded-full border border-line bg-paper"
      >
        <div
          className="h-full rounded-full bg-gold-ink transition-[width] duration-200"
          style={{ width: `${percent}%` }}
        />
      </div>
    </section>
  )
}
