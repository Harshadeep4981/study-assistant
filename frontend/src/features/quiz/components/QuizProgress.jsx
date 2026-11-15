export default function QuizProgress({ current, total, answered, statusLabel = 'answered' }) {
  const percent = total > 0 ? Math.round((current / total) * 100) : 0

  return (
    <section aria-label="Quiz progress" className="space-y-2.5">
      <div className="flex items-center justify-between gap-4 text-sm">
        <p className="font-bold text-deep">Question {current} of {total}</p>
        <p className="text-xs font-semibold text-forest">{answered} {statusLabel}</p>
      </div>
      <div
        role="progressbar"
        aria-label="Quiz position"
        aria-valuemin={0}
        aria-valuemax={total}
        aria-valuenow={current}
        aria-valuetext={`Question ${current} of ${total}`}
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
