export default function FlashcardResults({ mode, practiceCount, onShowAll, onReviewPractice }) {
  return (
    <div className="flex flex-col gap-3 rounded-2xl bg-paper p-4 text-ink sm:flex-row sm:items-center sm:justify-between">
      <div>
        <h2 className="text-sm font-bold text-deep">Your review</h2>
        <p className="mt-1 text-xs leading-5 font-medium text-muted">Cards marked Need Practice stay in the original set and can be reviewed on their own.</p>
      </div>
      {mode === 'practice' ? (
        <button
          type="button"
          onClick={onShowAll}
          className="btn btn--secondary min-h-10 shrink-0 px-4 text-sm"
        >
          Back to all cards
        </button>
      ) : (
        <button
          type="button"
          disabled={practiceCount === 0}
          onClick={onReviewPractice}
          className="btn btn--secondary min-h-10 shrink-0 px-4 text-sm"
        >
          Review Need Practice ({practiceCount})
        </button>
      )}
    </div>
  )
}
