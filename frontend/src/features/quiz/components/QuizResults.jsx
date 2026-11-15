import { ArrowLeft, ArrowRight, Check, RotateCcw, Trophy } from 'lucide-react'

export default function QuizResults({ studySet, onBack, onReviewIncorrect, onRetryIncorrect }) {
  const attempts = studySet.quizAttempts
  const latest = attempts.at(-1)
  const previous = attempts.at(-2)
  const incorrectCount = latest?.incorrectQuestionIds.length || 0
  const scoreMessage = latest?.percentage >= 80 ? 'Excellent recall.' : latest?.percentage >= 50 ? 'Good progress. Keep practising.' : 'A solid start. Review and try again.'

  return (
    <main className="mx-auto max-w-5xl px-4 pb-14 pt-6 sm:px-7 sm:pb-20 sm:pt-8">
      <div className="flex flex-col gap-4 border-b border-line pb-5 sm:flex-row sm:items-center sm:justify-between">
        <button type="button" onClick={onBack} className="btn btn--secondary min-h-10 w-fit px-4 text-sm">
          <ArrowLeft size={17} aria-hidden="true" /> Back to Workspace
        </button>
        <p className="truncate text-sm font-semibold text-muted">Studywise</p>
      </div>

      <section className="mt-7 overflow-hidden rounded-3xl bg-deep text-ivory sm:mt-9" aria-labelledby="quiz-results-heading">
        <div className="grid gap-8 p-6 sm:p-9 lg:grid-cols-[minmax(0,1fr)_auto] lg:items-center lg:p-12">
          <div>
            <span className="grid h-12 w-12 place-items-center rounded-2xl bg-gold text-deep"><Trophy size={23} aria-hidden="true" /></span>
            <p className="mt-6 text-[11px] font-bold uppercase tracking-[0.18em] text-gold">A moment to reflect</p>
            <h1 id="quiz-results-heading" className="font-display mt-2 text-4xl font-medium tracking-tight text-ivory sm:text-5xl">Quiz complete</h1>
            <p className="mt-3 text-sm leading-6 text-sage">{scoreMessage}</p>
          </div>
          <div className="score-ring relative grid h-40 w-40 place-items-center text-center sm:h-48 sm:w-48">
            <svg aria-hidden="true" viewBox="0 0 120 120" className="absolute inset-0 h-full w-full -rotate-90"><defs><linearGradient id="score-gradient"><stop stopColor="var(--aurora-emerald)"/><stop offset="1" stopColor="var(--aurora-teal)"/></linearGradient></defs><circle cx="60" cy="60" r="52" fill="none" stroke="rgba(255,255,255,.12)" strokeWidth="7"/><circle cx="60" cy="60" r="52" fill="none" stroke="url(#score-gradient)" strokeWidth="7" strokeLinecap="round" strokeDasharray={`${Math.max(0, Math.min(100, latest?.percentage ?? 0)) * 3.267} 326.7`}/></svg>
            <div><p className="font-display text-5xl leading-none text-ivory sm:text-6xl">{latest?.percentage ?? 0}<span className="text-2xl">%</span></p><p className="mt-2 text-[10px] font-bold uppercase tracking-[0.16em] text-sage">Your score</p></div>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-px bg-moss sm:grid-cols-4">
          <div className="bg-forest p-4 text-center sm:p-5"><p className="font-mono text-2xl font-bold text-ivory">{latest?.total ?? 0}</p><p className="mt-1 text-xs font-semibold text-sage">Questions</p></div>
          <div className="bg-forest p-4 text-center sm:p-5"><p className="font-mono text-2xl font-bold text-sage">{latest?.correctCount ?? 0}</p><p className="mt-1 text-xs font-semibold text-sage">Correct</p></div>
          <div className="bg-forest p-4 text-center sm:p-5"><p className="font-mono text-2xl font-bold text-ivory">{incorrectCount}</p><p className="mt-1 text-xs font-semibold text-sage">Incorrect</p></div>
          <div className="bg-forest p-4 text-center sm:p-5"><p className="font-mono text-2xl font-bold text-gold">{latest?.percentage ?? 0}%</p><p className="mt-1 text-xs font-semibold text-sage">Score</p></div>
        </div>

        <div className="flex flex-col gap-4 p-5 sm:flex-row sm:items-center sm:justify-between sm:px-9 sm:py-6 lg:px-12">
          <p className="text-sm font-medium text-sage">{latest?.correctCount ?? 0} of {latest?.total ?? 0} correct · {incorrectCount} incorrect {incorrectCount === 1 ? 'answer' : 'answers'}</p>
          {incorrectCount > 0 ? (
            <div className="flex flex-col gap-2 sm:flex-row">
              <button type="button" onClick={() => onReviewIncorrect(latest.id)} className="btn btn--secondary min-h-11 px-5 text-sm">
                <Check size={16} aria-hidden="true" /> Review incorrect answers
              </button>
              <button type="button" onClick={onRetryIncorrect} className="btn btn--primary min-h-11 px-5 text-sm">
                <RotateCcw size={15} aria-hidden="true" /> Retry incorrect questions
              </button>
            </div>
          ) : (
            <p className="inline-flex items-center gap-2 rounded-full bg-sage px-4 py-2 text-sm font-bold text-deep"><Check size={16} aria-hidden="true" /> All answers correct</p>
          )}
        </div>
      </section>

      {previous && (
        <section className="mt-5 flex flex-col gap-3 rounded-2xl bg-paper p-5 text-ink sm:flex-row sm:items-center sm:justify-between" aria-label="Previous quiz attempt">
          <div><h2 className="text-sm font-bold text-deep">Previous attempt</h2><p className="mt-1 text-sm font-medium text-muted">{previous.correctCount} of {previous.total} correct · {previous.percentage}%</p></div>
          {previous.incorrectQuestionIds.length > 0 && <button type="button" onClick={() => onReviewIncorrect(previous.id)} className="inline-flex min-h-10 items-center gap-2 text-sm font-bold text-forest hover:text-gold-ink focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-gold-ink">Review previous mistakes <ArrowRight size={15} aria-hidden="true" /></button>}
        </section>
      )}

      <div className="mt-7 flex justify-center">
        <button type="button" onClick={onBack} className="btn btn--secondary min-h-11 w-full px-5 text-sm sm:w-auto">
          <ArrowLeft size={16} aria-hidden="true" /> Back to Study Workspace
        </button>
      </div>
    </main>
  )
}
