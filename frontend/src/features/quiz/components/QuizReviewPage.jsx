import { ArrowLeft, ArrowRight, Check, CircleX, RotateCcw } from 'lucide-react'
import QuizProgress from './QuizProgress.jsx'

export default function QuizReviewPage({ studySet, onBackToResults, onMove, onRetryIncorrect }) {
  const review = studySet.quizReview
  const result = studySet.quizAttempts.find((attempt) => attempt.id === review.resultId)
  const questionId = review.questionIds[review.index]
  const question = studySet.quiz.find((item) => item.id === questionId)
  const submittedAnswer = result?.answers[questionId]?.answer
  const total = review.questionIds.length

  return (
    <main className="mx-auto max-w-4xl px-4 pb-14 pt-6 sm:px-7 sm:pb-20 sm:pt-8">
      <div className="flex flex-col gap-4 border-b border-line pb-5 sm:flex-row sm:items-center sm:justify-between">
        <button type="button" onClick={onBackToResults} className="inline-flex min-h-10 w-fit items-center gap-2 rounded-full border border-line bg-paper px-4 py-2 text-sm font-bold text-forest transition hover:border-forest hover:bg-forest hover:text-ivory focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-gold-ink">
          <ArrowLeft size={17} aria-hidden="true" /> Back to Results
        </button>
        <p className="truncate text-sm font-semibold text-muted">Studywise</p>
      </div>

      {question && (
        <section className="mt-8 space-y-5 sm:mt-10 sm:space-y-6" aria-labelledby="quiz-review-heading">
          <div>
            <p className="text-[11px] font-bold uppercase tracking-[0.18em] text-gold-ink">Learn from the misses</p>
            <h1 id="quiz-review-heading" className="font-display mt-1 text-3xl font-medium tracking-tight text-deep sm:text-4xl">Review incorrect answers</h1>
            <p className="mt-2 text-sm text-muted">Take another look at your answer and the reasoning behind the correct one.</p>
          </div>
          <QuizProgress current={review.index + 1} total={total} answered={total - review.index} statusLabel="to review" />

          <article className="overflow-hidden rounded-3xl bg-paper text-ink">
            <div className="p-5 sm:p-8">
              <p className="text-[10px] font-bold uppercase tracking-[0.17em] text-gold-ink">Question {review.index + 1}</p>
              <h2 className="font-display mt-2 text-2xl font-medium leading-snug text-deep sm:text-3xl">{question.question}</h2>
              <ul className="mt-6 space-y-3" aria-label="Answer options">
                {question.options.map((option) => {
                  const correct = option === question.correctAnswer
                  const selectedWrong = option === submittedAnswer && !correct
                  const state = correct ? 'correct answer' : selectedWrong ? 'your incorrect answer' : 'answer option'
                  return (
                    <li key={`${question.id}-${option}`} className={`flex min-h-14 items-center gap-3.5 rounded-2xl px-4 py-3.5 ${correct ? 'bg-success text-ivory' : selectedWrong ? 'bg-error text-ivory' : 'bg-ivory text-ink'}`}>
                      <span aria-hidden="true" className={`grid h-8 w-8 shrink-0 place-items-center rounded-full ${correct || selectedWrong ? 'bg-ivory text-deep' : 'bg-paper text-muted'}`}>
                        {correct ? <Check size={16} /> : selectedWrong ? <CircleX size={16} /> : null}
                      </span>
                      <span className="min-w-0 flex-1 text-sm font-medium leading-6">{option}</span>
                      <span className="shrink-0 text-right text-[10px] font-bold uppercase tracking-wide">{state}</span>
                    </li>
                  )
                })}
              </ul>

              <section className="mt-6 rounded-2xl bg-forest p-5 text-ivory sm:p-6" aria-labelledby="review-explanation">
                <p className="text-sm leading-6 text-sage"><span className="font-bold text-ivory">Your answer:</span> {submittedAnswer || 'No answer was submitted.'}</p>
                <h3 id="review-explanation" className="mt-4 text-xs font-bold uppercase tracking-[0.16em] text-gold">Why this is correct</h3>
                <p className="mt-2 text-sm leading-6 text-ivory">{question.explanation}</p>
              </section>
            </div>

            <div className="flex flex-col gap-4 border-t border-line px-5 py-4 sm:flex-row sm:items-center sm:justify-between sm:px-8">
              <button type="button" disabled={review.index === 0} onClick={() => onMove(review.index - 1)} className="inline-flex min-h-10 items-center justify-center gap-2 rounded-full border border-line px-4 text-sm font-bold text-forest transition hover:bg-ivory focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-gold-ink disabled:cursor-not-allowed disabled:text-muted">
                <ArrowLeft size={16} aria-hidden="true" /> Previous
              </button>
              <span className="font-mono text-xs font-bold tabular-nums text-muted">{review.index + 1} / {total}</span>
              <button type="button" disabled={review.index >= total - 1} onClick={() => onMove(review.index + 1)} className="inline-flex min-h-10 items-center justify-center gap-2 rounded-full border border-line px-4 text-sm font-bold text-forest transition hover:bg-ivory focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-gold-ink disabled:cursor-not-allowed disabled:text-muted">
                Next <ArrowRight size={16} aria-hidden="true" />
              </button>
            </div>
          </article>

          <div className="flex justify-center">
            <button type="button" onClick={onRetryIncorrect} className="inline-flex min-h-11 items-center justify-center gap-2 rounded-full bg-deep px-6 text-sm font-bold text-ivory transition hover:bg-forest focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-gold-ink">
              <RotateCcw size={16} aria-hidden="true" /> Retry incorrect questions
            </button>
          </div>
        </section>
      )}
    </main>
  )
}
