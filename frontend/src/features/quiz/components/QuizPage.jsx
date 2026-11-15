import { ArrowLeft, ArrowRight, Check, CircleHelp, LoaderCircle, RotateCcw, X } from 'lucide-react'
import QuizFeedback from './QuizFeedback.jsx'
import QuizOptions from './QuizOptions.jsx'
import QuizProgress from './QuizProgress.jsx'

export default function QuizPage({
  studySet, isLoading, error, onRetry, onBack,
  onChooseAnswer, onSubmitAnswer, onMoveQuestion, onContinue,
}) {
  const attempt = studySet.quizAttempt
  const questionId = attempt?.questionIds[attempt.currentIndex]
  const question = studySet.quiz.find((item) => item.id === questionId)
  const selectedAnswer = questionId ? attempt?.selectedAnswers[questionId] || '' : ''
  const submittedAnswer = questionId ? attempt?.submittedAnswers[questionId] : null
  const total = attempt?.questionIds.length || 0
  const current = total > 0 ? attempt.currentIndex + 1 : 0
  const answered = Object.keys(attempt?.submittedAnswers || {}).length
  const remaining = Math.max(0, total - answered)
  const correctCount = Object.values(attempt?.submittedAnswers || {}).filter((answer) => answer.isCorrect).length
  const wrongCount = answered - correctCount

  return (
    <main className="mx-auto max-w-7xl px-4 pb-14 pt-6 sm:px-7 sm:pb-20 sm:pt-8">
      <div className="flex flex-col gap-4 border-b border-line pb-5 sm:flex-row sm:items-center sm:justify-between">
        <button type="button" onClick={onBack} className="btn btn--secondary min-h-11 w-fit px-4 text-sm">
          <ArrowLeft size={17} aria-hidden="true" /> Back to Workspace
        </button>
        <p className="flex min-w-0 items-center gap-2 text-sm font-semibold text-muted">
          <CircleHelp size={17} className="shrink-0 text-gold-ink" aria-hidden="true" />
          <span className="truncate">Studywise</span>
        </p>
      </div>

      {isLoading ? (
        <section className="mt-8 flex min-h-72 flex-col items-center justify-center rounded-3xl bg-deep px-5 py-12 text-center text-ivory" role="status" aria-live="polite">
          <LoaderCircle size={28} className="animate-spin text-gold" aria-hidden="true" />
          <h1 className="font-display mt-5 text-2xl">Building your quiz...</h1>
          <p className="mt-2 max-w-md text-sm leading-6 text-sage">We're preparing questions from your study material.</p>
        </section>
      ) : error ? (
        <section className="mt-8 rounded-3xl bg-deep px-5 py-10 text-center text-ivory sm:px-8" aria-labelledby="quiz-error-title">
          <h1 id="quiz-error-title" className="font-display text-2xl">Quiz could not be created</h1>
          <p role="alert" className="mx-auto mt-3 max-w-xl text-sm leading-6 text-sage">{error}</p>
          <div className="mt-6 flex flex-col justify-center gap-3 sm:flex-row">
            <button type="button" onClick={onRetry} className="inline-flex min-h-11 items-center justify-center gap-2 rounded-full bg-gold px-5 text-sm font-bold text-deep transition hover:bg-sage focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-gold-ink">
              <RotateCcw size={16} aria-hidden="true" /> Retry
            </button>
            <button type="button" onClick={onBack} className="min-h-11 rounded-full border border-moss px-5 text-sm font-bold text-sage transition hover:bg-forest focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-gold">
              Back to Workspace
            </button>
          </div>
        </section>
      ) : question ? (
        <div className="mt-8 space-y-6 sm:mt-10">
          <div>
            <p className="text-[11px] font-bold uppercase tracking-[0.18em] text-gold-ink">Active recall <span className="px-1 text-gold">/</span> Studywise</p>
            <h1 id="quiz-heading" className="font-display mt-1 text-3xl font-medium tracking-tight text-deep sm:text-4xl">Quick quiz</h1>
          </div>

          <div className="grid items-start gap-5 lg:grid-cols-[minmax(0,7fr)_minmax(16rem,3fr)] lg:gap-6">
            <div className="min-w-0 space-y-5">
              <QuizProgress current={current} total={total} answered={answered} />
              <section className="space-y-6 rounded-3xl bg-deep p-5 text-ivory sm:p-8 lg:p-10" aria-labelledby="quiz-question">
                <div>
                  <p className="text-[11px] font-bold uppercase tracking-[0.17em] text-gold">Question {current} <span className="px-1 text-moss">/</span> {total}</p>
                  <h2 id="quiz-question" className="font-display mt-3 text-2xl font-medium leading-snug text-ivory sm:text-3xl sm:leading-tight">{question.question}</h2>
                </div>

                <QuizOptions question={question} selectedAnswer={selectedAnswer} submittedAnswer={submittedAnswer} onSelect={(answer) => onChooseAnswer(question.id, answer)} />

                {submittedAnswer && <QuizFeedback answer={submittedAnswer.answer} correctAnswer={question.correctAnswer} explanation={question.explanation} />}

                <div className="flex flex-col-reverse gap-3 border-t border-moss pt-5 sm:flex-row sm:items-center sm:justify-between">
                  <button type="button" disabled={attempt.currentIndex === 0} onClick={() => onMoveQuestion(attempt.currentIndex - 1)} className="btn btn--secondary min-h-11 px-5 text-sm">
                    <ArrowLeft size={16} aria-hidden="true" /> Previous
                  </button>
                  {submittedAnswer ? (
                    <button type="button" onClick={onContinue} className="btn btn--primary min-h-11 px-6 text-sm">
                      {current === total ? 'View Results' : 'Continue'} <ArrowRight size={16} aria-hidden="true" />
                    </button>
                  ) : (
                    <button type="button" disabled={!selectedAnswer} onClick={onSubmitAnswer} className="btn btn--primary min-h-11 px-6 text-sm">
                      Submit Answer <ArrowRight size={16} aria-hidden="true" />
                    </button>
                  )}
                </div>
              </section>
            </div>

            <aside aria-label="Quiz overview" className="space-y-5 p-5 lg:sticky lg:top-6">
              <div>
                <p className="text-[10px] font-bold uppercase tracking-[0.17em] text-gold-ink">Quiz overview</p>
                <h3 className="mt-1 truncate text-base font-bold text-deep">Quiz overview</h3>
              </div>

              <div className="overview-stats grid grid-cols-3 gap-2 p-3 sm:grid-cols-5">
                {[['Total', total, ''], ['Answered', answered, 'stat-answered'], ['Left', remaining, ''], ['Correct', correctCount, 'stat-correct'], ['Wrong', wrongCount, 'stat-wrong']].map(([label, value, color]) => <div key={label} className="text-center"><p className={`font-display text-2xl font-semibold ${color}`}>{value}</p><p className="mt-0.5 text-[10px] font-semibold text-muted">{label}</p></div>)}
              </div>

              <div aria-label={`Score so far: ${correctCount} correct, ${wrongCount} wrong, ${remaining} unanswered`} className="flex h-2 overflow-hidden rounded-full bg-white/10">
                {total > 0 && <span className="bg-emerald-400 transition-[width]" style={{ width: `${correctCount / total * 100}%` }} />}
                {total > 0 && <span className="bg-rose-400 transition-[width]" style={{ width: `${wrongCount / total * 100}%` }} />}
              </div>

              <div>
                <p className="mb-2.5 text-[10px] font-bold uppercase tracking-[0.17em] text-muted">Questions</p>
                <div className="grid grid-cols-5 gap-1.5 sm:grid-cols-6 lg:grid-cols-4">
                  {attempt.questionIds.map((id, index) => {
                    const isCurrent = index === attempt.currentIndex
                    const answer = attempt.submittedAnswers[id]
                    const state = answer ? answer.isCorrect ? 'is-correct' : 'is-wrong' : ''
                    return (
                      <button key={id} type="button" onClick={() => onMoveQuestion(index)} className={`overview-tile ${state} ${isCurrent ? 'is-current' : ''} grid h-12 place-items-center text-xs font-bold transition`} aria-label={`Jump to Question ${index + 1}${isCurrent ? ' (current)' : ''}${answer ? answer.isCorrect ? ' (correct)' : ' (wrong)' : ' (unanswered)'}`}>
                        {answer ? answer.isCorrect ? <><Check size={12} aria-hidden="true" /> <span>{index + 1}</span></> : <><X size={12} aria-hidden="true" /> <span>{index + 1}</span></> : index + 1}
                      </button>
                    )
                  })}
                </div>
                <p className="mt-3 text-[10px] leading-5 text-muted"><span className="legend-correct">●</span> Correct <span className="mx-1 legend-wrong">●</span> Wrong <span className="mx-1 legend-current">◎</span> Current <span className="mx-1 legend-unanswered">●</span> Unanswered</p>
              </div>
            </aside>
          </div>
        </div>
      ) : null}
    </main>
  )
}
