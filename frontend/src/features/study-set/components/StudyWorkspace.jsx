import { AlignLeft, ArrowRight, BookOpenCheck, CircleHelp, FileText, Plus, ShieldCheck } from 'lucide-react'

export default function StudyWorkspace({ studySet, onStartNew, onStartFlashcards, onStartQuiz }) {
  const isPastedText = studySet.sourceType === 'pasted_text'
  const sourceLabel = isPastedText ? 'Pasted notes' : studySet.sourceType.toUpperCase()
  const sourceName = studySet.sourceName || 'Notes entered in Studywise'
  const characterCount = studySet.metadata.character_count ?? studySet.extractedText.length
  const createdAt = new Date(studySet.createdAt).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })

  return (
    <main id="workspace" className="mx-auto max-w-6xl px-4 pb-14 pt-7 sm:px-7 sm:pb-20 sm:pt-10">
      <section className="relative overflow-hidden rounded-3xl bg-deep p-5 text-ivory sm:p-8 lg:p-10">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
          <div className="min-w-0">
            <p className="text-[11px] font-bold uppercase tracking-[0.18em] text-gold">Your active study set</p>
            <h1 className="font-display mt-2 break-words text-3xl font-medium leading-tight tracking-tight text-ivory sm:text-4xl">Your study material</h1>
            <p className="mt-2 text-sm font-medium text-sage">Created {createdAt}</p>
          </div>
          <span className="inline-flex w-fit shrink-0 items-center gap-2 rounded-full bg-sage px-3.5 py-2 text-xs font-bold text-deep">
            <ShieldCheck size={16} aria-hidden="true" /> Processed and ready
          </span>
        </div>

        <div className="mt-7 grid gap-4 rounded-2xl border border-moss bg-forest p-4 sm:grid-cols-[1fr_auto] sm:items-center sm:p-5">
          <div className="flex min-w-0 items-start gap-3">
            <span className="mt-0.5 grid h-10 w-10 shrink-0 place-items-center rounded-xl bg-deep text-gold">
              {isPastedText ? <AlignLeft size={17} aria-hidden="true" /> : <FileText size={17} aria-hidden="true" />}
            </span>
            <div className="min-w-0">
              <p className="text-[10px] font-bold uppercase tracking-[0.16em] text-sage">Source · {sourceLabel}</p>
              <p className="mt-1 break-words text-sm font-semibold text-ivory">{sourceName}</p>
            </div>
          </div>
          <p className="pl-12 font-mono text-xs font-bold tabular-nums text-sage sm:pl-0">{characterCount.toLocaleString()} characters</p>
        </div>

        {studySet.warnings.length > 0 && (
          <ul className="mt-4 space-y-2" aria-label="Document processing notices">
            {studySet.warnings.map((warning) => (
              <li key={warning} className="rounded-xl border-2 border-gold bg-deep p-3 text-sm font-semibold text-sage">
                {warning}
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="mt-10 sm:mt-14" aria-labelledby="ready-heading">
        <div className="mb-6 flex flex-col gap-3 sm:mb-8 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <p className="text-[11px] font-bold uppercase tracking-[0.18em] text-gold-ink">Your next step</p>
            <h2 id="ready-heading" className="font-display mt-1 text-3xl font-medium tracking-tight text-deep sm:text-4xl">Choose how you want to learn.</h2>
          </div>
          <p className="max-w-xs text-sm leading-6 text-muted">Two ways to practise. Pick the one that fits your moment.</p>
        </div>
        <div className="grid gap-4 sm:grid-cols-2 sm:gap-5">
          <article className="study-mode-card group relative flex min-h-72 flex-col justify-between overflow-hidden rounded-3xl p-6 sm:p-8">
            <div>
              <div className="flex items-start justify-between">
                <span className="grid h-12 w-12 place-items-center rounded-2xl bg-gold text-deep">
                  <BookOpenCheck size={23} aria-hidden="true" />
                </span>
                <span className="font-display text-5xl leading-none text-moss">01</span>
              </div>
              <h3 className="font-display mt-8 text-2xl text-ivory">Flashcards</h3>
              <p className="mt-2 max-w-sm text-sm leading-6 text-sage">Strengthen recall one clear question at a time, and keep track of what needs another look.</p>
            </div>
            <button
              type="button"
              onClick={onStartFlashcards}
              className="btn btn--primary mt-7 w-full sm:w-fit"
            >
              Start learning <ArrowRight size={16} aria-hidden="true" />
            </button>
          </article>
          <article className="study-mode-card group flex min-h-72 flex-col justify-between overflow-hidden rounded-3xl p-6 sm:p-8">
            <div>
              <div className="flex items-start justify-between">
                <span className="grid h-12 w-12 place-items-center rounded-2xl bg-deep text-gold">
                  <CircleHelp size={23} aria-hidden="true" />
                </span>
                <span className="font-display text-5xl leading-none text-moss">02</span>
              </div>
              <h3 className="font-display mt-8 text-2xl text-deep">Quick quiz</h3>
              <p className="mt-2 max-w-sm text-sm leading-6 text-forest">Test your understanding with focused questions and clear explanations after every answer.</p>
            </div>
            <button
              type="button"
              onClick={onStartQuiz}
              className="btn btn--primary mt-7 w-full sm:w-fit"
            >
              Take a quiz <ArrowRight size={16} aria-hidden="true" />
            </button>
          </article>
        </div>
      </section>

      <div className="mt-7 flex justify-center sm:mt-8">
        <button
          type="button"
          onClick={onStartNew}
          className="btn btn--secondary w-full sm:w-auto"
        >
          <Plus size={17} aria-hidden="true" /> Start New Study Set
        </button>
      </div>
    </main>
  )
}
