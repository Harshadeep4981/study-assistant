import { Check, X } from 'lucide-react'

export default function QuizOptions({ question, selectedAnswer, submittedAnswer, onSelect }) {
  const submitted = Boolean(submittedAnswer)

  return (
    <fieldset className="space-y-3">
      <legend className="mb-3 text-sm font-bold text-sage">Choose one answer</legend>
      {question.options.map((option, index) => {
        const selected = selectedAnswer === option
        const correct = submitted && option === question.correctAnswer
        const selectedIncorrect = submitted && selected && !submittedAnswer.isCorrect

        const state = correct ? 'correct' : selectedIncorrect ? 'incorrect' : selected ? 'selected' : 'default'

        return (
          <label
            key={`${question.id}-${option}`}
            data-state={state}
            data-submitted={submitted}
            className={`question-option flex cursor-pointer items-center gap-4 rounded-[18px] px-4 py-4 transition-all focus-within:outline focus-within:outline-2 focus-within:outline-offset-2 focus-within:outline-white ${submitted ? 'cursor-default' : ''}`}
          >
            <input
              type="radio"
              name={`quiz-answer-${question.id}`}
              value={option}
              checked={selected}
              disabled={submitted}
              onChange={() => onSelect(option)}
              className="sr-only"
            />
            <span
              aria-hidden="true"
              className={`grid h-10 w-10 shrink-0 place-items-center rounded-full border ${
                correct
                  ? 'border-success bg-success text-ivory'
                  : selectedIncorrect
                    ? 'border-error bg-error text-ivory'
                    : selected
                      ? 'border-gold bg-deep text-ivory'
                      : 'border-moss bg-paper text-muted'
              }`}
            >
              {correct ? (
                <Check size={14} />
              ) : selectedIncorrect ? (
                <X size={14} />
              ) : (
                <span className="text-[11px] font-bold">{String.fromCharCode(65 + index)}</span>
              )}
            </span>
            <span className="min-w-0 flex-1 text-sm font-medium leading-6">{option}</span>
            {correct && <span className="shrink-0 text-xs font-bold text-ivory">Correct answer</span>}
            {selectedIncorrect && <span className="shrink-0 text-xs font-bold text-ivory">Your answer</span>}
            {!submitted && <span className="sr-only">Option {index + 1}</span>}
          </label>
        )
      })}
    </fieldset>
  )
}
