import { Check, CircleX } from 'lucide-react'

export default function QuizFeedback({ answer, correctAnswer, explanation }) {
  const isCorrect = answer === correctAnswer

  return (
    <section role="status" aria-live="polite" className={`rounded-2xl p-4 text-ivory sm:p-5 ${isCorrect ? 'bg-success' : 'bg-error'}`}>
      <div className="flex items-start gap-3">
        <span className="mt-0.5 grid h-8 w-8 shrink-0 place-items-center rounded-full bg-ivory text-deep">
          {isCorrect ? <Check size={18} aria-hidden="true" /> : <CircleX size={18} aria-hidden="true" />}
        </span>
        <div className="min-w-0">
          <h3 className="text-sm font-bold text-ivory">{isCorrect ? "That's correct." : 'Not quite.'}</h3>
          {!isCorrect && <p className="mt-1 text-sm font-medium leading-6 text-ivory"><span className="font-bold">Correct answer:</span> {correctAnswer}</p>}
          <p className="mt-2 text-sm font-medium leading-6 text-ivory"><span className="font-bold">Explanation:</span> {explanation}</p>
        </div>
      </div>
    </section>
  )
}
