import { BookOpen } from 'lucide-react'

export default function AppHeader({ onWorkspaceClick }) {
  return (
    <header className="app-header sticky top-0 z-40 border-b border-moss bg-deep text-ivory">
      <div className="mx-auto flex h-[4.25rem] max-w-7xl items-center justify-between px-5 sm:px-8">
        <a className="flex items-center gap-3 font-bold tracking-tight text-ivory" href="#home" aria-label="Studywise home">
          <span className="grid h-9 w-9 place-items-center rounded-xl bg-gold text-deep">
            <BookOpen size={18} aria-hidden="true" />
          </span>
          <span className="font-display text-[1.35rem] font-semibold">studywise</span>
        </a>
        <nav aria-label="Main navigation" className="flex items-center gap-3 text-sm font-semibold">
          {onWorkspaceClick ? (
            <button
              type="button"
              onClick={onWorkspaceClick}
              className="inline-flex min-h-10 items-center rounded-full border border-moss px-4 py-2 text-sage transition hover:border-gold hover:bg-forest hover:text-ivory focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-gold"
            >
              Workspace
            </button>
          ) : (
            <a
              className="inline-flex min-h-10 items-center rounded-full border border-moss px-4 py-2 text-sage transition hover:border-gold hover:bg-forest hover:text-ivory focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-gold"
              href="#workspace"
            >
              Workspace
            </a>
          )}
        </nav>
      </div>
    </header>
  )
}
