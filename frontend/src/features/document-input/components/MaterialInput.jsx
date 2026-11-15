import { useRef } from 'react'
import DocumentUpload from './DocumentUpload.jsx'
import PasteNotesInput from './PasteNotesInput.jsx'

const tabs = [
  { id: 'paste', label: 'Paste notes' },
  { id: 'upload', label: 'Upload document' },
]

export default function MaterialInput({ inputMode, onInputModeChange, text, onTextChange, file, onFileSelect, onFileRemove, message, disabled = false, children }) {
  const tabRefs = useRef([])

  function handleTabKeyDown(event, index) {
    let nextIndex = index
    if (event.key === 'ArrowRight') nextIndex = (index + 1) % tabs.length
    else if (event.key === 'ArrowLeft') nextIndex = (index + tabs.length - 1) % tabs.length
    else if (event.key === 'Home') nextIndex = 0
    else if (event.key === 'End') nextIndex = tabs.length - 1
    else return
    event.preventDefault()
    onInputModeChange(tabs[nextIndex].id)
    tabRefs.current[nextIndex]?.focus()
  }

  return (
    <section aria-labelledby="material-title">
      <div className="pb-4 sm:pb-5">
        <div className="flex flex-col gap-1 sm:flex-row sm:items-end sm:justify-between sm:gap-4">
          <div>
            <p className="text-[11px] font-bold uppercase tracking-[0.16em] text-gold-ink">Your material</p>
            <h2 id="material-title" className="font-display mt-1 text-xl text-deep sm:text-2xl">What are you studying?</h2>
          </div>
          <p className="text-xs font-medium text-muted">Switch anytime; your other input stays saved.</p>
        </div>
        <div className="mt-4 inline-flex max-w-full rounded-xl bg-paper p-1" role="tablist" aria-label="Study material input type">
          {tabs.map((tab, index) => {
            const selected = tab.id === inputMode
            return (
              <button
                key={tab.id}
                ref={(element) => { tabRefs.current[index] = element }}
                id={`material-tab-${tab.id}`}
                type="button"
                disabled={disabled}
                role="tab"
                aria-selected={selected}
                aria-controls="material-tabpanel"
                tabIndex={selected ? 0 : -1}
                onClick={() => onInputModeChange(tab.id)}
                onKeyDown={(event) => handleTabKeyDown(event, index)}
                className={`min-h-10 rounded-lg px-3.5 text-sm font-bold transition focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-gold sm:px-4 ${
                  selected ? 'bg-deep text-ivory' : 'text-muted hover:text-deep'
                }`}
              >
                {tab.label}
              </button>
            )
          })}
        </div>
      </div>
      <div id="material-tabpanel" role="tabpanel" aria-labelledby={`material-tab-${inputMode}`} className="space-y-4">
        {inputMode === 'paste'
          ? <PasteNotesInput text={text} onTextChange={onTextChange} disabled={disabled} />
          : <DocumentUpload file={file} onFileSelect={onFileSelect} onFileRemove={onFileRemove} disabled={disabled} />}
        {message && (
          <p role="status" className="rounded-xl border border-error bg-paper p-3.5 text-sm font-semibold text-error">
            {message}
          </p>
        )}
        {children}
      </div>
    </section>
  )
}
