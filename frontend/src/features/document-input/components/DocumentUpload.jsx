import { useRef, useState } from 'react'
import { FileText, Upload, X } from 'lucide-react'

function formatFileSize(bytes) {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

export function SelectedFile({ file, onRemove, onReplace, disabled = false }) {
  return (
    <div className="flex flex-col gap-3 rounded-2xl border border-moss bg-forest p-4 text-ivory sm:flex-row sm:items-center sm:justify-between">
      <div className="flex min-w-0 items-center gap-3">
        <span className="grid h-10 w-10 shrink-0 place-items-center rounded-lg border border-gold bg-deep text-gold">
          <FileText size={19} />
        </span>
        <div className="min-w-0">
          <p className="truncate text-sm font-bold text-sage" title={file.name}>{file.name}</p>
          <p className="mt-0.5 text-xs font-semibold text-gold">{formatFileSize(file.size)}</p>
        </div>
      </div>
      <div className="flex shrink-0 items-center gap-2">
        <button
          type="button"
          disabled={disabled}
          onClick={onReplace}
          className="min-h-9 rounded-lg border-2 border-gold bg-deep px-3 text-xs font-bold text-gold transition hover:bg-gold hover:text-deep focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-gold disabled:cursor-not-allowed"
        >
          Replace
        </button>
        <button
          type="button"
          disabled={disabled}
          onClick={onRemove}
          aria-label={`Remove ${file.name}`}
          className="grid h-9 w-9 place-items-center rounded-lg border-2 border-gold bg-deep text-gold transition hover:bg-gold hover:text-deep focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-gold disabled:cursor-not-allowed"
        >
          <X size={16} />
        </button>
      </div>
    </div>
  )
}

export default function DocumentUpload({ file, onFileSelect, onFileRemove, disabled = false }) {
  const fileInputRef = useRef(null)
  const [dragging, setDragging] = useState(false)

  function openPicker() { fileInputRef.current?.click() }
  function handleFile(nextFile) { if (!disabled && nextFile) onFileSelect(nextFile) }
  function handleInputChange(event) {
    handleFile(event.target.files?.[0])
    event.target.value = ''
  }
  function handleDrop(event) {
    event.preventDefault()
    if (disabled) return
    setDragging(false)
    handleFile(event.dataTransfer.files?.[0])
  }

  return (
    <div className="space-y-3">
      <input
        ref={fileInputRef}
        disabled={disabled}
        className="sr-only"
        type="file"
        accept=".txt,.pdf,text/plain,application/pdf"
        onChange={handleInputChange}
        aria-label="Choose a TXT or PDF document"
      />
      {file ? (
        <SelectedFile file={file} onRemove={onFileRemove} onReplace={openPicker} disabled={disabled} />
      ) : (
        <div
          onDragEnter={(event) => { event.preventDefault(); if (!disabled) setDragging(true) }}
          onDragOver={(event) => event.preventDefault()}
          onDragLeave={(event) => { if (!event.currentTarget.contains(event.relatedTarget)) setDragging(false) }}
          onDrop={handleDrop}
          className={`rounded-2xl border border-dashed px-5 py-8 text-center transition-colors sm:py-10 ${
            dragging ? 'border-gold-ink bg-sage text-deep' : 'border-moss bg-paper hover:border-forest'
          }`}
        >
          <span className="mx-auto grid h-12 w-12 place-items-center rounded-2xl bg-forest text-sage">
            <Upload size={20} />
          </span>
          <p className="mt-3 text-sm font-bold text-deep sm:text-base">Drop your document here</p>
          <p className="mt-1 text-sm font-medium text-muted">or choose a file from your device</p>
          <button
            type="button"
            disabled={disabled}
            onClick={openPicker}
            className="mt-4 inline-flex min-h-10 items-center justify-center rounded-full bg-deep px-5 text-sm font-bold text-ivory transition hover:bg-forest focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-gold-ink disabled:cursor-not-allowed"
          >
            Browse files
          </button>
          <p className="mt-3 text-xs font-semibold text-muted">TXT or PDF · PDFs up to 100 pages · one document at a time</p>
        </div>
      )}
    </div>
  )
}
