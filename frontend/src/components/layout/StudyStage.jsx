import { useRef } from 'react'
import { useScrollProgress } from '../../hooks/useScrollProgress.js'

export default function StudyStage({ children }) {
  const stage = useRef(null)
  useScrollProgress(stage)
  return <div ref={stage} className="study-stage">{children}</div>
}
