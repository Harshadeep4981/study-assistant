import { useEffect, useRef } from 'react'

export function useScrollProgress(ref) {
  const frame = useRef(0)
  useEffect(() => {
    const update = () => {
      cancelAnimationFrame(frame.current)
      frame.current = requestAnimationFrame(() => {
        const root = ref.current
        if (!root) return
        const range = Math.max(1, document.documentElement.scrollHeight - window.innerHeight)
        const progress = Math.min(1, Math.max(0, window.scrollY / range))
        root.style.setProperty('--scroll', String(progress))
        root.style.setProperty('--scroll-x', `${Math.round(progress * 120)}px`)
        root.style.setProperty('--scroll-y', `${Math.round(progress * -80)}px`)
        root.style.setProperty('--scroll-hue', `${Math.round(progress * 40)}deg`)
      })
    }
    update()
    window.addEventListener('scroll', update, { passive: true })
    window.addEventListener('resize', update, { passive: true })
    return () => { cancelAnimationFrame(frame.current); window.removeEventListener('scroll', update); window.removeEventListener('resize', update) }
  }, [ref])
}
