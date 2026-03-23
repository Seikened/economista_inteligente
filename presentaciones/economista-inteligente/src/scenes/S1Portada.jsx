import { useEffect, useRef } from 'react'
import gsap from 'gsap'
import { tl } from '../utils/motion'

export default function S1Portada({ isActive }) {
  const root = useRef()
  useEffect(() => {
    if (!isActive) return
    gsap.set(root.current, { opacity: 1 })
    const ctx = gsap.context(() => {
      tl()
        .from('.s1-eye',   { x: -20, opacity: 0, duration: 0.5 })
        .from('.s1-h1',    { yPercent: 55, opacity: 0, duration: 1.2, ease: 'expo.out' }, '-=0.3')
        .from('.s1-meta',  { opacity: 0, y: 9, duration: 0.55 }, '-=0.4')
        .from('.s1-ghost', { opacity: 0, x: 44, duration: 1.2, ease: 'expo.out' }, '<-=0.8')
    }, root)
    return () => ctx.revert()
  }, [isActive])

  return (
    <div ref={root} className="scene s1">
      <div className="eyebrow accent s1-eye">Material de Econometría · 2026</div>
      <h1 className="s1-h1">
        Economista<br />
        <em>Inteligente</em>
      </h1>
      <p className="s1-meta">Análisis de Sentimientos Financieros · 19 Tickers</p>
      <div className="s1-ghost">EI</div>
    </div>
  )
}
