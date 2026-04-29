import { useEffect, useRef } from 'react'
import gsap from 'gsap'
import { tl } from '../utils/motion'

const COMPONENTES = [
  { name: 'FinBERT',    question: '¿Qué tan positiva es la noticia?' },
  { name: 'Tokenizer',  question: '¿Cómo fragmentar el texto financiero?' },
  { name: 'Ponderación',question: '¿Headline 60% · Resumen 40%?' },
]

export default function S4Tecnico({ isActive }) {
  const root = useRef()
  useEffect(() => {
    if (!isActive) return
    gsap.set(root.current, { opacity: 1 })
    const ctx = gsap.context(() => {
      tl()
        .from('.s4-eye',     { x: -20, opacity: 0, duration: 0.5 })
        .from('.s4-title',   { yPercent: 55, opacity: 0, duration: 1.2, ease: 'expo.out' }, '-=0.3')
        .from('.s4-formula', { opacity: 0, y: 9, duration: 0.55 }, '-=0.3')
        .from('.s4-row',     { x: 40, opacity: 0, duration: 0.55, stagger: 0.12, ease: 'expo.out' }, '-=0.4')
    }, root)
    return () => ctx.revert()
  }, [isActive])

  return (
    <div ref={root} className="scene s4">
      <div className="s4-left">
        <div className="eyebrow accent s4-eye">03 / Clasificador NLP</div>
        <div className="s4-title">NLP</div>
        <div className="s4-sub">Análisis de sentimientos financieros</div>
        <div className="s4-formula">
          <strong>Headline</strong> × 0.6<br />
          + <strong>Resumen</strong> × 0.4<br />
          = Score ponderado
        </div>
      </div>
      <div className="s4-right">
        {COMPONENTES.map(c => (
          <div className="s4-row" key={c.name}>
            <span className="s4-metric">{c.name}</span>
            <span className="s4-question">{c.question}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
