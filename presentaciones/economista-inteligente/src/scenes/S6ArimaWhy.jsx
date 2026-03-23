import { useEffect, useRef } from 'react'
import gsap from 'gsap'
import { tl } from '../utils/motion'

const RAZONES = [
  { idx: '01', name: 'No estacionariedad', desc: 'Los precios tienen tendencia → d ≥ 1 obligatorio' },
  { idx: '02', name: 'Memoria del pasado',  desc: 'Hoy depende de ayer (y anteayer) → componente AR(p)' },
  { idx: '03', name: 'Shocks de mercado',   desc: 'Los errores se propagan en el tiempo → componente MA(q)' },
  { idx: '04', name: 'Selección objetiva',  desc: 'AIC penaliza complejidad → evita overfitting automáticamente' },
]

export default function S6ArimaWhy({ isActive }) {
  const root = useRef()
  useEffect(() => {
    if (!isActive) return
    gsap.set(root.current, { opacity: 1 })
    const ctx = gsap.context(() => {
      tl()
        .from('.saw-eye',   { x: -20, opacity: 0, duration: 0.5 })
        .from('.saw-title', { yPercent: 55, opacity: 0, duration: 1.2, ease: 'expo.out' }, '-=0.3')
        .from('.saw-desc',  { opacity: 0, y: 9, duration: 0.55 }, '-=0.5')
        .from('.saw-item',  { x: 40, opacity: 0, duration: 0.55, stagger: 0.12, ease: 'expo.out' }, '-=0.4')
    }, root)
    return () => ctx.revert()
  }, [isActive])

  return (
    <div ref={root} className="scene saw">
      <div className="saw-left">
        <div className="eyebrow accent saw-eye">05 / Modelo econométrico</div>
        <div className="saw-title">¿Por<br /><em>qué</em><br />ARIMA?</div>
        <p className="saw-desc">
          Los precios financieros violan los supuestos de modelos simples.
          ARIMA fue diseñado específicamente para series de tiempo con
          tendencia, autocorrelación y shocks estocásticos.
        </p>
      </div>
      <div className="saw-right">
        {RAZONES.map(r => (
          <div className="saw-item" key={r.idx}>
            <span className="saw-idx">{r.idx}</span>
            <div className="saw-item-body">
              <div className="saw-item-name">{r.name}</div>
              <div className="saw-item-desc">{r.desc}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
