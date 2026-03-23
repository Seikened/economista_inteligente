import { useEffect, useRef } from 'react'
import gsap from 'gsap'
import { tl } from '../utils/motion'

const DETALLES = [
  { name: 'Grid de órdenes', question: 'p∈[0–3]  d∈[1–2]  q∈[0–2]  →  16 modelos' },
  { name: 'Criterio AIC',    question: '¿Mejor balance ajuste / complejidad?' },
  { name: 'Forecast',        question: 'Predicción de 5 precios hacia adelante' },
  { name: 'Tendencia',       question: 'ALCISTA si rendimiento esperado > 0' },
]

export default function S6Cerebro({ isActive }) {
  const root = useRef()
  useEffect(() => {
    if (!isActive) return
    gsap.set(root.current, { opacity: 1 })
    const ctx = gsap.context(() => {
      tl()
        .from('.s6-eye',     { x: -20, opacity: 0, duration: 0.5 })
        .from('.s6-title',   { yPercent: 55, opacity: 0, duration: 1.2, ease: 'expo.out' }, '-=0.3')
        .from('.s6-formula', { opacity: 0, y: 9, duration: 0.55 }, '-=0.3')
        .from('.s6-row',     { x: 40, opacity: 0, duration: 0.55, stagger: 0.12, ease: 'expo.out' }, '-=0.4')
    }, root)
    return () => ctx.revert()
  }, [isActive])

  return (
    <div ref={root} className="scene s6-arima">
      <div className="s6a-left">
        <div className="eyebrow accent s6-eye">05 / Modelo econométrico</div>
        <div className="s6-title">ARIMA</div>
        <div className="s6a-sub">Selección automática de orden por AIC</div>
        <div className="s6-formula">
          <strong>AR(p)</strong> — memoria del pasado<br />
          <strong>I(d)</strong>  — diferenciación (d≥1)<br />
          <strong>MA(q)</strong> — corrección de errores
        </div>
      </div>
      <div className="s6a-right">
        {DETALLES.map(d => (
          <div className="s6-row" key={d.name}>
            <span className="s6a-metric">{d.name}</span>
            <span className="s6a-question">{d.question}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
