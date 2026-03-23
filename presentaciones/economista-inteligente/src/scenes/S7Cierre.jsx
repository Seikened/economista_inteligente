import { useEffect, useRef } from 'react'
import gsap from 'gsap'
import { tl } from '../utils/motion'

export default function S7Cierre({ isActive }) {
  const root = useRef()
  useEffect(() => {
    if (!isActive) return
    gsap.set(root.current, { opacity: 1 })
    const ctx = gsap.context(() => {
      tl()
        .from('.s7-eye',     { x: -20, opacity: 0, duration: 0.5 })
        .from('.s7-heading', { yPercent: 55, opacity: 0, duration: 1.3, ease: 'expo.out' }, '-=0.3')
        .from('.s7-sub',     { opacity: 0, y: 9, duration: 0.6 }, '-=0.4')
        .from('.s7-pill',    { opacity: 0, y: 8, duration: 0.4, stagger: 0.1 }, '-=0.3')
        .from('.s7-ghost',   { opacity: 0, x: 44, duration: 1.2, ease: 'expo.out' }, '<-=1.0')
    }, root)
    return () => ctx.revert()
  }, [isActive])

  return (
    <div ref={root} className="scene s7">
      <div className="eyebrow accent s7-eye">Conclusión</div>
      <h2 className="s7-heading">
        Sentimiento<br /><em>+ tiempo.</em>
      </h2>
      <p className="s7-sub">
        FinBERT captura el tono del mercado en noticias.
        ARIMA modela la memoria del precio y proyecta 5 días.
        Juntos forman una señal completa sobre las 19 empresas
        tech más grandes. Reproducible y sin intervención manual.
      </p>
      <div className="s7-pills">
        <span className="s7-pill">NLP + Econometría</span>
        <span className="s7-pill">19 Tickers</span>
        <span className="s7-pill">AIC Auto-select</span>
        <span className="s7-pill">End-to-End</span>
      </div>
      <div className="s7-ghost">∑</div>
    </div>
  )
}
