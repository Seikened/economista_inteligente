import { useEffect, useRef } from 'react'
import gsap from 'gsap'
import { tl } from '../utils/motion'

const OUTPUTS = [
  { campo: 'orden',      ejemplo: 'ARIMA(2,1,1)',                    desc: 'Mejor combinación (p,d,q) por AIC' },
  { campo: 'AIC',        ejemplo: '4 523.18',                        desc: 'Criterio de selección — menor es mejor' },
  { campo: 'precio hoy', ejemplo: '$ 150.25',                        desc: 'Último cierre del historial' },
  { campo: 'forecast',   ejemplo: '152 · 153 · 155 · 156 · 158',    desc: '5 precios diarios proyectados' },
  { campo: 'retorno',    ejemplo: '+ 5.43 %',                        desc: 'Variación esperada al día 5' },
  { campo: 'tendencia',  ejemplo: 'ALCISTA',                         desc: 'Si retorno esperado > 0' },
]

export default function S7ArimaOut({ isActive }) {
  const root = useRef()
  useEffect(() => {
    if (!isActive) return
    gsap.set(root.current, { opacity: 1 })
    const ctx = gsap.context(() => {
      tl()
        .from('.sao-eye',     { x: -20, opacity: 0, duration: 0.5 })
        .from('.sao-heading', { yPercent: 55, opacity: 0, duration: 1.2, ease: 'expo.out' }, '-=0.3')
        .from('.sao-row',     { x: 40, opacity: 0, duration: 0.5, stagger: 0.09, ease: 'back.out(1.2)' }, '-=0.5')
    }, root)
    return () => ctx.revert()
  }, [isActive])

  return (
    <div ref={root} className="scene sao">
      <div className="sao-header">
        <div className="eyebrow accent sao-eye">07 / Output por ticker</div>
        <h2 className="sao-heading">Resultados<br /><em>del modelo</em></h2>
      </div>
      <div className="sao-table">
        {OUTPUTS.map(o => (
          <div className="sao-row" key={o.campo}>
            <span className="sao-campo">{o.campo}</span>
            <span className="sao-ejemplo">{o.ejemplo}</span>
            <span className="sao-desc">{o.desc}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
