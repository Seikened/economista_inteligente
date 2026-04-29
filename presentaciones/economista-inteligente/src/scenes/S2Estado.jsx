import { useEffect, useRef } from 'react'
import gsap from 'gsap'
import { tl } from '../utils/motion'

const MODULOS = [
  { badge: 'Orquestador · Activo', name: 'director.py',             desc: 'Pipeline end-to-end para todos los tickers',    active: true  },
  { badge: 'Fetcher · Activo',     name: 'get_noticias',            desc: 'Finnhub API + yfinance con paginación y cache', active: true  },
  { badge: 'Clasificador · Activo',name: 'cliente_clasificador',    desc: 'Polars lazy frames + merge de precios',         active: true  },
  { badge: 'NLP · En progreso',    name: 'modelo_matematico',       desc: 'Ratios financieros y modelos econométricos',    active: false },
]

export default function S2Estado({ isActive }) {
  const root = useRef()
  useEffect(() => {
    if (!isActive) return
    gsap.set(root.current, { opacity: 1 })
    const ctx = gsap.context(() => {
      tl()
        .from('.s2-eye',     { x: -20, opacity: 0, duration: 0.5 })
        .from('.s2-heading', { yPercent: 55, opacity: 0, duration: 1.2, ease: 'expo.out' }, '-=0.3')
        .from('.s2-sub',     { opacity: 0, y: 9, duration: 0.55 }, '-=0.5')
        .from('.s2-card',    { x: 40, opacity: 0, duration: 0.55, stagger: 0.12, ease: 'expo.out' }, '-=0.4')
    }, root)
    return () => ctx.revert()
  }, [isActive])

  return (
    <div ref={root} className="scene s2">
      <div className="s2-left">
        <div className="eyebrow accent s2-eye">01 / Stack actual</div>
        <h2 className="s2-heading">
          Módulos<br /><em>activos</em>
        </h2>
        <p className="s2-sub">Python 3.12 · UV · Polars</p>
      </div>
      <div className="s2-right">
        {MODULOS.map(m => (
          <div className={`s2-card ${m.active ? 'active' : ''}`} key={m.name}>
            <span className={`s2-badge ${m.active ? 'active' : ''}`}>{m.badge}</span>
            <div className="s2-card-name">{m.name}</div>
            <div className="s2-card-desc">{m.desc}</div>
          </div>
        ))}
      </div>
    </div>
  )
}
