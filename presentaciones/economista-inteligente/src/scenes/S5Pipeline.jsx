import { useEffect, useRef } from 'react'
import gsap from 'gsap'
import { tl } from '../utils/motion'

const NODOS = [
  { icon: '🌐', lbl: 'Finnhub API',   desc: 'Noticias por ticker',         hi: false },
  { icon: '💾', lbl: 'noticias.json', desc: 'Cache + precios yfinance',    hi: false },
  { icon: '🧠', lbl: 'FinBERT',       desc: 'Clasificación NLP ponderada', hi: true  },
  { icon: '🗄️', lbl: 'Parquet',       desc: 'Cache analítico por ticker',  hi: false },
  { icon: '📊', lbl: 'Gráficas',      desc: 'PNG + tablas Rich',           hi: false },
]

export default function S5Pipeline({ isActive }) {
  const root = useRef()
  useEffect(() => {
    if (!isActive) return
    gsap.set(root.current, { opacity: 1 })
    const ctx = gsap.context(() => {
      tl()
        .from('.s5-eye',     { x: -20, opacity: 0, duration: 0.5 })
        .from('.s5-heading', { yPercent: 55, opacity: 0, duration: 1.2, ease: 'expo.out' }, '-=0.3')
        .from('.s5-node, .s5-arrow', { x: 28, opacity: 0, duration: 0.5, stagger: 0.09, ease: 'back.out(1.2)' }, '-=0.5')
    }, root)
    return () => ctx.revert()
  }, [isActive])

  return (
    <div ref={root} className="scene s5">
      <div className="s5-header">
        <div className="eyebrow accent s5-eye">04 / Flujo de datos</div>
        <h2 className="s5-heading">Data<br /><em>Pipeline</em></h2>
      </div>
      <div className="s5-flow">
        {NODOS.map((n, i) => (
          <>
            <div className={`s5-node ${n.hi ? 'hi' : ''}`} key={n.lbl}>
              <div className="s5-node-icon">{n.icon}</div>
              <div className="s5-node-lbl">{n.lbl}</div>
              <div className="s5-node-desc">{n.desc}</div>
            </div>
            {i < NODOS.length - 1 && <div className="s5-arrow" key={`a${i}`}>→</div>}
          </>
        ))}
      </div>
    </div>
  )
}
