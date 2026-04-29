import { useEffect, useRef } from 'react'
import gsap from 'gsap'
import { tl } from '../utils/motion'

const AREAS = [
  { num: '01', icon: '📰', name: 'Recopilación\nde Noticias',  desc: 'Finnhub API · yfinance · JSON cache' },
  { num: '02', icon: '🧠', name: 'Análisis de\nSentimientos',  desc: 'FinBERT · Weighted avg · Pos/Neg/Neutral' },
  { num: '03', icon: '📈', name: 'Datos de\nMercado',          desc: '19 Tickers · Retornos · Merge por fecha' },
  { num: '04', icon: '📊', name: 'Visualización\ny BI',         desc: 'Parquet · Matplotlib · Ratios Rich' },
]

export default function S3Areas({ isActive }) {
  const root = useRef()
  useEffect(() => {
    if (!isActive) return
    gsap.set(root.current, { opacity: 1 })
    const ctx = gsap.context(() => {
      tl()
        .from('.s3-eye',     { x: -20, opacity: 0, duration: 0.5 })
        .from('.s3-heading', { yPercent: 55, opacity: 0, duration: 1.2, ease: 'expo.out' }, '-=0.3')
        .from('.s3-cell',    { y: 30, opacity: 0, duration: 0.55, stagger: 0.1, ease: 'back.out(1.3)' }, '-=0.5')
    }, root)
    return () => ctx.revert()
  }, [isActive])

  return (
    <div ref={root} className="scene s3">
      <div className="s3-header">
        <div className="eyebrow accent s3-eye">02 / Áreas del proyecto</div>
        <h2 className="s3-heading">4 áreas<br /><em>clave</em></h2>
      </div>
      <div className="s3-grid">
        {AREAS.map(a => (
          <div className="s3-cell" key={a.num}>
            <span className="s3-num">{a.num}</span>
            <span className="s3-icon">{a.icon}</span>
            <div className="s3-name">
              {a.name.split('\n').map((l, i) => <span key={i}>{l}<br /></span>)}
            </div>
            <div className="s3-desc">{a.desc}</div>
          </div>
        ))}
      </div>
    </div>
  )
}
