import { useEffect, useRef } from 'react'
import gsap from 'gsap'
import { tl } from '../utils/motion'

const GRUPOS = [
  {
    region: 'US · Mega Cap',
    color: 'accent',
    tickers: ['NVDA','MSFT','AAPL','GOOGL','AMZN','META','TSLA','AVGO','ORCL','NFLX','PLTR','CSCO','IBM','AMD'],
  },
  {
    region: 'Asia-Pac',
    color: 'blue',
    tickers: ['TSM','TCEHY','BABA'],
  },
  {
    region: 'Europa',
    color: 'blue',
    tickers: ['ASML','SAP'],
  },
]

export default function S2Empresas({ isActive }) {
  const root = useRef()
  useEffect(() => {
    if (!isActive) return
    gsap.set(root.current, { opacity: 1 })
    const ctx = gsap.context(() => {
      tl()
        .from('.s2e-eye',    { x: -20, opacity: 0, duration: 0.5 })
        .from('.s2e-num',    { yPercent: 55, opacity: 0, duration: 1.2, ease: 'expo.out' }, '-=0.3')
        .from('.s2e-sub',    { opacity: 0, y: 9, duration: 0.5 }, '-=0.5')
        .from('.s2e-group',  { y: 28, opacity: 0, duration: 0.5, stagger: 0.12, ease: 'back.out(1.3)' }, '-=0.4')
    }, root)
    return () => ctx.revert()
  }, [isActive])

  return (
    <div ref={root} className="scene s2e">
      <div className="s2e-left">
        <div className="eyebrow accent s2e-eye">01 / Universo de análisis</div>
        <div className="s2e-num">19</div>
        <div className="s2e-sub">
          empresas · 7 años<br />de historial · ~1M noticias
        </div>
      </div>
      <div className="s2e-right">
        {GRUPOS.map(g => (
          <div className="s2e-group" key={g.region}>
            <div className={`s2e-region ${g.color}`}>{g.region}</div>
            <div className="s2e-chips">
              {g.tickers.map(t => (
                <span className={`s2e-chip ${g.color}`} key={t}>{t}</span>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
