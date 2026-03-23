import { useState, useEffect, useRef, useCallback } from 'react'
import gsap from 'gsap'
import S1Portada   from './scenes/S1Portada'
import S2Empresas  from './scenes/S2Empresas'
import S2Estado    from './scenes/S2Estado'
import S3Areas     from './scenes/S3Areas'
import S4Tecnico   from './scenes/S4Tecnico'
import S5Pipeline  from './scenes/S5Pipeline'
import S6ArimaWhy  from './scenes/S6ArimaWhy'
import S6Cerebro   from './scenes/S6Cerebro'
import S7ArimaOut  from './scenes/S7ArimaOut'
import S7Cierre    from './scenes/S7Cierre'

const SCENES = [
  S1Portada,
  S2Empresas,
  S2Estado,
  S3Areas,
  S4Tecnico,
  S5Pipeline,
  S6ArimaWhy,
  S6Cerebro,
  S7ArimaOut,
  S7Cierre,
]

export default function App() {
  const [current, setCurrent] = useState(0)
  const [dimmed, setDimmed]   = useState(false)
  const [hintOn, setHintOn]   = useState(true)
  const [light, setLight]     = useState(true)
  const [exporting, setExporting] = useState(false)
  const transitioning         = useRef(false)

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', light ? 'light' : 'dark')
  }, [light])

  const goTo = useCallback((idx) => {
    if (transitioning.current) return
    transitioning.current = true
    setDimmed(true)
    setTimeout(() => {
      setCurrent(idx)
      setDimmed(false)
      transitioning.current = false
    }, 200)
  }, [])

  const goNext = useCallback(() => goTo((current + 1) % SCENES.length), [current, goTo])
  const goPrev = useCallback(() => goTo((current - 1 + SCENES.length) % SCENES.length), [current, goTo])

  useEffect(() => {
    const onKey = (e) => {
      if (['Space','ArrowRight','Enter'].includes(e.code)) { e.preventDefault(); goNext() }
      if (e.code === 'ArrowLeft') { e.preventDefault(); goPrev() }
      if (e.code === 'KeyL') setLight(l => !l)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [goNext, goPrev])

  useEffect(() => {
    const t = setTimeout(() => setHintOn(false), 5000)
    return () => clearTimeout(t)
  }, [])

  const exportPDF = useCallback(async () => {
    if (exporting) return
    setExporting(true)
    try {
      const [{ jsPDF }, { default: html2canvas }] = await Promise.all([
        import('jspdf'),
        import('html2canvas'),
      ])
      const stage = document.getElementById('stage')
      const W = stage.offsetWidth
      const H = stage.offsetHeight
      const pdf = new jsPDF({ orientation: 'landscape', unit: 'px', format: [W, H], compress: true })
      const saved = current

      for (let i = 0; i < SCENES.length; i++) {
        transitioning.current = false
        setCurrent(i)
        await new Promise(r => setTimeout(r, 80))
        gsap.globalTimeline.progress(1, true)
        await new Promise(r => requestAnimationFrame(r))
        const canvas = await html2canvas(stage, {
          scale: 2,
          useCORS: true,
          logging: false,
          imageTimeout: 0,
          ignoreElements: el => ['hud', 'seg-bar', 'overlay'].includes(el.id),
        })
        if (i > 0) pdf.addPage()
        pdf.addImage(canvas.toDataURL('image/png'), 'PNG', 0, 0, W, H)
      }

      pdf.save('presentacion.pdf')
      transitioning.current = false
      setCurrent(saved)
    } finally {
      setExporting(false)
    }
  }, [exporting, current])

  const Scene = SCENES[current]

  return (
    <div id="stage" onClick={goNext}>
      <div id="overlay" className={dimmed ? 'dim' : ''} />
      <Scene isActive={!dimmed} key={current} />
      <div id="hud">
        <span id="hint" style={{ opacity: hintOn ? 1 : 0 }}>click · espacio · ←→ · L=tema</span>
        <button
          id="export-btn"
          onClick={e => { e.stopPropagation(); exportPDF() }}
          disabled={exporting}
          title="Exportar PDF"
        >
          {exporting ? '…' : '↓'}
        </button>
        <button
          id="theme-toggle"
          onClick={e => { e.stopPropagation(); setLight(l => !l) }}
          title="Cambiar tema (L)"
        >
          {light ? '◑' : '○'}
        </button>
        <span id="counter">{String(current + 1).padStart(2,'0')} — {String(SCENES.length).padStart(2,'0')}</span>
      </div>
      <div id="seg-bar">
        {SCENES.map((_, i) => (
          <div key={i}
            className={`seg ${i < current ? 'done' : ''} ${i === current ? 'current' : ''}`}
            onClick={e => { e.stopPropagation(); goTo(i) }}
          />
        ))}
      </div>
    </div>
  )
}
