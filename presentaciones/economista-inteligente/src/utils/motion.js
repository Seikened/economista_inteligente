import gsap from 'gsap'
export const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches
export function tl(config = {}) {
  const defaults = reducedMotion
    ? { duration: 0, delay: 0 }
    : { ease: 'power3.out', duration: 0.7 }
  return gsap.timeline({ defaults, ...config })
}
