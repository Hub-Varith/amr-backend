import { useEffect } from 'react'
import { useLocation } from 'react-router'

/** Client-side routing does not scroll on its own: go to the #hash target if there is one, else the top. */
export function ScrollManager() {
  const { pathname, hash } = useLocation()

  useEffect(() => {
    // Wait a frame so the new page has rendered its sections.
    const frame = requestAnimationFrame(() => {
      const target = hash ? document.getElementById(hash.slice(1)) : null
      if (target) target.scrollIntoView()
      else window.scrollTo(0, 0)
    })
    return () => cancelAnimationFrame(frame)
  }, [pathname, hash])

  return null
}
