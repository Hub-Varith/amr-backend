import '@fontsource-variable/source-serif-4'
import '@fontsource/ibm-plex-sans/400.css'
import '@fontsource/ibm-plex-sans/400-italic.css'
import '@fontsource/ibm-plex-sans/500.css'
import '@fontsource/ibm-plex-mono/400.css'
import '@fontsource/ibm-plex-mono/500.css'
import './index.css'

import { MotionConfig } from 'motion/react'
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, Route, Routes } from 'react-router'

import { ScrollManager } from './components/ScrollManager'
import { AnalyzePlaceholderPage } from './pages/AnalyzePlaceholderPage'
import { IntroPage } from './pages/IntroPage'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    {/* "user" turns animations off for people who set reduced motion in their OS. */}
    <MotionConfig reducedMotion="user">
      <BrowserRouter>
        <ScrollManager />
        <Routes>
          <Route path="/" element={<IntroPage />} />
          <Route path="/analyze" element={<AnalyzePlaceholderPage />} />
          <Route path="*" element={<IntroPage />} />
        </Routes>
      </BrowserRouter>
    </MotionConfig>
  </StrictMode>,
)
