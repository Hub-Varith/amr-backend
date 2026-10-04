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
import { AnalysisPage } from './pages/AnalysisPage'
import { AnalyzePage } from './pages/AnalyzePage'
import { DemoAnalysisPage } from './pages/DemoAnalysisPage'
import { IntroPage } from './pages/IntroPage'
import { SampleReportPage } from './pages/SampleReportPage'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    {/* "user" turns animations off for people who set reduced motion in their OS. */}
    <MotionConfig reducedMotion="user">
      <BrowserRouter>
        <ScrollManager />
        <Routes>
          <Route path="/" element={<IntroPage />} />
          <Route path="/analyze" element={<AnalyzePage />} />
          <Route path="/analysis/:jobId" element={<AnalysisPage />} />
          <Route path="/demo/:key" element={<DemoAnalysisPage />} />
          <Route path="/report/sample/:key" element={<SampleReportPage />} />
          <Route path="*" element={<IntroPage />} />
        </Routes>
      </BrowserRouter>
    </MotionConfig>
  </StrictMode>,
)
