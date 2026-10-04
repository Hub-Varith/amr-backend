export function ExampleReport() {
  return <section className="specimen" aria-label="Illustrative MIC report">
    <div className="specimen-top"><span>Example report / 001</span><span>Illustrative data</span></div>
    <h2>Klebsiella pneumoniae</h2><p className="specimen-sub">Genome → resistance features → MIC</p>
    <div className="sequence" aria-hidden="true">ATCG ATGC GCTA CGAT ATGC</div>
    <div className="report-row"><small>ANTIBIOTIC</small><small>EXAMPLE MIC · mg/L</small></div>
    <div className="report-row"><span>Meropenem</span><span>8</span></div>
    <div className="report-row"><span>Amikacin</span><span>4</span></div>
    <div className="report-row"><span>Ciprofloxacin</span><span>2</span></div>
    <p className="report-note">Fictional values to demonstrate the interface. No analysis was run. MIC estimates require laboratory confirmation.</p>
  </section>
}
