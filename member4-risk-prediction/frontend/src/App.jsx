
import { useEffect, useState } from 'react'
import './App.css'

const API_URL = 'http://127.0.0.1:8000'

function App() {
  const getTomorrow = () => {
    const d = new Date()
    d.setDate(d.getDate() + 1)
    return d.toISOString().slice(0, 10)
  }

  const [people, setPeople] = useState([])
  const [area, setArea] = useState('Central')
  const [date, setDate] = useState(getTomorrow)
  const [previousDay, setPreviousDay] = useState(0)
  const [previousWeek, setPreviousWeek] = useState(0)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [activeNav, setActiveNav] = useState('Predictions')

  useEffect(() => {
    fetch('/sample_risk_results.json')
      .then((res) => {
        if (!res.ok) throw new Error('Could not load sample results')
        return res.json()
      })
      .then(setPeople)
      .catch((err) => setError(err.message))
  }, [])

  async function predictIncidents(event) {
    event.preventDefault()
    setLoading(true)
    setError('')
    setResult(null)

    try {
      const response = await fetch(`${API_URL}/api/predict/historical`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          area,
          date,
          previous_day_count: Number(previousDay),
          previous_7_day_count: Number(previousWeek),
        }),
      })

      const data = await response.json()
      if (!response.ok) {
        throw new Error(data.detail || 'Prediction request failed')
      }
      setResult(data)
    } catch (err) {
      setError(
        'Historical API connection failed. Check FastAPI on port 8000. ' +
        err.message
      )
    } finally {
      setLoading(false)
    }
  }

  const high = people.filter((p) => p.risk_level === 'high').length
  const medium = people.filter((p) => p.risk_level === 'medium').length
  const low = people.filter((p) => p.risk_level === 'low').length

  const navigation = [
    ['▦', 'Dashboard'],
    ['◎', 'Object Detection'],
    ['⌁', 'Tracking'],
    ['⌁', 'Behavior Analysis'],
    ['⌕', 'Risk Assessment'],
    ['◈', 'Predictions'],
    ['♧', 'Alerts'],
    ['▤', 'Reports'],
    ['⚙', 'Settings'],
  ]

  return (
    <div className="dashboard-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">◇</div>
          <div>
            <strong>AI Shoplifting</strong>
            <span>Detection System</span>
          </div>
        </div>

        <p className="nav-label">WORKSPACE</p>

        <nav>
          {navigation.map(([icon, label]) => (
            <button
              type="button"
              key={label}
              className={`nav-item ${activeNav === label ? 'active' : ''}`}
              onClick={() => setActiveNav(label)}
            >
              <span className="nav-icon">{icon}</span>
              {label}
            </button>
          ))}
        </nav>

        <div className="sidebar-bottom">
          <span className="online-dot" />
          <div>
            <strong>Research prototype</strong>
            <small>Human-reviewed decisions</small>
          </div>
        </div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <span className="live-pill">
            <i className="online-dot" /> DEMO MODE
          </span>
          <div className="topbar-right">
            <span className="top-select">▣ Sample camera / video</span>
            <span className="top-date">Component 4</span>
          </div>
        </header>

        <div className="page-content">
          <div className="page-heading">
            <div>
              <p className="eyebrow">AI SHOPLIFTING RESEARCH · COMPONENT 4</p>
              <h1>Pattern Learning &amp; Future Risk Prediction</h1>
              <p className="muted">
                Historical incident estimation, risk scoring and person-density
                visualisation
              </p>
            </div>
            <span className="prototype-tag">● STANDALONE PROTOTYPE</span>
          </div>

          <section className="summary-grid">
            <article className="summary-card">
              <div className="summary-icon orange">◷</div>
              <div>
                <span className="card-label">SAMPLE RISK SCORING</span>
                <h2>{people.length} people</h2>
                <p><b>{high} High</b> · {medium} Medium · {low} Low</p>
              </div>
            </article>

            <article className="summary-card">
              <div className="summary-icon purple">⌖</div>
              <div>
                <span className="card-label">VIDEO ANALYSIS</span>
                <h2>60 frames</h2>
                <p>Sample clip analysis</p>
              </div>
            </article>

            <article className="summary-card">
              <div className="summary-icon green">▤</div>
              <div>
                <span className="card-label">HISTORICAL MODEL</span>
                <h2>{result ? result.predicted_incident_count : 'Ready'}</h2>
                <p>{result ? 'Estimated incident count' : 'Run a prediction below'}</p>
              </div>
            </article>
          </section>

          <section className="content-grid top-grid">
            <article className="panel">
              <div className="panel-heading">
                <div>
                  <h2>Person-Density Heatmap</h2>
                  <p>YOLO detections from the sample video</p>
                </div>
                <span className="small-tag">SAMPLE VIDEO</span>
              </div>

              <div className="heatmap-frame">
<video
    controls
    muted
    loop
    playsInline
    preload="metadata"
    style={{
      display: 'block',
      width: '100%',
      height: 'auto',
      borderRadius: '8px',
    }}
  >
   <source src="/detected_video_h264.mp4" type="video/mp4" />
    Your browser does not support video playback.
  </video>
</div>

              <div className="legend-row">
                <span><i className="legend-dot blue" /> Lower density</span>
                <span><i className="legend-dot red" /> Higher density</span>
              </div>

              <p className="fine-print">
                This image shows person-detection density in the analysed
                frames. It does not identify theft or show actual theft risk.
              </p>
            </article>

            <article className="panel">
              <div className="panel-heading">
                <div>
                  <h2>Risk Score Overview</h2>
                  <p>Sample risk scores</p>
                </div>
                <span className="small-tag neutral">DEMO DATA</span>
              </div>

              <div className="score-bars">
                {people.map((person) => (
                  <div className="score-bar-row" key={person.person_id}>
                    <div className="score-bar-label">
                      <span>Person {person.person_id}</span>
                      <strong>{person.risk_score}/100</strong>
                    </div>
                    <div className="bar-track">
                      <div
                        className={`bar-fill ${person.risk_level}`}
                        style={{
                          width: `${Math.max(0, Math.min(100, person.risk_score))}%`,
                        }}
                      />
                    </div>
                  </div>
                ))}
              </div>

              <div className="score-summary">
                <span>High <b>{high}</b></span>
                <span>Medium <b>{medium}</b></span>
                <span>Low <b>{low}</b></span>
              </div>

              <p className="fine-print">
                These scores come from the sample JSON file and are not
                calibrated theft probabilities.
              </p>
            </article>
          </section>

          <section className="content-grid lower-grid">
            <article className="panel">
              <div className="panel-heading">
                <div>
                  <h2>Sample Risk Assessment</h2>
                  <p>Standalone risk-scoring output</p>
                </div>
                <span className="small-tag neutral">{people.length} RECORDS</span>
              </div>

              <div className="table-head">
                <span>PERSON</span>
                <span>SCORE</span>
                <span>RISK LEVEL</span>
              </div>

              {people.map((person) => (
                <div className="person-row" key={person.person_id}>
                  <div className="person-cell">
                    <span className="avatar">P{person.person_id}</span>
                    <div>
                      <strong>Person {person.person_id}</strong>
                      <small>Sample input · demo only</small>
                    </div>
                  </div>
                  <strong className="score-number">{person.risk_score}/100</strong>
                  <span className={`risk-badge ${person.risk_level}`}>
                    {person.risk_level}
                  </span>
                </div>
              ))}

              <p className="fine-print">
                Sample values are for demonstration only, not real-person
                assessments.
              </p>
            </article>

            <article className="panel">
              <div className="panel-heading">
                <div>
                  <h2>Model &amp; Data Notes</h2>
                  <p>Current prototype scope</p>
                </div>
              </div>

              <div className="insight-item">
                <span className="insight-symbol orange">↗</span>
                <div>
                  <strong>Historical model</strong>
                  <p>Estimates area-level incident counts from historical data.</p>
                </div>
              </div>

              <div className="insight-item">
                <span className="insight-symbol blue">◎</span>
                <div>
                  <strong>Risk scoring</strong>
                  <p>Converts a supplied score from 0–100 into Low, Medium or High.</p>
                </div>
              </div>

              <div className="insight-item">
                <span className="insight-symbol green">✓</span>
                <div>
                  <strong>Human review</strong>
                  <p>Use outputs to support review, not automatic accusations.</p>
                </div>
              </div>
            </article>
          </section>

          <section className="panel historical-panel">
            <div className="panel-heading">
              <div>
                <h2>Historical Incident Prediction</h2>
                <p>Uses the existing Python model via FastAPI</p>
              </div>
              <span className="small-tag">API REQUEST</span>
            </div>

            <form className="prediction-form" onSubmit={predictIncidents}>
              <label>
                Historical data area
                <input
                  value={area}
                  onChange={(e) => setArea(e.target.value)}
                  required
                />
                <small>Area-level historical data, not a specific shop.</small>
              </label>

              <label>
                Prediction date
                <input
                  type="date"
                  value={date}
                  min={getTomorrow()}
                  onChange={(e) => setDate(e.target.value)}
                  required
                />
              </label>

              <label>
                Incidents on previous day
                <input
                  type="number"
                  min="0"
                  value={previousDay}
                  onChange={(e) => setPreviousDay(e.target.value)}
                  required
                />
              </label>

              <label>
                Incidents in previous 7 days
                <input
                  type="number"
                  min="0"
                  value={previousWeek}
                  onChange={(e) => setPreviousWeek(e.target.value)}
                  required
                />
              </label>

              <button className="primary-button" type="submit" disabled={loading}>
                {loading ? 'Analysing…' : 'Generate Historical Prediction →'}
              </button>
            </form>

            {error && <div className="error-message">{error}</div>}

            {result && (
              <div className="api-result">
                <div>
                  <span className="card-label">ESTIMATED INCIDENT COUNT</span>
                  <strong>{result.predicted_incident_count}</strong>
                </div>
                <div>
                  <p><b>Area:</b> {result.area}</p>
                  <p><b>Date:</b> {result.prediction_date}</p>
                  <p><b>Type:</b> {result.prediction_type}</p>
                </div>
              </div>
            )}

            <p className="fine-print">
              Output is an estimated area-level incident count, not a
              store-specific forecast or probability of theft.
            </p>
          </section>

          <footer className="footer-note">
            Component 4 · Risk Scoring &amp; Future Prediction
            <span>Research prototype · results require validation before operational use.</span>
          </footer>
        </div>
      </main>
    </div>
  )
}

export default App
