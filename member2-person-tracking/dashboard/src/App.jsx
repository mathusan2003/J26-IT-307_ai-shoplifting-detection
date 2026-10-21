import { useEffect, useMemo, useState } from "react";
import Alerts from "./Alerts.jsx";
import BehaviourAnalysis from "./BehaviourAnalysis.jsx";
import ObjectDetection from "./ObjectDetection.jsx";

const NAV = [
  { label: "Dashboard", page: null },
  { label: "Object Detection", page: "objects" },
  { label: "Tracking", page: "tracking" },
  { label: "Behavior Analysis", page: "behaviour" },
  { label: "Risk Assessment", page: null },
  { label: "Predictions", page: null },
  { label: "Alerts", page: "alerts" },
  { label: "Reports", page: null },
  { label: "Settings", page: null },
];

function displayZone(zone) {
  if (!zone || String(zone).toLowerCase().includes("shelf")) return "—";
  return zone;
}

function statusClass(status) {
  if (status === "active") return "ok";
  if (status === "temporarily_lost") return "warn";
  return "muted";
}

const LIVE_WEBCAM = {
  id: "webcam",
  file: "webcam",
  label: "Live webcam",
  group: "live",
  group_label: "Live camera",
  description: "Laptop camera through the shared Member 2 live session",
  camera: "Live webcam",
};

function mergeSources(sources, currentSource) {
  const rows = Array.isArray(sources) ? [...sources] : [];
  if (!rows.some((item) => item.id === "webcam")) rows.unshift(LIVE_WEBCAM);
  if (currentSource && !rows.some((item) => item.id === currentSource || item.file === currentSource)) {
    rows.push({
      id: currentSource,
      file: currentSource,
      label: currentSource,
      group_label: "Live camera",
      description: currentSource,
    });
  }
  return rows;
}

export default function App() {
  const [state, setState] = useState(null);
  const [sources, setSources] = useState([]);
  const [streamKey, setStreamKey] = useState(0);
  const [page, setPage] = useState("tracking");

  async function postControl(body) {
    const response = await fetch("/api/control", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const payload = await response.json();
    if (!response.ok) {
      console.error("control failed", payload);
      return;
    }
    setState(payload);
  }

  useEffect(() => {
    function loadSources() {
      fetch("/api/sources")
        .then((res) => res.json())
        .then((payload) => setSources(payload.videos || []))
        .catch(() => setSources([]));
    }
    loadSources();
    const poll = setInterval(() => {
      fetch("/api/state")
        .then((res) => res.json())
        .then(setState)
        .catch(() => {});
    }, 700);
    const sourcePoll = setInterval(loadSources, 4000);
    return () => {
      clearInterval(poll);
      clearInterval(sourcePoll);
    };
  }, []);

  const videoTime = state?.timestamp || "00:00:00.000";
  const tracks = (state?.tracks || []).filter((track) => track.status !== "terminated");
  const active = tracks.filter((track) => track.status === "active");
  const lostNow = tracks.filter((track) => track.status === "temporarily_lost");
  const selected = state?.selected && state.selected.status !== "terminated" ? state.selected : active[0];
  const summary = state?.summary || { people_now: 0, active: 0, lost: 0 };
  const peopleNow = Math.max(summary.people_now ?? 0, summary.detections_now ?? 0, active.length);
  const zones = state?.zones || { Entrance: 0, "Inside Store": 0, Checkout: 0 };
  const momentEvents = (state?.events || []).filter((event) => event.time === videoTime);
  const sourceList = mergeSources(sources, state?.source);
  const grouped = [];
  for (const item of sourceList) {
    const key = item.group_label || "Other clips";
    const last = grouped[grouped.length - 1];
    if (!last || last.label !== key) grouped.push({ label: key, items: [item] });
    else last.items.push(item);
  }
  const metrics = state?.eval || { mota: 0, motp: 0, idf1: 0, idsw: 0, reference: "" };
  const value = state?.value || {
    people_now: peopleNow,
    persistent_ids: active.map((track) => track.id),
    trajectory_points: 0,
    occlusion_held: lostNow.length,
    zone_crossings_now: momentEvents.filter((event) => event.text?.includes("moved")).length,
    shelf_visits_now: active.filter((track) => track.shelf_visit).length,
    entries_now: momentEvents.filter((event) => event.text?.includes("entered")).length,
    exits_now: momentEvents.filter((event) => event.text?.includes("checkout") || event.text?.includes("exit")).length,
    fps: state?.fps || 0,
  };

  const trailPath = useMemo(() => {
    const points = selected?.trail || [];
    if (points.length < 2) return "";
    const xs = points.map((p) => p[0]);
    const ys = points.map((p) => p[1]);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const sx = (x) => 16 + ((x - minX) / Math.max(1, maxX - minX)) * 220;
    const sy = (y) => 110 - ((y - minY) / Math.max(1, maxY - minY)) * 90;
    return points.map((p, i) => `${i ? "L" : "M"} ${sx(p[0])} ${sy(p[1])}`).join(" ");
  }, [selected]);

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="logo">S</div>
          <div>
            <strong>AI Shoplifting</strong>
            <span>Detection System</span>
          </div>
        </div>
        <nav>
          {NAV.map((item) => {
            const active = item.page && item.page === page;
            return (
              <button
                key={item.label}
                className={`nav ${active ? "active" : "idle"}`}
                type="button"
                onClick={() => {
                  if (item.page) setPage(item.page);
                }}
              >
                {item.label}
              </button>
            );
          })}
        </nav>
        <div className="side-foot">
          <span className="dot" /> System Status
          <small>All systems operational</small>
          <button className="logout" type="button">Logout</button>
        </div>
      </aside>

      <main>
        <header className="topbar">
          <div className="live-pill">
            <span className="pulse" /> LIVE
            <strong>{state?.camera || "Camera 01 - Aisle 3"}</strong>
          </div>
          <div className="clock">
            Camera time <strong>{videoTime}</strong>
          </div>
        </header>

        {page === "alerts" ? (
          <Alerts state={state} />
        ) : page === "behaviour" ? (
          <BehaviourAnalysis
            state={state}
            sources={sourceList}
            postControl={postControl}
            streamKey={streamKey}
            setStreamKey={setStreamKey}
            selectedId={selected?.id}
            onSelectPerson={(personId) => postControl({ action: "select", person_id: personId })}
          />
        ) : page === "objects" ? (
          <ObjectDetection
            state={state}
            sources={sourceList}
            postControl={postControl}
            streamKey={streamKey}
            setStreamKey={setStreamKey}
          />
        ) : (
        <section className="page">
          <h1>AI SHOPLIFTING DETECTION SYSTEM</h1>
          <div className="grid">
            <div className="card video-card">
              <div className="card-head">
                <span className="rec">REC</span>
                Live CCTV Feed - Multi-Person Tracking
                <select
                  className="view-mode scenario-select"
                  value={state?.source || ""}
                  onChange={async (event) => {
                    const source = event.target.value;
                    if (!source) return;
                    await postControl({ action: "open", source });
                    await postControl({ action: "resume" });
                    setStreamKey((value) => value + 1);
                  }}
                >
                  <option value="">Select live webcam or a clip</option>
                  {grouped.map((group) => (
                    <optgroup key={group.label} label={group.label}>
                      {group.items.map((item) => (
                        <option key={item.id} value={item.id} title={item.description}>
                          {item.label}
                        </option>
                      ))}
                    </optgroup>
                  ))}
                </select>
              </div>
              <div className="video-wrap">
                <img alt="Live tracking" src={`/stream.mjpg?k=${streamKey}`} />
                <div className="badge">{state?.algorithm || "BYTETRACK_ACTIVE"}</div>
                <div className="fps-chip">{(state?.fps || 0).toFixed(1)} FPS</div>
                {(state?.alerts || []).length > 0 && (
                  <div className="hud-alerts">{(state.alerts || []).join("  ")}</div>
                )}
                {state?.scenario?.description && (
                  <div className="scenario-chip">{state.scenario.description}</div>
                )}
                {state?.status === "loading" && <div className="overlay">Loading YOLOv8 detector… first start takes a minute on CPU</div>}
                {state?.source_error && state?.status !== "loading" && (
                  <div className="overlay">{state.source_error}</div>
                )}
                {!state?.source && !state?.source_error && state?.status !== "loading" && (
                  <div className="overlay">Select Live webcam or a video clip to start the shared session.</div>
                )}
              </div>
            </div>

            <div className="stack">
              <div className="card eval-card">
                <h3>LIVE TRACKING EVALUATION</h3>
                <p className="value-lead">{metrics.reference || "Session metrics for this camera clip"}</p>
                <div className="eval-grid">
                  <div><small>MOTA</small><b>{Number(metrics.mota || 0).toFixed(3)}</b></div>
                  <div><small>MOTP</small><b>{Number(metrics.motp || 0).toFixed(3)}</b></div>
                  <div><small>IDF1</small><b>{Number(metrics.idf1 || 0).toFixed(3)}</b></div>
                  <div><small>IDSW</small><b className="warn">{metrics.idsw || 0}</b></div>
                  <div><small>FPS</small><b className="ok">{(state?.fps || 0).toFixed(1)}</b></div>
                  <div><small>Active tracks</small><b className="ok">{peopleNow}</b></div>
                </div>
              </div>
              <div className="card">
                <h3>THIS CAMERA MOMENT</h3>
                <div className="now-count">
                  <small>People on camera now · {videoTime}</small>
                  <b>{peopleNow}</b>
                </div>
                {summary.detections_now != null && (
                  <p className="value-lead">Detected boxes {summary.detections_now} · tracked IDs {active.length}</p>
                )}
                <ul className="summary">
                  <li><span>Visible now</span><b className="ok">{peopleNow}</b></li>
                  <li><span>Occluded / lost now</span><b className="warn">{lostNow.length}</b></li>
                  <li><span>Camera time</span><b>{videoTime}</b></li>
                </ul>
              </div>
              <div className="card">
                <div className="card-head">
                  <h3>PEOPLE IN THIS FRAME</h3>
                  <small>{videoTime}</small>
                </div>
                <table>
                  <thead>
                    <tr>
                      <th>ID</th><th>STATUS</th><th>VIDEO TIME</th><th>CURRENT ZONE</th>
                    </tr>
                  </thead>
                  <tbody>
                    {tracks.slice(0, 12).map((track) => (
                      <tr key={track.id} onClick={() => postControl({ action: "select", person_id: track.id })}>
                        <td style={{ color: track.color }}>{track.id}</td>
                        <td className={statusClass(track.status)}>{track.status === "active" ? "Active" : "Lost"}</td>
                        <td>{videoTime}</td>
                        <td>{displayZone(track.zone)}</td>
                      </tr>
                    ))}
                    {!tracks.length && <tr><td colSpan="4">Nobody in this camera moment.</td></tr>}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <div className="card value-card">
            <h3>COMPONENT 2 VALUE — THIS CAMERA MOMENT</h3>
            <p className="value-lead">Live CCTV + YOLOv8 + ByteTrack + Kalman. Data only — no shoplifting label.</p>
            <div className="value-grid">
              <article>
                <small>Real-time multi-person monitoring</small>
                <b>{value.people_now} people</b>
                <span>{(value.fps || 0).toFixed(1)} FPS automated CCTV</span>
              </article>
              <article>
                <small>Consistent Person IDs + trajectories</small>
                <b>IDs {value.persistent_ids?.length ? value.persistent_ids.join(", ") : "—"}</b>
                <span>{value.trajectory_points || 0} path points this moment</span>
              </article>
              <article>
                <small>Occlusion / overlap / lost recovery</small>
                <b>{value.occlusion_held} held</b>
                <span>Kalman + ByteTrack keep ID through gaps</span>
              </article>
              <article>
                <small>Store data: zones, shelf, entry/exit, occupancy</small>
                <b>{value.shelf_visits_now} shelf · {value.zone_crossings_now} zone</b>
                <span>In {value.entries_now} · Out {value.exits_now} · Occ {value.people_now}</span>
              </article>
              <article>
                <small>Reliable feed for behaviour analysis</small>
                <b>Ready</b>
                <span>IDs, paths, zones, occupancy → Component 3</span>
              </article>
              <article>
                <small>Automated CCTV, less manual watch</small>
                <b>{state?.paused ? "Paused" : "Live"}</b>
                <span>Security sees tracks instead of raw feed only</span>
              </article>
            </div>
          </div>

          <div className="metrics">
            <div className="card metric">
              <h3>ZONE STATUS NOW</h3>
              <div className="zone-row">
                <div><small>ENTRY ZONE</small><b>{zones.Entrance}</b></div>
                <div><small>INSIDE STORE</small><b className="ok">{zones["Inside Store"]}</b></div>
                <div><small>EXIT ZONE</small><b>{zones.Checkout}</b></div>
              </div>
            </div>
            <div className="card metric">
              <h3>THIS MOMENT</h3>
              <div className="zone-row">
                <div><small>PEOPLE NOW</small><b className="ok">{peopleNow}</b></div>
                <div><small>LOST NOW</small><b className="warn">{lostNow.length}</b></div>
                <div><small>CAMERA TIME</small><b>{videoTime}</b></div>
                <div><small>FPS</small><b className="ok">{(state?.fps || 0).toFixed(1)}</b></div>
              </div>
              <p className="latency">Live frame {state?.frame_id || 0} · latency {state?.latency_ms || 0} ms</p>
            </div>
          </div>

          <div className="lower">
            <div className="card">
              <h3>TRACK CONTROLS</h3>
              <button className="primary" type="button" onClick={() => postControl({ action: (state == null || state.paused) ? "resume" : "pause" })}>
                {(state == null || state.paused) ? "Resume Tracking" : "Pause Tracking"}
              </button>
              <button type="button" onClick={() => postControl({ action: "reset" })}>Reset View</button>
              <p className="hint">YOLOv8n person detector + ByteTrack + Kalman. No shoplifting classification.</p>
            </div>
            <div className="card details">
              <h3>TRACK DETAILS {selected ? `(SELECTED ID: ${selected.id})` : ""}</h3>
              {selected ? (
                <div className="details-grid">
                  {selected.crop_jpeg ? <img className="crop" alt="" src={selected.crop_jpeg} /> : <div className="crop empty" />}
                  <dl>
                    <div><dt>track id</dt><dd>{selected.id}</dd></div>
                    <div><dt>status</dt><dd className={statusClass(selected.status)}>{selected.status}</dd></div>
                    <div><dt>camera time</dt><dd>{videoTime}</dd></div>
                    <div><dt>current zone</dt><dd>{displayZone(selected.zone)}</dd></div>
                    <div><dt>det conf</dt><dd>{Math.round((selected.confidence || 0) * 1000) / 10}%</dd></div>
                  </dl>
                </div>
              ) : <p>Select an active ID.</p>}
            </div>
            <div className="card">
              <h3>PATH IN THIS MOMENT</h3>
              <svg viewBox="0 0 250 130" className="trail">
                <rect x="8" y="8" width="234" height="114" rx="10" fill="#f8fafc" stroke="#e2e8f0" />
                {trailPath && <path d={trailPath} fill="none" stroke="#2563eb" strokeWidth="3" />}
              </svg>
            </div>
            <div className="card">
              <h3>EVENTS AT {videoTime}</h3>
              <ul className="events">
                {momentEvents.slice(0, 8).map((event, index) => (
                  <li key={`${event.time}-${index}`}>
                    <time>{event.time}</time>
                    <span>{event.text}</span>
                  </li>
                ))}
                {!momentEvents.length && <li>No new zone event in this frame.</li>}
              </ul>
            </div>
          </div>
          <footer>ByteTrack is used for robust multi-person tracking and consistent ID assignment across frames.</footer>
        </section>
        )}
      </main>
    </div>
  );
}
