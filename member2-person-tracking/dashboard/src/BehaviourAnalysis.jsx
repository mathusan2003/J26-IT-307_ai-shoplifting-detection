import { AlertFeed, INDICATOR_LABELS, suspectedAlertsFromState } from "./Alerts.jsx";

const SOURCE_HELP =
  "Select Live webcam for the shared session, or add .mp4 files to member2-person-tracking/outputs or member2-person-tracking/data.";

function displayZone(zone) {
  if (!zone) return "—";
  return zone;
}

function statusClass(status) {
  if (status === "detected" || status === "running" || status === "ready" || status === "scored") return "ok";
  if (status === "waiting" || status === "pending" || status === "paused" || status === "medium") return "warn";
  if (status === "unavailable" || status === "error" || status === "high") return "danger";
  return "muted";
}

function levelClass(level) {
  if (level === "high") return "danger";
  if (level === "medium" || level === "low") return "warn";
  return "ok";
}

function formatDuration(seconds) {
  if (seconds == null || Number.isNaN(Number(seconds))) return "—";
  return `${Number(seconds).toFixed(1)} s`;
}

function formatMs(value) {
  if (value == null || value === "") return "—";
  return `${Number(value).toFixed(1)} ms`;
}

function reasonLabel(item) {
  if (!item) return "";
  if (typeof item === "string") return item;
  const type = item.type || "indicator";
  const reason = item.reason ? ` — ${item.reason}` : "";
  return `${type}${reason}`;
}

function scoreDisplay(person, field) {
  if (!person?.scored) {
    return { text: "—", hint: "unavailable", kind: "muted" };
  }
  const value = person[field];
  if (person.score_status === "pending" && !(person.behaviours || []).length) {
    return { text: String(value ?? 0), hint: "waiting", kind: "warn" };
  }
  return { text: String(value ?? 0), hint: null, kind: "" };
}

function indicatorCounts(people) {
  const counts = {};
  for (const person of people) {
    for (const item of person.indicators || []) {
      const bucket = counts[item.type] || { detected: 0, waiting: 0, unavailable: 0, not_detected: 0 };
      bucket[item.status] = (bucket[item.status] || 0) + 1;
      counts[item.type] = bucket;
    }
  }
  return counts;
}

export default function BehaviourAnalysis({
  state,
  sources,
  postControl,
  streamKey,
  setStreamKey,
  selectedId,
  onSelectPerson,
}) {
  const videoTime = state?.timestamp || "00:00:00.000";
  const behaviour = state?.behaviour || { people: [], events: [], available: true, note: "" };
  const pipeline = state?.pipeline || {};
  const people = Array.isArray(behaviour.people) ? behaviour.people : [];
  const events = Array.isArray(behaviour.events) ? behaviour.events : [];
  const objects = state?.objects || {};
  const objectDetections = Array.isArray(behaviour.object_detections) ? behaviour.object_detections : [];
  const objectInput = behaviour.object_input || (objects.error ? "unavailable" : objectDetections.length ? "populated" : "empty");
  const objectDebug = behaviour.object_debug || {};
  const itemAlerts = suspectedAlertsFromState(state);
  const selected = people.find((person) => person.person_id === selectedId) || people[0] || null;
  const grouped = [];
  for (const item of sources) {
    const key = item.group_label || "Other clips";
    const last = grouped[grouped.length - 1];
    if (!last || last.label !== key) grouped.push({ label: key, items: [item] });
    else last.items.push(item);
  }
  const hasSources = sources.length > 0;
  const hasSource = Boolean(state?.source || pipeline.source);
  const live = state?.status === "playing" && !state?.paused;
  const behaviourStatus = pipeline.behaviour || {};
  const analysisState = behaviourStatus.state || (live ? "running" : "ready");
  const counts = indicatorCounts(people);
  const history = selected?.history || {};

  async function openSource(source) {
    if (!source) return;
    await postControl({ action: "open", source });
    await postControl({ action: "resume" });
    setStreamKey((value) => value + 1);
  }

  return (
    <section className="page ba-page">
      <div className="ba-heading">
        <h1>BEHAVIOR ANALYSIS</h1>
        <p className="value-lead">
          Component 3 — Suspicious Behaviour Analysis. Person IDs identify who Member 3 scored.
          A person box is not shoplifting, concealment, or object interaction.
        </p>
      </div>

      {behaviour.error && (
        <div className="card">
          <h3>ANALYSIS UNAVAILABLE</h3>
          <p className="value-lead">{behaviour.error}</p>
        </div>
      )}

      <div className="card ba-preview">
        <div className="card-head">
          <span className={`live-flag ${live ? "on" : "off"}`}>
            <span className="pulse" />
            {live ? "LIVE" : "PAUSED"}
          </span>
          <strong>Behaviour analysis preview</strong>
          <small className="ba-meta">Member 3 live overlay on the shared webcam session</small>
          <select
            className="view-mode scenario-select"
            value={state?.source || ""}
            onChange={(event) => openSource(event.target.value)}
          >
            <option value="">Select live webcam or a video</option>
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
        <div className="video-wrap ba-video">
          <img alt="Behaviour analysis live preview" src={`/stream_behaviour.mjpg?k=${streamKey}`} />
          <div className="badge">COMPONENT 3 · LIVE DEMO</div>
          {state?.status === "loading" && (
            <div className="overlay">Waiting for the shared live session to load the person detector…</div>
          )}
          {state?.source_error && state?.status !== "loading" && (
            <div className="overlay">
              {state.source_error}
              <small>{SOURCE_HELP}</small>
            </div>
          )}
          {!hasSource && !state?.source_error && state?.status !== "loading" && (
            <div className="overlay">
              Select Live webcam so Member 3 can receive tracked people.
              <small>{SOURCE_HELP}</small>
            </div>
          )}
        </div>
        <div className="ba-toolbar">
          <button className="primary" type="button" onClick={() => postControl({ action: live ? "pause" : "resume" })}>
            {live ? "Pause" : "Resume"}
          </button>
          <button type="button" onClick={() => postControl({ action: "reset" })}>Reset</button>
          <p className="hint">
            {behaviour.has_tracking_input || people.length
              ? "Member 3 is receiving tracked person IDs from the shared session."
              : "Member 3 is waiting for tracked people. Tracking metrics stay on the Tracking tab."}
          </p>
        </div>
      </div>

      <div className="ba-status">
        <article className="card">
          <h3>ANALYSIS SESSION</h3>
          <b className={statusClass(analysisState)}>{analysisState}</b>
          <ul className="summary">
            <li><span>Bridge</span><b>{behaviourStatus.loaded ? "loaded" : "not loaded"}</b></li>
            <li><span>Available</span><b>{behaviour.available === false ? "no" : "yes"}</b></li>
            <li><span>Tracking input</span><b>{behaviour.has_tracking_input ? "receiving IDs" : "none this frame"}</b></li>
            <li><span>Object evidence</span><b>{behaviour.object_evidence ? `yes (${objectDetections.length})` : objectInput}</b></li>
          </ul>
        </article>
        <article className="card">
          <h3>PEOPLE UNDER ANALYSIS</h3>
          <b className="ok">{behaviour.people_analysed ?? people.length}</b>
          <ul className="summary">
            <li><span>Currently analysed</span><b>{people.length}</b></li>
            <li><span>Assessments complete</span><b>{behaviour.assessments_complete ?? 0}</b></li>
            <li><span>Waiting / pending</span><b>{people.filter((person) => person.score_status === "pending").length}</b></li>
            <li><span>History in store</span><b>{behaviourStatus.people_scored ?? people.length}</b></li>
          </ul>
        </article>
        <article className="card">
          <h3>ANALYSIS TIMING</h3>
          <b>{live ? "running" : hasSource ? (state?.paused ? "paused" : "ready") : "idle"}</b>
          <ul className="summary">
            <li><span>Frame</span><b>{behaviour.frame_id ?? state?.frame_id ?? 0}</b></li>
            <li><span>Timestamp</span><b>{videoTime}</b></li>
            <li><span>Elapsed</span><b>{formatDuration(behaviour.elapsed_s)}</b></li>
            <li><span>Process time</span><b>{formatMs(behaviour.process_ms)}</b></li>
          </ul>
        </article>
        <article className="card">
          <h3>EVIDENCE LIMITS</h3>
          <b className={objectInput === "populated" ? "ok" : objectInput === "unavailable" || objects.error ? "danger" : "muted"}>
            {objects.error || objectInput === "unavailable"
              ? "object detector unavailable"
              : objectInput === "populated"
                ? `${objectDetections.length} non-person object${objectDetections.length === 1 ? "" : "s"}`
                : "Member 1 ran; no non-person boxes"}
          </b>
          <p className="hint">
            {objects.error
              ? objects.error
              : "Object boxes are diagnostic input. They are not pickup or concealment proof. COCO cannot confirm a hand grasp or a store SKU."}
          </p>
        </article>
      </div>

      <div className="card value-card">
        <div className="card-head">
          <h3>MEMBER 1 OBJECTS THIS FRAME</h3>
          <small>
            {objectInput === "unavailable"
              ? "Detector unavailable — detections are None, not an empty success"
              : objectInput === "populated"
                ? `${objectDetections.length} non-person box${objectDetections.length === 1 ? "" : "es"} reached Member 3`
                : `Member 1 returned ${objectDebug.raw ?? 0} raw box${Number(objectDebug.raw) === 1 ? "" : "es"}; ${objectDebug.dropped_person ?? 0} person / ${objectDebug.dropped_invalid ?? 0} invalid filtered`}
          </small>
        </div>
        <ul className="events ba-objects">
          {objectDetections.map((item, index) => (
            <li key={`${item.class}-${item.detection_id ?? index}-${index}`}>
              <time>{item.class}</time>
              <span>
                {item.bbox ? `[${item.bbox.map((value) => Number(value).toFixed(0)).join(", ")}]` : "bbox missing"}
                {item.confidence != null ? ` · ${Number(item.confidence).toFixed(2)}` : ""}
                {item.detection_id != null ? ` · id ${item.detection_id}` : ""}
              </span>
            </li>
          ))}
          {!objectDetections.length && (
            <li>
              {objectInput === "unavailable" || objects.error
                ? `No objects delivered to Member 3. ${objects.error || "Member 1 did not run this frame."}`
                : "No non-person objects this frame. This is not an alert, and person boxes are not treated as merchandise."}
            </li>
          )}
        </ul>
      </div>

      <div className="card value-card">
        <div className="card-head">
          <h3>SUSPICIOUS BEHAVIOUR INDICATORS</h3>
          <small>Implemented Member 3 detectors — not Component 2 tracking metrics</small>
        </div>
        <div className="ba-indicators">
          {Object.entries(INDICATOR_LABELS).map(([type, label]) => {
            const bucket = counts[type] || {};
            return (
              <article key={type}>
                <small>{label}</small>
                <b className={bucket.detected ? "ok" : "muted"}>{bucket.detected ? `${bucket.detected} detected` : "none detected"}</b>
                <span>
                  waiting {bucket.waiting || 0} · unavailable {bucket.unavailable || 0} · not detected {bucket.not_detected || 0}
                </span>
              </article>
            );
          })}
        </div>
      </div>

      <div className="card value-card">
        <div className="card-head">
          <h3>SUSPICIOUS ACTIVITY ALERTS</h3>
          <small>Item pickup and concealment only — person ID and snap, not a theft verdict</small>
        </div>
        <AlertFeed
          alerts={itemAlerts}
          empty={
            objects.error
              ? `No alerts. Member 1 is unavailable: ${objects.error}`
              : "No item-pickup or concealment alert yet. A snap appears only when that suspected behaviour fires."
          }
        />
      </div>

      <div className="card value-card">
        <div className="card-head">
          <h3>PERSON ASSESSMENTS</h3>
          <small>{videoTime} · 0 is a measured Member 3 score · — means this person has not been analysed</small>
        </div>
        <div className="ba-table-wrap">
          <table>
            <thead>
              <tr>
                <th>PERSON ID</th>
                <th>ZONE</th>
                <th>INDICATORS</th>
                <th>BEHAVIOUR SCORE</th>
                <th>COMBINED</th>
                <th>LEVEL</th>
                <th>DURATION</th>
                <th>HISTORY</th>
                <th>CONTRIBUTING</th>
              </tr>
            </thead>
            <tbody>
              {people.map((person) => {
                const detected = (person.indicators || [])
                  .filter((item) => item.status === "detected")
                  .map((item) => `${item.type}:${item.score}`)
                  .join(", ");
                const behaviourScore = scoreDisplay(person, "behaviour_score");
                const combinedScore = scoreDisplay(person, "combined_score");
                return (
                  <tr
                    key={person.person_id}
                    className={selected?.person_id === person.person_id ? "selected-row" : ""}
                    onClick={() => onSelectPerson(person.person_id)}
                  >
                    <td>{person.person_id}</td>
                    <td>{displayZone(person.zone)}</td>
                    <td>{detected || (person.score_status === "pending" ? "pending" : "none detected")}</td>
                    <td className={behaviourScore.kind}>
                      {behaviourScore.text}
                      {behaviourScore.hint ? <small className="score-hint"> {behaviourScore.hint}</small> : null}
                    </td>
                    <td className={combinedScore.kind}>
                      {combinedScore.text}
                      {combinedScore.hint ? <small className="score-hint"> {combinedScore.hint}</small> : null}
                    </td>
                    <td className={levelClass(person.suspicious_level)}>{person.suspicious_level || "none"}</td>
                    <td>{formatDuration(person.duration_s)}</td>
                    <td>{person.history_length ?? person.history?.length ?? "—"}</td>
                    <td>{(person.contributing || []).join(", ") || "none"}</td>
                  </tr>
                );
              })}
              {!people.length && (
                <tr>
                  <td colSpan="9">
                    {hasSources
                      ? "No Member 3 assessments yet. Resume the shared session so tracked IDs can be analysed."
                      : `No people to analyse. ${SOURCE_HELP}`}
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      <div className="lower behaviour-lower">
        <div className="card details">
          <h3>MOVEMENT HISTORY {selected ? `(ID ${selected.person_id})` : ""}</h3>
          {selected ? (
            <dl>
              <div><dt>person id</dt><dd>{selected.person_id}</dd></div>
              <div><dt>history length</dt><dd>{history.length ?? selected.history_length ?? "—"}</dd></div>
              <div><dt>observation</dt><dd>{formatDuration(history.duration_s ?? selected.duration_s)}</dd></div>
              <div><dt>current zone</dt><dd>{displayZone(history.zone || selected.zone)}</dd></div>
              <div><dt>zones visited</dt><dd>{(history.zones_visited || []).join(", ") || "—"}</dd></div>
              <div><dt>zone transitions</dt><dd>{history.zone_transitions?.length ? history.zone_transitions.map((item) => `${item.from} → ${item.to}`).join("; ") : "—"}</dd></div>
              <div><dt>shelf visits</dt><dd>{history.had_shelf_contact ? Object.entries(history.shelf_counts || {}).map(([name, count]) => `${name}:${count}`).join(", ") : "none yet"}</dd></div>
              <div><dt>shelf repeats</dt><dd>{history.shelf_repeats ?? 0}</dd></div>
              <div><dt>longest still</dt><dd>{formatDuration(history.longest_still_s)}</dd></div>
            </dl>
          ) : (
            <p>Select a person with a Member 3 history.</p>
          )}
        </div>
        <div className="card">
          <h3>DETECTOR STATUS</h3>
          <ul className="events">
            {(selected?.indicators || []).map((item) => (
              <li key={item.type}>
                <time className={statusClass(item.status)}>{item.status}</time>
                <span>
                  {INDICATOR_LABELS[item.type] || item.type}
                  {item.score != null ? `: ${item.score}` : ""}
                  {item.detail ? ` — ${item.detail}` : ""}
                </span>
              </li>
            ))}
            {!selected && <li>No person selected.</li>}
          </ul>
        </div>
        <div className="card">
          <h3>ASSESSMENT</h3>
          {selected ? (
            <dl>
              <div><dt>behaviour score</dt><dd className={scoreDisplay(selected, "behaviour_score").kind}>{scoreDisplay(selected, "behaviour_score").text}</dd></div>
              <div><dt>combined score</dt><dd className={scoreDisplay(selected, "combined_score").kind}>{scoreDisplay(selected, "combined_score").text}</dd></div>
              <div><dt>suspicious level</dt><dd className={levelClass(selected.suspicious_level)}>{selected.suspicious_level || "none"}</dd></div>
              <div><dt>window</dt><dd>{selected.window_seconds != null ? `${selected.window_seconds} s` : "—"}</dd></div>
              <div><dt>contributing</dt><dd>{(selected.contributing || []).join(", ") || "none"}</dd></div>
              <div><dt>not contributing</dt><dd>{(selected.not_contributing || []).map(reasonLabel).join("; ") || "none"}</dd></div>
            </dl>
          ) : (
            <p>No assessment selected.</p>
          )}
          <p className="hint">Combined score uses overlapping timed indicators. Isolated loitering stays level none. No confidence score is produced by Member 3.</p>
        </div>
        <div className="card">
          <h3>BEHAVIOUR EVENTS</h3>
          <ul className="events">
            {events.slice(0, 10).map((event, index) => (
              <li key={`${event.person_id}-${event.type}-${event.elapsed_s}-${index}`}>
                <time className={event.level ? levelClass(event.level) : undefined}>{formatDuration(event.elapsed_s)}</time>
                <span>
                  ID {event.person_id} · {event.type}:{event.score}
                  {event.object_class ? ` · ${event.object_class}` : ""}
                  {event.level ? ` · ${event.level}` : ""}
                  {event.zone ? ` · ${event.zone}` : ""}
                  {event.evidence ? ` — ${event.evidence}` : ""}
                </span>
              </li>
            ))}
            {!events.length && (
              <li>No Member 3 detector events yet. Events appear when an implemented indicator actually fires, not as placeholders.</li>
            )}
          </ul>
        </div>
      </div>
      <footer>
        Tracking evaluation (MOTA, MOTP, IDF1, ID switches, detector FPS) remains on the Tracking tab.
        Member 3 does not treat person detection as concealment, object interaction, or shoplifting.
      </footer>
    </section>
  );
}
