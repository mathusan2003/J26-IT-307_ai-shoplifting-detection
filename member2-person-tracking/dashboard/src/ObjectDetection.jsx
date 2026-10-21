const SOURCE_HELP =
  "Select Live webcam for the shared session, or add .mp4 files to member2-person-tracking/outputs or member2-person-tracking/data.";

export default function ObjectDetection({
  state,
  sources,
  postControl,
  streamKey,
  setStreamKey,
}) {
  const objects = state?.objects || { detections: [], available: true, note: "" };
  const detections = Array.isArray(objects.detections) ? objects.detections : [];
  const grouped = [];
  for (const item of sources) {
    const key = item.group_label || "Other clips";
    const last = grouped[grouped.length - 1];
    if (!last || last.label !== key) grouped.push({ label: key, items: [item] });
    else last.items.push(item);
  }
  const live = Boolean(state?.source) && !state?.paused && state?.status !== "loading";
  const hasSource = Boolean(state?.source);

  async function openSource(source) {
    if (!source) return;
    await postControl({ action: "open", source });
    await postControl({ action: "resume" });
    setStreamKey((value) => value + 1);
  }

  return (
    <section className="page ba-heading">
      <h1>OBJECT DETECTION</h1>
      <p className="value-lead">
        Component 1 — pretrained YOLO prototype. Boxes and labels come from the loaded COCO model.
        This page does not track people, score behaviour, or decide shoplifting. It does not recognise every retail product.
      </p>
      <p className="value-lead">{objects.note}</p>

      {objects.error && (
        <div className="card">
          <h3>DETECTOR UNAVAILABLE</h3>
          <p className="value-lead">{objects.error}</p>
        </div>
      )}

      <div className="card ba-preview">
        <div className="card-head">
          <span className={`live-flag ${live ? "on" : "off"}`}>
            <span className="pulse" />
            {live ? "LIVE" : "PAUSED"}
          </span>
          <strong>Object detection preview</strong>
          <small className="ba-meta">Shared live session — Member 1 overlay, not Tracking</small>
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
          <img alt="Object detection live preview" src={`/stream_objects.mjpg?k=${streamKey}`} />
          <div className="badge">COMPONENT 1 · PRETRAINED YOLO</div>
          <div className="fps-chip">{Number(objects.infer_fps || 0).toFixed(1)} DET FPS</div>
          {state?.status === "loading" && (
            <div className="overlay">Waiting for the shared live session to load detectors…</div>
          )}
          {state?.source_error && state?.status !== "loading" && (
            <div className="overlay">
              {state.source_error}
              <small>{SOURCE_HELP}</small>
            </div>
          )}
          {!hasSource && !state?.source_error && state?.status !== "loading" && (
            <div className="overlay">
              Select Live webcam so Member 1 can run on the shared frames.
              <small>{SOURCE_HELP}</small>
            </div>
          )}
        </div>
        <div className="ba-toolbar">
          <button
            className="primary"
            type="button"
            onClick={() => postControl({ action: state?.paused === false ? "pause" : "resume" })}
          >
            {state?.paused === false ? "Pause" : "Resume"}
          </button>
          <button
            type="button"
            onClick={async () => {
              await postControl({ action: "reset" });
              setStreamKey((value) => value + 1);
            }}
          >
            Reset
          </button>
          <p className="hint">
            {detections.length
              ? `${detections.length} object box${detections.length === 1 ? "" : "es"} in this frame.`
              : "No objects detected in this frame."}
          </p>
        </div>
      </div>

      <div className="ba-status">
        <div className="card">
          <h3>PROTOTYPE SESSION</h3>
          <ul className="summary">
            <li><span>Model</span><b>{objects.model || "not loaded"}</b></li>
            <li><span>Available</span><b>{objects.available ? "yes" : "no"}</b></li>
            <li><span>Device</span><b>{objects.device || "—"}</b></li>
            <li><span>Supported classes</span><b>{objects.class_count || 0}</b></li>
          </ul>
        </div>
        <div className="card">
          <h3>THIS FRAME</h3>
          <ul className="summary">
            <li><span>Frame</span><b>{objects.frame_id ?? state?.frame_id ?? "—"}</b></li>
            <li><span>Detections</span><b>{detections.length}</b></li>
            <li><span>Inference FPS</span><b>{objects.infer_fps != null ? Number(objects.infer_fps).toFixed(1) : "—"}</b></li>
            <li><span>Source</span><b>{state?.source || "—"}</b></li>
          </ul>
        </div>
        <div className="card">
          <h3>WHAT THIS IS NOT</h3>
          <p className="value-lead">
            A person box is not an object interaction. Overlap is not ownership.
            COCO labels are not store SKUs. No concealment or shoplifting score is produced here.
          </p>
        </div>
        <div className="card">
          <h3>SHARED PIPELINE</h3>
          <p className="value-lead">
            Tracking and Behavior Analysis keep their own overlays. Member 1 only adds this page and `/api/objects`.
          </p>
        </div>
      </div>

      <div className="card" style={{ marginTop: 16 }}>
        <h3>DETECTIONS IN THIS FRAME</h3>
        <div className="ba-table-wrap">
          <table>
            <thead>
              <tr>
                <th>ID</th>
                <th>CLASS</th>
                <th>CLASS ID</th>
                <th>CONFIDENCE</th>
                <th>BBOX</th>
              </tr>
            </thead>
            <tbody>
              {detections.length ? detections.map((item) => (
                <tr key={item.detection_id}>
                  <td>{item.detection_id}</td>
                  <td>{item.class_name}</td>
                  <td>{item.class_id}</td>
                  <td>{Number(item.confidence || 0).toFixed(2)}</td>
                  <td>
                    {item.bbox
                      ? `${item.bbox.x1}, ${item.bbox.y1}, ${item.bbox.x2}, ${item.bbox.y2}`
                      : "—"}
                  </td>
                </tr>
              )) : (
                <tr>
                  <td colSpan={5} className="muted">No objects in this frame.</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}
