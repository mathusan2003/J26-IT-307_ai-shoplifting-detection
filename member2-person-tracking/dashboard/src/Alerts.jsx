export const INDICATOR_LABELS = {
  loitering: "Loitering",
  repeated_shelf_visit: "Repeated shelf visits",
  abnormal_movement: "Abnormal movement",
  object_interaction: "Object interaction",
  possible_concealment: "Possible concealment",
  item_pickup_suspected: "Item pickup suspected",
  possible_item_concealment: "Possible item concealment",
  item_removed_from_shelf: "Item removed from shelf",
  repeated_shelf_interaction: "Repeated shelf interaction",
};

function levelClass(level) {
  if (level === "high") return "danger";
  if (level === "medium" || level === "low") return "warn";
  return "ok";
}

function formatDuration(seconds) {
  if (seconds == null || Number.isNaN(Number(seconds))) return "—";
  return `${Number(seconds).toFixed(1)} s`;
}

const SHOPLIFTING_ALERT_TYPES = new Set([
  "item_pickup_suspected",
  "possible_item_concealment",
  "item_removed_from_shelf",
  "possible_concealment",
]);

export function suspectedAlertsFromState(state) {
  const behaviour = state?.behaviour || {};
  const fromPayload = Array.isArray(behaviour.suspected_alerts) ? behaviour.suspected_alerts : [];
  const fromEvents = Array.isArray(behaviour.events) ? behaviour.events : [];
  const rows = fromPayload.length ? fromPayload : fromEvents;
  return rows.filter(
    (item) => item && item.person_id != null && SHOPLIFTING_ALERT_TYPES.has(item.type),
  );
}

export function AlertFeed({ alerts, empty }) {
  return (
    <div className="alert-grid">
      {alerts.map((alert, index) => (
        <article className={`alert-card ${levelClass(alert.level)}`} key={`${alert.person_id}-${alert.type}-${alert.elapsed_s}-${index}`}>
          {alert.crop_jpeg ? (
            <img className="alert-crop" alt={`Person ${alert.person_id}`} src={alert.crop_jpeg} />
          ) : (
            <div className="alert-crop empty">No image</div>
          )}
          <div>
            <b>ID {alert.person_id}</b>
            <small className={levelClass(alert.level)}>{alert.level || "none"}</small>
            <p>{INDICATOR_LABELS[alert.type] || alert.type}</p>
            <span>
              {formatDuration(alert.elapsed_s)}
              {alert.object_class ? ` · ${alert.object_class}` : ""}
              {alert.score != null ? ` · score ${alert.score}` : ""}
              {alert.zone ? ` · ${alert.zone}` : ""}
            </span>
            {alert.evidence ? <em>{alert.evidence}</em> : null}
          </div>
        </article>
      ))}
      {!alerts.length && <p className="hint">{empty}</p>}
    </div>
  );
}

export default function Alerts({ state }) {
  const alerts = suspectedAlertsFromState(state);
  const live = state?.status === "playing" && !state?.paused;
  return (
    <section className="page ba-page">
      <div className="ba-heading">
        <h1>ALERTS</h1>
        <p className="value-lead">
          Only item pickup and possible concealment. Each card has the person ID and a snap from that
          frame. Abnormal movement and loitering stay on Behaviour Analysis. This is not a theft verdict.
        </p>
      </div>
      <div className="card value-card">
        <div className="card-head">
          <span className={`live-flag ${live ? "on" : "off"}`}>
            <span className="pulse" />
            {live ? "LIVE" : "PAUSED"}
          </span>
          <h3>ITEM PICKUP / CONCEALMENT</h3>
          <small>{alerts.length} shoplifting-related alert{alerts.length === 1 ? "" : "s"} with person ID and snap</small>
        </div>
        <AlertFeed
          alerts={alerts}
          empty="No item-pickup or concealment alert yet. A snap appears only when that suspected behaviour actually fires."
        />
      </div>
    </section>
  );
}
