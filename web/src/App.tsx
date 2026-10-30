import { useCallback, useEffect, useMemo, useState } from "react";
import {
  completeRide, getPlaces, getRecommendations, getRider, getRideOptions,
  type Place, type Recommendations, type Rider, type RideOptions,
} from "./api";
import { CityMap } from "./CityMap";

const MOMENTS = [
  { label: "Tue 8:00 am", at: "2026-10-20T08:00:00", pickup: "home" },
  { label: "Fri 6:00 pm", at: "2026-10-23T18:00:00", pickup: "work" },
  { label: "Sat 10:30 pm", at: "2026-10-24T22:30:00", pickup: "home" },
  { label: "Sun 11:00 am", at: "2026-10-25T11:00:00", pickup: "home" },
] as const;

const RIDER_IDS = Array.from({ length: 24 }, (_, i) => i);
const STAGES = ["features", "retrieval", "ranking", "heuristic", "explain", "fallback"];

export default function App() {
  const [riders, setRiders] = useState<Rider[]>([]);
  const [places, setPlaces] = useState<Place[]>([]);
  const [riderId, setRiderId] = useState<number | null>(null);
  const [moment, setMoment] = useState(0);
  const [pickup, setPickup] = useState<number | null>(null);
  const [recs, setRecs] = useState<Recommendations | null>(null);
  const [selected, setSelected] = useState<number | null>(null);
  const [group, setGroup] = useState(1);
  const [ride, setRide] = useState<RideOptions | null>(null);
  const [chosenMode, setChosenMode] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const rider = useMemo(() => riders.find((r) => r.rider_id === riderId) ?? null, [riders, riderId]);
  const at = MOMENTS[moment].at;

  useEffect(() => {
    Promise.all([Promise.all(RIDER_IDS.map(getRider)), getPlaces()])
      .then(([rs, ps]) => {
        setRiders(rs);
        setPlaces(ps);
        setRiderId(rs[0].rider_id);
        setPickup(rs[0].home.place_id);
      })
      .catch((e) => setError(e.message));
  }, []);

  const refresh = useCallback(async (id: number, place: number) => {
    setError(null);
    setBusy(true);
    try {
      const r = await getRecommendations(id, place, at);
      setRecs(r);
      setSelected(r.destinations[0]?.place_id ?? null);
    } catch (e) {
      setRecs(null);
      setSelected(null);
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }, [at]);

  useEffect(() => {
    if (riderId !== null && pickup !== null) refresh(riderId, pickup);
  }, [riderId, pickup, refresh]);

  useEffect(() => {
    if (riderId === null || pickup === null || selected === null) return;
    let live = true;
    getRideOptions(riderId, pickup, selected, at, group)
      .then((o) => {
        if (!live) return;
        setRide(o);
        setChosenMode(o.recommended);
      })
      .catch((e) => live && setError(e.message));
    return () => { live = false; };
  }, [riderId, pickup, selected, at, group]);

  function chooseRider(id: number) {
    const r = riders.find((x) => x.rider_id === id)!;
    setRiderId(id);
    setPickup(MOMENTS[moment].pickup === "work" ? r.work.place_id : r.home.place_id);
    setNotice(null);
  }

  function chooseMoment(i: number) {
    setMoment(i);
    if (rider) setPickup(MOMENTS[i].pickup === "work" ? rider.work.place_id : rider.home.place_id);
    setNotice(null);
  }

  async function book() {
    if (!recs || riderId === null || selected === null || !chosenMode) return;
    const place = recs.destinations.find((d) => d.place_id === selected);
    const done = await completeRide(riderId, selected, chosenMode, recs.request_id);
    const fresh = await getRider(riderId);
    setRiders((rs) => rs.map((r) => (r.rider_id === riderId ? fresh : r)));
    setPickup(selected);
    setNotice(`Arrived at ${place?.name.split(" · ")[0]}. That was suggestion #${done.suggested_rank ?? "–"}, and your recent trips just updated.`);
  }

  const total = recs ? STAGES.reduce((s, k) => s + (recs.timings_ms[k] ?? 0), 0) : 0;

  return (
    <div className="app">
      <header>
        <div className="brand"><span className="mark" />Waypoint</div>
        <div className="controls">
          <label>Rider
            <select value={riderId ?? ""} onChange={(e) => chooseRider(Number(e.target.value))}>
              {riders.map((r) => (
                <option key={r.rider_id} value={r.rider_id}>#{r.rider_id} · {r.segment}</option>
              ))}
            </select>
          </label>
          <div className="moments" role="group" aria-label="Time of day">
            {MOMENTS.map((m, i) => (
              <button key={m.label} className={i === moment ? "on" : ""} onClick={() => chooseMoment(i)}>{m.label}</button>
            ))}
          </div>
        </div>
      </header>

      {error && <div className="banner error" role="alert">{error}</div>}
      {notice && <div className="banner" role="status">{notice}</div>}

      <main>
        <section className="panel">
          <p className="eyebrow">Picking up at</p>
          <h1>{recs?.origin.name.split(" · ")[0] ?? "…"}</h1>
          <h2>Where to?</h2>
          <ol className={`dest ${busy ? "busy" : ""}`}>
            {recs?.destinations.map((d, i) => (
              <li key={d.place_id}>
                <button className={d.place_id === selected ? "on" : ""} onClick={() => setSelected(d.place_id)}>
                  <span className="rank">{i + 1}</span>
                  <span className="info">
                    <strong>{d.name.split(" · ")[0]}</strong>
                    <small>{d.reasons.join(" · ")}</small>
                  </span>
                  <span className="fare">${d.standard_fare.toFixed(2)}<small>{d.distance_km} km</small></span>
                </button>
              </li>
            ))}
          </ol>

          {ride && (
            <>
              <div className="row">
                <h2>Choose a ride</h2>
                <label className="party">Riders
                  <select value={group} onChange={(e) => setGroup(Number(e.target.value))}>
                    {[1, 2, 3, 4, 5, 6].map((n) => <option key={n}>{n}</option>)}
                  </select>
                </label>
              </div>
              <div className="modes">
                {ride.options.map((o) => (
                  <button key={o.mode} disabled={!o.available} className={o.mode === chosenMode ? "on" : ""} onClick={() => setChosenMode(o.mode)}>
                    <span><strong>{o.label}</strong><small>{o.available ? `${o.pickup_minutes} min pickup · seats ${o.seats}` : `Seats ${o.seats}: too small`}</small></span>
                    <span className="price">${o.price.toFixed(2)}{o.mode === ride.recommended && <small>Best match</small>}</span>
                  </button>
                ))}
              </div>
              <button className="primary" onClick={book} disabled={!chosenMode}>Request ride</button>
            </>
          )}
        </section>

        <section className="side">
          <CityMap places={places} origin={pickup} destinations={recs?.destinations ?? []} selected={selected} onSelect={setSelected} />
          {recs && (
            <div className="hood">
              <h3>Under the hood</h3>
              <dl>
                <div><dt>Experiment arm</dt><dd><span className={`tag ${recs.experiment.variant}`}>{recs.experiment.variant}</span></dd></div>
                <div><dt>Served by</dt><dd>{recs.path === "model" ? "Two-tower + ranker" : recs.path === "heuristic" ? "Frequency heuristic" : recs.path}</dd></div>
                <div><dt>Service time</dt><dd>{total.toFixed(1)} ms</dd></div>
                <div><dt>Model version</dt><dd>{recs.model_version}</dd></div>
              </dl>
              <ul className="stages">
                {STAGES.filter((s) => recs.timings_ms[s] !== undefined).map((s) => (
                  <li key={s}>
                    <span>{s}</span>
                    <i style={{ width: `${Math.max(2, (recs.timings_ms[s] / Math.max(total, 0.01)) * 100)}%` }} />
                    <span>{recs.timings_ms[s].toFixed(2)} ms</span>
                  </li>
                ))}
              </ul>
              {rider && (
                <>
                  <h3>Recent trips (real-time features)</h3>
                  <ol className="recent">
                    {rider.recent_trips.slice(0, 4).map((t, i) => <li key={`${t.place_id}-${i}`}>{t.name.split(" · ")[0]}</li>)}
                  </ol>
                </>
              )}
            </div>
          )}
        </section>
      </main>
      <footer>Portfolio demo with synthetic data. Not affiliated with any ride-share company.</footer>
    </div>
  );
}
