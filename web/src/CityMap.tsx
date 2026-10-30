import type { Destination, Place } from "./api";

const SIZE = 20;

export function CityMap({ places, origin, destinations, selected, onSelect }: {
  places: Place[];
  origin: number | null;
  destinations: Destination[];
  selected: number | null;
  onSelect: (placeId: number) => void;
}) {
  const byId = new Map(places.map((p) => [p.place_id, p]));
  const from = origin ? byId.get(origin) : undefined;
  const to = selected ? byId.get(selected) : undefined;
  return (
    <svg className="map" viewBox={`0 0 ${SIZE} ${SIZE}`} role="img" aria-label="City map of pickup and suggested places">
      {[4, 8, 12, 16].map((v) => (
        <g key={v} className="street">
          <line x1={v} y1={0} x2={v} y2={SIZE} />
          <line x1={0} y1={v} x2={SIZE} y2={v} />
        </g>
      ))}
      {places.filter((p) => p.category !== "residential").map((p) => (
        <circle key={p.place_id} className="poi" cx={p.x_km} cy={SIZE - p.y_km} r={0.09} />
      ))}
      {from && to && <line className="route" x1={from.x_km} y1={SIZE - from.y_km} x2={to.x_km} y2={SIZE - to.y_km} />}
      {destinations.map((d, i) => {
        const p = byId.get(d.place_id);
        if (!p) return null;
        return (
          <g key={d.place_id} className={`pin ${d.place_id === selected ? "on" : ""}`} onClick={() => onSelect(d.place_id)}>
            <circle cx={p.x_km} cy={SIZE - p.y_km} r={0.55} />
            <text x={p.x_km} y={SIZE - p.y_km + 0.2}>{i + 1}</text>
          </g>
        );
      })}
      {from && <circle className="pickup" cx={from.x_km} cy={SIZE - from.y_km} r={0.42} />}
    </svg>
  );
}
