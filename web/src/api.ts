export type Destination = {
  place_id: number;
  name: string;
  category: string;
  neighborhood: string;
  distance_km: number;
  standard_fare: number;
  score: number;
  reasons: string[];
};

export type RideOption = {
  mode: string;
  label: string;
  price: number;
  seats: number;
  pickup_minutes: number;
  available: boolean;
  probability: number;
};

export type RideOptions = {
  distance_km: number;
  surge: number;
  recommended: string;
  options: RideOption[];
  personalized: boolean;
  timings_ms: Record<string, number>;
};

export type Recommendations = {
  request_id: string;
  origin: { place_id: number; name: string; neighborhood: string };
  experiment: { name: string; variant: string };
  path: string;
  destinations: Destination[];
  ride_options: RideOptions | null;
  timings_ms: Record<string, number>;
  model_version: string;
};

export type Rider = {
  rider_id: number;
  segment: string;
  home: { place_id: number; name: string };
  work: { place_id: number; name: string };
  recent_trips: { place_id: number; name: string }[];
  ride_modes: Record<string, number>;
  top_places: { place_id: number; name: string; visits: number }[];
  experiment_variant: string;
};

export type Place = { place_id: number; name: string; category: string; x_km: number; y_km: number };

async function call<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(`/api${path}`, {
    method: body ? "POST" : "GET",
    headers: body ? { "content-type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => null);
    throw new Error(detail?.detail ?? `Request failed (${res.status})`);
  }
  return res.json();
}

export const getRider = (id: number) => call<Rider>(`/v1/riders/${id}`);
export const getPlaces = () => call<Place[]>("/v1/places");
export const getRecommendations = (rider_id: number, place_id: number, at: string) =>
  call<Recommendations>("/v1/recommendations", { rider_id, origin: { place_id }, at, include_ride_options: false });
export const getRideOptions = (rider_id: number, place_id: number, destination: number, at: string, group_size: number) =>
  call<RideOptions>("/v1/ride-options", {
    rider_id,
    origin: { place_id },
    destination_place_id: destination,
    at,
    group_size,
  });
export const completeRide = (rider_id: number, destination: number, mode: string, request_id: string) =>
  call<{ suggested_rank: number | null; variant: string }>("/v1/events/ride-completed", {
    rider_id,
    destination_place_id: destination,
    mode,
    request_id,
  });
