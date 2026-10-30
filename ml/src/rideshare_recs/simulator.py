"""Synthetic city: places, riders and eight weeks of trips.

No real Lyft data is used. The generator plants the structure a real rider history has (weekday commutes, gym and
lunch habits, weekend errands, nightlife, airport trips, persona-driven ride-mode choice and surge price
sensitivity) plus noise and exploration, so the models can be compared against simple baselines honestly.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd

CITY_KM = 20.0
CENTER = np.array([10.0, 10.0])

CATEGORIES = ["residential", "office", "restaurant", "bar", "gym", "grocery", "mall", "hospital", "university",
              "stadium", "train_station", "airport"]
# category: (count, spread around downtown in km)
PUBLIC_PLACES = {"office": (60, 2.5), "restaurant": (80, 4.0), "bar": (30, 2.0), "gym": (20, 5.0),
                 "grocery": (25, 6.0), "mall": (6, 6.0), "hospital": (5, 5.0), "university": (3, 4.0),
                 "stadium": (2, 3.0), "train_station": (5, 4.0)}
AIRPORTS = [(18.5, 2.0)]

PERSONAS = ["commuter", "student", "nightlife", "business", "family"]
PERSONA_SHARE = [0.35, 0.15, 0.15, 0.15, 0.20]

RIDE_MODES = ["wait_save", "standard", "xl", "lux"]
MODE_PRICE = np.array([0.78, 1.0, 1.55, 2.2])      # multiplier on the standard fare
MODE_CAPACITY = np.array([2, 4, 6, 4])
STADIUM = -2                                        # itinerary placeholder, resolved to one of the stadiums

_NAME_PREFIX = ["Harbor", "Maple", "Summit", "Juniper", "Mission", "Granite", "Willow", "Union", "Cedar", "Bayview",
                "Sunset", "Lakeside", "Riverside", "Pioneer", "Golden", "Redwood", "Market", "Civic", "Orchard",
                "Hillcrest", "Beacon", "Alder", "Liberty", "Canyon", "Meridian", "Elm", "Northgate", "Oak", "Pacific",
                "Laurel"]
_NAME_SUFFIX = {"residential": "Residences", "office": "Tower", "restaurant": "Kitchen", "bar": "Tavern",
                "gym": "Fitness", "grocery": "Market", "mall": "Mall", "hospital": "Medical Center",
                "university": "University", "stadium": "Stadium", "train_station": "Station", "airport": "Airport"}
_HOODS = ["Eastgate", "Northeast Hills", "Northside", "Northwest Park", "Westwood", "Southwest Flats", "Southpark",
          "Harbor District"]


@dataclasses.dataclass(frozen=True)
class SimConfig:
    n_riders: int = 5000
    n_homes: int = 1000
    days: int = 56
    start: str = "2026-08-31"   # a Monday
    seed: int = 7


def _neighborhood(x: float, y: float) -> str:
    dx, dy = x - CENTER[0], y - CENTER[1]
    if np.hypot(dx, dy) < 3.0:
        return "Downtown"
    sector = int(((np.degrees(np.arctan2(dy, dx)) + 360 + 22.5) % 360) // 45)
    return _HOODS[sector]


def make_places(cfg: SimConfig, rng: np.random.Generator) -> pd.DataFrame:
    rows: list[tuple[str, float, float]] = []
    for _ in range(cfg.n_homes):
        x, y = np.clip(rng.normal(CENTER, 5.0), 0.3, CITY_KM - 0.3)
        rows.append(("residential", x, y))
    for category, (count, spread) in PUBLIC_PLACES.items():
        for _ in range(count):
            x, y = np.clip(rng.normal(CENTER, spread), 0.3, CITY_KM - 0.3)
            rows.append((category, x, y))
    rows += [("airport", x, y) for x, y in AIRPORTS]

    places = pd.DataFrame(rows, columns=["category", "x", "y"])
    places.insert(0, "place_id", np.arange(1, len(places) + 1))          # 0 is padding in the models
    places["neighborhood"] = [_neighborhood(x, y) for x, y in zip(places.x, places.y)]
    seen: dict[str, int] = {}
    names = []
    for category, hood in zip(places.category, places.neighborhood):
        base = f"{_NAME_PREFIX[rng.integers(len(_NAME_PREFIX))]} {_NAME_SUFFIX[category]}"
        name = f"{base} · {hood}"
        seen[name] = seen.get(name, 0) + 1
        names.append(name if seen[name] == 1 else f"{base} {seen[name]} · {hood}")
    places["name"] = names
    # how much a place attracts unplanned visits; homes attract none
    places["appeal"] = np.where(places.category == "residential", 0.0, rng.lognormal(0.0, 1.0, len(places)))
    return places


def _pick(rng: np.random.Generator, ids: np.ndarray, weights: np.ndarray, k: int = 1) -> np.ndarray:
    p = weights / weights.sum()
    return rng.choice(ids, size=k, replace=False, p=p)


def make_riders(cfg: SimConfig, places: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    xy = places[["x", "y"]].to_numpy()
    by_cat = {c: places.index[places.category == c].to_numpy() for c in CATEGORIES}
    homes = by_cat["residential"]

    persona_ix = rng.choice(len(PERSONAS), size=cfg.n_riders, p=PERSONA_SHARE)
    gym_rate = {"commuter": 0.4, "student": 0.5, "nightlife": 0.3, "business": 0.5, "family": 0.2}
    ride_rate = {"commuter": 0.45, "student": 0.35, "nightlife": 0.6, "business": 0.7, "family": 0.3}
    price_sens = {"commuter": 1.2, "student": 2.0, "nightlife": 1.0, "business": 0.4, "family": 1.3}
    time_sens = {"commuter": 1.0, "student": 0.4, "nightlife": 0.7, "business": 1.6, "family": 0.9}

    rows = []
    for rider_id, pi in enumerate(persona_ix):
        persona = PERSONAS[pi]
        home = rng.choice(homes)
        d = np.hypot(*(xy - xy[home]).T)

        def near(cat: str, scale: float, k: int = 1, by_appeal: bool = False) -> np.ndarray:
            ids = by_cat[cat]
            w = np.exp(-d[ids] / scale) * (places.appeal.to_numpy()[ids] if by_appeal else 1.0)
            return _pick(rng, ids, w + 1e-12, k)

        work = near("university", 6.0)[0] if persona == "student" else near("office", 6.0)[0]
        gym = near("gym", 2.0)[0] if rng.random() < gym_rate[persona] else -1
        restaurants = near("restaurant", 4.0, 3, by_appeal=True)
        bars = near("bar", 8.0, 2, by_appeal=True)
        rows.append({
            "rider_id": rider_id,
            "persona": persona,
            "home_ix": home,
            "work_ix": work,
            "gym_ix": gym,
            "restaurant_ix": restaurants.tolist(),
            "bar_ix": bars.tolist(),
            "grocery_ix": near("grocery", 1.5)[0],
            "mall_ix": near("mall", 5.0)[0],
            "ride_rate": min(0.95, ride_rate[persona] * rng.lognormal(0, 0.25)),
            "price_sens": price_sens[persona] * rng.lognormal(0, 0.3),
            "time_sens": time_sens[persona] * rng.lognormal(0, 0.3),
            "mode_taste": rng.normal(0, 0.6, len(RIDE_MODES)).tolist(),
        })
    riders = pd.DataFrame(rows)
    pid = places.place_id.to_numpy()
    riders["home_place_id"] = pid[riders.home_ix]
    riders["work_place_id"] = pid[riders.work_ix]
    return riders


def _itinerary(r: dict, dow: int, airport_ix: int, explore: np.ndarray, explore_p: np.ndarray,
               rng: np.random.Generator) -> tuple[int, list[tuple[float, int]]]:
    """One day of movement as (hour, place index) legs. Hours past 24 belong to the night after."""
    weekend = dow >= 5
    home, work, gym = r["home_ix"], r["work_ix"], r["gym_ix"]
    restaurant = r["restaurant_ix"][rng.choice(3, p=[0.6, 0.3, 0.1])]
    bar = r["bar_ix"][rng.choice(2, p=[0.7, 0.3])]
    persona = r["persona"]
    legs: list[tuple[float, int]] = []

    def outing(t: float, place: int, stay: float) -> None:
        legs.extend([(t, place), (t + stay, home)])

    if persona == "business" and not weekend:
        u = rng.random()
        if u < 0.05:
            return home, [(rng.normal(6.0, 0.7), airport_ix)]
        if u < 0.10:
            return airport_ix, [(rng.normal(19.5, 1.5), home)]
    if persona in ("commuter", "business") and not weekend:
        legs.append((rng.normal(8.0, 0.6), work))
        if rng.random() < 0.25:
            legs.extend([(rng.normal(12.2, 0.3), restaurant), (13.3, work)])
        t = rng.normal(17.6, 0.7)
        if gym >= 0 and rng.random() < 0.45:
            outing(t, gym, 1.5)
        elif rng.random() < 0.2:
            outing(t, restaurant, 1.8)
        else:
            legs.append((t, home))
    elif persona == "student" and not weekend:
        legs.append((rng.normal(9.5, 0.8), work))
        t = rng.normal(15.5, 1.2)
        if gym >= 0 and rng.random() < 0.35:
            outing(t, gym, 1.5)
        else:
            legs.append((t, home))
        if dow in (3, 4) and rng.random() < 0.4:
            outing(rng.normal(22.0, 0.8), bar, 3.5)
    elif persona == "nightlife" and not weekend and rng.random() < 0.7:
        legs.extend([(rng.normal(9.8, 1.0), work), (rng.normal(18.2, 1.0), home)])
    elif persona == "family" and not weekend:
        if rng.random() < 0.6:
            legs.extend([(rng.normal(8.3, 0.5), work), (rng.normal(17.3, 0.6), home)])
        if rng.random() < 0.2:
            outing(rng.normal(18.6, 0.4), r["grocery_ix"], 0.8)

    if weekend:
        fam = persona == "family"
        if rng.random() < (0.7 if fam else 0.45):
            outing(rng.normal(10.5, 0.8), r["grocery_ix"], 1.0)
        if rng.random() < (0.4 if fam else 0.2):
            outing(rng.normal(14.0, 0.8), r["mall_ix"], 2.0)
        if dow == 6 and rng.random() < 0.06:
            outing(rng.normal(12.5, 0.4), STADIUM, 3.5)
        if rng.random() < 0.35:
            outing(rng.normal(18.5, 0.7), restaurant, 1.8)
    if persona == "nightlife" and (dow in (4, 5) or rng.random() < 0.15):
        t = rng.normal(20.5, 0.8)
        legs.extend([(t, restaurant), (t + 2.0, bar), (t + 5.0 + rng.normal(0, 0.5), home)])

    # occasional unplanned trip somewhere appealing
    last = max((t for t, _ in legs), default=8.0)
    if rng.random() < 0.12 and last < 20.5:
        t = rng.uniform(max(10.0, last + 0.5), 21.5)
        outing(t, rng.choice(explore, p=explore_p), 1.2)

    legs.sort(key=lambda leg: leg[0])
    return home, legs


def make_trips(cfg: SimConfig, places: pd.DataFrame, riders: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    airport_ix = int(places.index[places.category == "airport"][0])
    stadiums = places.index[places.category == "stadium"].to_numpy()
    explore = places.index[places.appeal > 0].to_numpy()
    explore_p = places.appeal.to_numpy()[explore] / places.appeal.to_numpy()[explore].sum()
    start = pd.Timestamp(cfg.start)
    rider_dicts = riders.to_dict("records")

    out_rider, out_day, out_hour, out_origin, out_dest = [], [], [], [], []
    for day in range(cfg.days):
        dow = (start + pd.Timedelta(days=day)).day_of_week
        for r in rider_dicts:
            loc, legs = _itinerary(r, dow, airport_ix, explore, explore_p, rng)
            for t, dest in legs:
                if dest == STADIUM:                              # Sunday game
                    dest = int(rng.choice(stadiums))
                if dest == loc or t < 0:
                    loc = dest
                    continue
                late = t >= 21.0 or t < 5.0
                p = 0.9 if airport_ix in (loc, dest) else (0.85 if late else r["ride_rate"])
                if rng.random() < p:
                    out_rider.append(r["rider_id"])
                    out_day.append(day)
                    out_hour.append(t)
                    out_origin.append(loc)
                    out_dest.append(dest)
                loc = dest

    pid = places.place_id.to_numpy()
    hours = np.array(out_hour)
    trips = pd.DataFrame({
        "rider_id": np.array(out_rider, dtype=np.int64),
        "ts": start + pd.to_timedelta(np.array(out_day), unit="D") + pd.to_timedelta(np.clip(hours, 0, 47.9), unit="h"),
        "origin_place_id": pid[out_origin],
        "dest_place_id": pid[out_dest],
    })
    trips["ts"] = trips.ts.dt.floor("min")
    trips = trips.sort_values(["ts", "rider_id"], kind="stable").reset_index(drop=True)
    trips.insert(0, "trip_id", np.arange(len(trips)))
    trips["day"] = (trips.ts - start).dt.days
    trips["hour"] = trips.ts.dt.hour
    trips["dow"] = trips.ts.dt.day_of_week
    return _add_ride_modes(trips, places, riders, rng)


def surge_for(hour: np.ndarray, dow: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """Demand multiplier by time; with rng, adds the trip-level noise the simulator samples (serving quotes the
    expected surge)."""
    hour, dow = np.asarray(hour), np.asarray(dow)
    weekday = dow < 5
    peak = weekday & (((hour >= 7) & (hour <= 9)) | ((hour >= 17) & (hour <= 19)))
    night = ((dow == 4) | (dow == 5)) & (hour >= 22) | ((dow == 5) | (dow == 6)) & (hour <= 2)
    noise = rng.lognormal(0, 0.12, hour.shape) if rng is not None else 1.0
    surge = (1.0 + 0.3 * peak + 0.5 * night) * noise
    return np.round(np.clip(surge, 1.0, 3.0) * 20) / 20


def standard_fare(distance_km: np.ndarray, surge: np.ndarray) -> np.ndarray:
    return np.round((2.5 + 1.6 * distance_km) * surge, 2)


def _add_ride_modes(trips: pd.DataFrame, places: pd.DataFrame, riders: pd.DataFrame,
                    rng: np.random.Generator) -> pd.DataFrame:
    xy = places.set_index("place_id")[["x", "y"]]
    o, d = xy.loc[trips.origin_place_id].to_numpy(), xy.loc[trips.dest_place_id].to_numpy()
    trips["distance_km"] = np.round(np.hypot(*(o - d).T) * 1.3 + 0.3, 2)
    trips["surge"] = surge_for(trips.hour.to_numpy(), trips.dow.to_numpy(), rng)

    cat = places.set_index("place_id").category
    airport = (cat.loc[trips.origin_place_id].to_numpy() == "airport") | (cat.loc[trips.dest_place_id].to_numpy() == "airport")
    rd = riders.set_index("rider_id")
    persona = rd.persona.loc[trips.rider_id].to_numpy()
    late = (trips.hour >= 21) | (trips.hour < 5)

    n = len(trips)
    group = np.where(persona == "family", rng.choice([1, 2, 3, 4, 5, 6], n, p=[.25, .2, .2, .2, .1, .05]),
                     np.where(late, rng.choice([1, 2, 3, 4], n, p=[.3, .35, .2, .15]),
                              rng.choice([1, 2, 3, 4], n, p=[.75, .18, .05, .02])))
    luggage = np.where(airport, rng.random(n) < 0.85, rng.random(n) < 0.03)
    trips["group_size"] = group
    trips["luggage"] = luggage.astype(np.int8)
    trips["is_airport"] = airport.astype(np.int8)

    fare = standard_fare(trips.distance_km.to_numpy(), trips.surge.to_numpy())
    prices = np.round(fare[:, None] * MODE_PRICE[None, :], 2)
    for j, m in enumerate(RIDE_MODES):
        trips[f"price_{m}"] = prices[:, j]

    price_sens = rd.price_sens.loc[trips.rider_id].to_numpy()
    time_sens = rd.time_sens.loc[trips.rider_id].to_numpy()
    taste = np.stack(rd.mode_taste.loc[trips.rider_id].to_numpy())
    u = -price_sens[:, None] * np.log(prices) + taste
    u[:, 0] += -0.9 * time_sens - 0.6 * luggage
    u[:, 1] += 0.5
    u[:, 2] += 1.8 * (group >= 4) + 0.3 * (group == 3) + 1.0 * luggage - 0.4
    u[:, 3] += 1.6 * (persona == "business") + 0.6 * (late & (persona == "nightlife")) - 0.8
    u[group[:, None] > MODE_CAPACITY[None, :]] = -np.inf
    trips["mode"] = np.array(RIDE_MODES)[np.argmax(u + rng.gumbel(size=u.shape), axis=1)]
    return trips


def simulate(cfg: SimConfig = SimConfig()) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(cfg.seed)
    places = make_places(cfg, rng)
    riders = make_riders(cfg, places, rng)
    trips = make_trips(cfg, places, riders, rng)
    # keep only what a production rider profile would have (segment + saved places); the latent tastes stay hidden
    riders = riders[["rider_id", "persona", "home_place_id", "work_place_id"]]
    return places, riders, trips


def load_or_simulate(data_dir: Path, cfg: SimConfig = SimConfig()) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    files = [data_dir / f"{name}.parquet" for name in ("places", "riders", "trips")]
    if all(f.exists() for f in files):
        return tuple(pd.read_parquet(f) for f in files)  # type: ignore[return-value]
    data_dir.mkdir(parents=True, exist_ok=True)
    tables = simulate(cfg)
    for table, f in zip(tables, files):
        table.to_parquet(f, index=False)
    return tables
