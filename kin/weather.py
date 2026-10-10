"""Weather from structured services, never from web snippets.

Web search once told Kin it was "sunny, 29 degrees Fahrenheit" in California in September: a Celsius
figure with the wrong unit. Someone decides how to dress from this, so the numbers come from APIs
with explicit units instead.

Two sources, both free and keyless:
  - "Now" comes from the nearest National Weather Service station when there is one (US only): a
    thermometer, not a model. Oct 9: Open-Meteo's model said 74 F for San Jose while the house read 80.
  - The forecast (today, tomorrow) comes from Open-Meteo, which also gives "now" where no station is near.
"""

from datetime import datetime, timedelta, timezone
from functools import lru_cache

import httpx

from kin import config

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
NWS_POINTS_URL = "https://api.weather.gov/points/{latitude:.4f},{longitude:.4f}"
NWS_LATEST_URL = "https://api.weather.gov/stations/{station}/observations/latest"
NWS_HEADERS = {"User-Agent": "Kin companion robot (github.com/UBayouski/Kin)", "Accept": "application/geo+json"}
OBSERVATION_MAX_AGE = timedelta(minutes=90)  # a station that has gone quiet is no better than the model

_WMO = {
    0: "clear sky", 1: "mostly clear", 2: "partly cloudy", 3: "overcast", 45: "fog", 48: "freezing fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 56: "freezing drizzle", 57: "freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain", 66: "freezing rain", 67: "freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains", 80: "light showers", 81: "showers",
    82: "heavy showers", 85: "snow showers", 86: "heavy snow showers", 95: "thunderstorm",
    96: "thunderstorm with hail", 99: "thunderstorm with hail",
}


@lru_cache(maxsize=16)
def _locate(place: str) -> tuple[float, float, str]:
    """Latitude, longitude and a display name. 'San Jose, California' is matched on the city, then the region."""
    parts = [p.strip() for p in place.split(",") if p.strip()]
    response = httpx.get(GEOCODE_URL, params={"name": parts[0], "count": 10}, timeout=15)
    response.raise_for_status()
    results = response.json().get("results") or []
    if not results:
        raise ValueError(f"no place called {place!r} found")
    hints = [p.lower() for p in parts[1:]]
    def score(r: dict) -> int:
        fields = " ".join(str(r.get(k, "")) for k in ("admin1", "admin2", "country", "country_code")).lower()
        return sum(h in fields for h in hints)
    best = max(results, key=score)  # max keeps the first, most populous, result on a tie
    name = ", ".join(x for x in (best.get("name"), best.get("admin1"), best.get("country")) if x)
    return best["latitude"], best["longitude"], name


# --- the station ----------------------------------------------------------------------------------------------------

_stations: dict[tuple[float, float], str] = {}  # only successes are kept; a failed lookup is tried again next time


def _station(latitude: float, longitude: float) -> str | None:
    """The nearest NWS station's identifier, or None outside the US or when the service does not answer."""
    key = (round(latitude, 4), round(longitude, 4))
    if key in _stations:
        return _stations[key]
    points = httpx.get(NWS_POINTS_URL.format(latitude=latitude, longitude=longitude), headers=NWS_HEADERS, timeout=10)
    if points.status_code != 200:  # 404 is how the NWS says "not in the United States"
        return None
    stations = httpx.get(points.json()["properties"]["observationStations"], headers=NWS_HEADERS, timeout=10)
    stations.raise_for_status()
    features = stations.json().get("features") or []
    if not features:
        return None
    _stations[key] = features[0]["properties"]["stationIdentifier"]  # the list is ordered nearest first
    return _stations[key]


def observation_line(properties: dict, imperial: bool, now: datetime | None = None) -> str | None:
    """'Now: ...' from one NWS observation, or None when it is stale or has no temperature. Pure."""
    temperature = (properties.get("temperature") or {}).get("value")
    if temperature is None:
        return None
    taken = datetime.fromisoformat(properties["timestamp"])
    age = (now or datetime.now(timezone.utc)) - taken
    if age > OBSERVATION_MAX_AGE:
        return None
    deg, speed = ("degrees Fahrenheit", "miles per hour") if imperial else ("degrees Celsius", "kilometres per hour")
    if imperial:
        temperature = temperature * 9 / 5 + 32
    parts = [f"{temperature:.0f} {deg}"]
    if sky := (properties.get("textDescription") or "").strip():
        parts.append(sky.lower())
    wind = (properties.get("windSpeed") or {}).get("value")
    if wind is not None:
        parts.append(f"wind {wind / 1.609344 if imperial else wind:.0f} {speed}")
    minutes = max(1, int(age.total_seconds() // 60))
    return f"Now: {', '.join(parts)} (measured by a weather station {minutes} minutes ago)."


def _observed_now(latitude: float, longitude: float, imperial: bool) -> str | None:
    try:
        station = _station(latitude, longitude)
        if not station:
            return None
        response = httpx.get(NWS_LATEST_URL.format(station=station), headers=NWS_HEADERS, timeout=10)
        response.raise_for_status()
        return observation_line(response.json()["properties"], imperial)
    except Exception as e:  # the forecast model's "now" stands in; never let the station cost the answer
        print(f"  (no station reading: {type(e).__name__}: {str(e)[:60]}; using the forecast model for now)", flush=True)
        return None


# --- what to wear ---------------------------------------------------------------------------------------------------


def dressing(high: float, low: float, rain_chance: float, wind: float, imperial: bool) -> str:
    """One practical line for the day, from the forecast numbers. Pure, and the same every time, so the
    advice is regular instead of left to the planner's mood (Oct 9: a check-in gave the forecast and no
    word on clothes). Thresholds are in Fahrenheit and miles per hour; metric input is converted."""
    if not imperial:
        high, low, wind = high * 9 / 5 + 32, low * 9 / 5 + 32, wind / 1.609344
    if high >= 80:
        wear = "light clothes, and a hat if out in the sun"
    elif high >= 70:
        wear = "light clothes"
    elif high >= 60:
        wear = "a light jacket or cardigan"
    elif high >= 50:
        wear = "a warm jacket"
    else:
        wear = "a warm coat, with a hat and gloves"
    parts = [wear]
    if high >= 70 and low < 60:
        parts.append("a layer to take off, as mornings and evenings are much cooler")
    if wind >= 20:
        parts.append("something that closes at the front, as it is windy")
    if rain_chance >= 30:
        parts.append("an umbrella")
    return "To wear: " + ", and ".join(parts) + "."


# --- the report -----------------------------------------------------------------------------------------------------


def report(place: str = "") -> str:
    """Current conditions plus today and tomorrow, as plain text with explicit units."""
    place = place.strip() or config.home_location()
    if not place:
        return "error: no place given and KIN_LOCATION is not set"
    latitude, longitude, name = _locate(place)
    imperial = config.units() == "imperial"
    response = httpx.get(FORECAST_URL, timeout=15, params={
        "latitude": latitude, "longitude": longitude, "timezone": "auto", "forecast_days": 2,
        "current": "temperature_2m,apparent_temperature,weather_code,wind_speed_10m",
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
        "temperature_unit": "fahrenheit" if imperial else "celsius",
        "wind_speed_unit": "mph" if imperial else "kmh",
    })
    response.raise_for_status()
    data = response.json()
    deg, speed = ("degrees Fahrenheit", "miles per hour") if imperial else ("degrees Celsius", "kilometres per hour")
    now, daily = data["current"], data["daily"]

    def sky(code) -> str:
        return _WMO.get(int(code), "mixed conditions")

    measured = _observed_now(latitude, longitude, imperial)
    lines = [f"{name}. " + (measured or (
        f"Now: {now['temperature_2m']:.0f} {deg}, {sky(now['weather_code'])}, "
        f"feels like {now['apparent_temperature']:.0f}, wind {now['wind_speed_10m']:.0f} {speed}."))]
    for i, label in enumerate(("Today", "Tomorrow")):
        rain = daily["precipitation_probability_max"][i]
        lines.append(
            f"{label}: {sky(daily['weather_code'][i])}, high {daily['temperature_2m_max'][i]:.0f}, "
            f"low {daily['temperature_2m_min'][i]:.0f} {deg}, "
            f"{'no rain expected' if not rain else f'{rain:.0f} percent chance of rain'}."
        )
    lines.append(dressing(daily["temperature_2m_max"][0], daily["temperature_2m_min"][0],
                          daily["precipitation_probability_max"][0] or 0, now["wind_speed_10m"], imperial))
    return "\n".join(lines)
