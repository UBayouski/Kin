"""Weather from a forecast service (Open-Meteo: free, no key), not from web snippets.

Web search once told Kin it was "sunny, 29 degrees Fahrenheit" in California in September: a Celsius
figure with the wrong unit. Someone decides how to dress from this, so the numbers come from an API
with explicit units instead.
"""

from functools import lru_cache

import httpx

from kin import config

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

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

    lines = [
        f"{name}. Now: {now['temperature_2m']:.0f} {deg}, {sky(now['weather_code'])}, "
        f"feels like {now['apparent_temperature']:.0f}, wind {now['wind_speed_10m']:.0f} {speed}."
    ]
    for i, label in enumerate(("Today", "Tomorrow")):
        rain = daily["precipitation_probability_max"][i]
        lines.append(
            f"{label}: {sky(daily['weather_code'][i])}, high {daily['temperature_2m_max'][i]:.0f}, "
            f"low {daily['temperature_2m_min'][i]:.0f} {deg}, "
            f"{'no rain expected' if not rain else f'{rain:.0f} percent chance of rain'}."
        )
    return "\n".join(lines)
