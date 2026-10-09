"""Weather: a measured 'now' from the nearest NWS station, the forecast from Open-Meteo, explicit units throughout."""
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from kin import weather

NOW = datetime(2026, 10, 9, 22, 45, tzinfo=timezone.utc)
OBSERVATION = {
    "timestamp": "2026-10-09T22:25:00+00:00", "textDescription": "Clear",
    "temperature": {"unitCode": "wmoUnit:degC", "value": 23}, "windSpeed": {"unitCode": "wmoUnit:km_h-1", "value": 20.376},
}


def test_a_station_reading_is_converted_with_explicit_units():
    line = weather.observation_line(OBSERVATION, imperial=True, now=NOW)
    assert line == "Now: 73 degrees Fahrenheit, clear, wind 13 miles per hour (measured by a weather station 20 minutes ago)."
    assert weather.observation_line(OBSERVATION, imperial=False, now=NOW).startswith("Now: 23 degrees Celsius, clear, wind 20 kilometres per hour")


def test_a_stale_or_empty_reading_is_not_used():
    old = datetime(2026, 10, 10, 1, 0, tzinfo=timezone.utc)  # two and a half hours later
    assert weather.observation_line(OBSERVATION, imperial=True, now=old) is None, "a station gone quiet is no better than the model"
    assert weather.observation_line({**OBSERVATION, "temperature": {"value": None}}, imperial=True, now=NOW) is None
    missing_wind = weather.observation_line({**OBSERVATION, "windSpeed": {"value": None}, "textDescription": ""}, imperial=True, now=NOW)
    assert missing_wind.startswith("Now: 73 degrees Fahrenheit (measured")


def _fake_get(answers):
    """httpx.get standing in for every service: the answer is picked by what the URL starts with."""
    def get(url, **kwargs):
        for prefix, (status, body) in answers.items():
            if url.startswith(prefix):
                return SimpleNamespace(status_code=status, json=lambda: body, raise_for_status=lambda: None)
        raise AssertionError(f"unexpected request {url}")
    return get


FORECAST = {
    "current": {"temperature_2m": 74.3, "apparent_temperature": 73.1, "weather_code": 0, "wind_speed_10m": 8.2},
    "daily": {"weather_code": [45, 2], "temperature_2m_max": [74.0, 71.2], "temperature_2m_min": [58.1, 54.0],
              "precipitation_probability_max": [2, 0]},
}


@pytest.fixture
def home(monkeypatch):
    monkeypatch.setenv("KIN_LOCATION", "San Jose, California")
    monkeypatch.setenv("KIN_UNITS", "imperial")
    monkeypatch.setattr(weather, "_locate", lambda place: (37.3394, -121.895, "San Jose, California, United States"))
    weather._stations.clear()


def test_now_comes_from_the_station_and_the_forecast_from_the_model(home, monkeypatch):
    """Oct 9: the model said 74 F for the city centre while the house read 80; a thermometer is the better
    thing to quote, and it says so in the text."""
    monkeypatch.setattr(weather.httpx, "get", _fake_get({
        weather.FORECAST_URL: (200, FORECAST),
        "https://api.weather.gov/points/": (200, {"properties": {"observationStations": "https://api.weather.gov/gridpoints/MTR/99,82/stations"}}),
        "https://api.weather.gov/gridpoints/": (200, {"features": [{"properties": {"stationIdentifier": "KSJC", "name": "San Jose International Airport"}}]}),
        "https://api.weather.gov/stations/KSJC/": (200, {"properties": {**OBSERVATION, "timestamp": datetime.now(timezone.utc).isoformat()}}),
    }))

    text = weather.report()

    assert text.startswith("San Jose, California, United States. Now: 73 degrees Fahrenheit, clear, wind 13 miles per hour (measured by a weather station 1 minutes ago).")
    assert "Today: fog, high 74, low 58 degrees Fahrenheit, 2 percent chance of rain." in text
    assert "Tomorrow: partly cloudy, high 71, low 54 degrees Fahrenheit, no rain expected." in text
    assert weather._stations, "the station is remembered, so the next report is one request shorter"


def test_outside_the_us_or_with_the_station_down_the_model_still_answers(home, monkeypatch):
    monkeypatch.setattr(weather.httpx, "get", _fake_get({
        weather.FORECAST_URL: (200, FORECAST),
        "https://api.weather.gov/points/": (404, {"title": "Data Unavailable For Requested Point"}),
    }))
    text = weather.report()
    assert "Now: 74 degrees Fahrenheit, clear sky, feels like 73, wind 8 miles per hour." in text
    assert "measured" not in text and not weather._stations, "a miss is not remembered as a station"

    def broken(url, **kwargs):
        if url.startswith(weather.FORECAST_URL):
            return SimpleNamespace(status_code=200, json=lambda: FORECAST, raise_for_status=lambda: None)
        raise ConnectionError("weather.gov is down")
    monkeypatch.setattr(weather.httpx, "get", broken)
    assert "Now: 74 degrees Fahrenheit" in weather.report(), "the station failing never costs the answer"
