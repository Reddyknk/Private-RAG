#!/usr/bin/env python3
"""
env_tools.py - Environment & Weather Tool for Autonomous Agents.

Uses the free, public Open-Meteo API (no API key required) to retrieve:
1. Geocoding coordinates and timezone for any city worldwide.
2. Current local time formatted according to the city's timezone.
3. Current weather metrics: temperature, apparent temperature, weather condition,
   relative humidity, precipitation, and wind speed.
"""

import sys
import json
import argparse
import urllib.request
import urllib.parse
from datetime import datetime
from typing import Dict, Any, Optional

# WMO Weather interpretation codes (WW) defined by World Meteorological Organization
WMO_WEATHER_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm (slight or moderate)",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail"
}


def geocode_city(city_name: str) -> Optional[Dict[str, Any]]:
    """
    Search Open-Meteo Geocoding API for city coordinates and timezone.
    Does not require any API key.
    """
    encoded_name = urllib.parse.quote(city_name.strip())
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={encoded_name}&count=1&language=en&format=json"
    req = urllib.request.Request(url, headers={"User-Agent": "PrivateRAG-Agent/1.0"})
    
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            results = data.get("results")
            if not results:
                return None
            first = results[0]
            return {
                "_request_url": url,
                "name": first.get("name"),
                "country": first.get("country", ""),
                "admin1": first.get("admin1", ""),
                "latitude": first.get("latitude"),
                "longitude": first.get("longitude"),
                "timezone": first.get("timezone", "UTC"),
                "elevation": first.get("elevation")
            }
    except Exception as e:
        sys.stderr.write(f"Geocoding error for '{city_name}': {e}\n")
        return None


def get_city_weather_and_time(city_name: str, unit: str = "celsius") -> Dict[str, Any]:
    """
    Retrieve current time and weather conditions for a specified city using Open-Meteo.
    unit can be 'celsius' or 'fahrenheit'.
    """
    geo = geocode_city(city_name)
    if not geo:
        return {
            "status": "error",
            "message": f"Could not find coordinates for city: '{city_name}'"
        }

    lat = geo["latitude"]
    lon = geo["longitude"]
    temp_param = "&temperature_unit=fahrenheit" if unit.lower() == "fahrenheit" else ""
    weather_url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
        f"&current=temperature_2m,relative_humidity_2m,apparent_temperature,is_day,precipitation,weather_code,wind_speed_10m"
        f"{temp_param}&timezone=auto"
    )

    req = urllib.request.Request(weather_url, headers={"User-Agent": "PrivateRAG-Agent/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            current = data.get("current", {})
            current_units = data.get("current_units", {})

            wcode = current.get("weather_code", 0)
            condition_desc = WMO_WEATHER_CODES.get(wcode, "Unknown condition")
            iso_time = current.get("time", "")

            formatted_time = iso_time
            if iso_time:
                try:
                    dt = datetime.fromisoformat(iso_time)
                    formatted_time = dt.strftime("%A, %B %d, %Y, %I:%M %p")
                except Exception:
                    formatted_time = iso_time

            temp_val = current.get("temperature_2m")
            temp_unit = current_units.get("temperature_2m", "°C")
            apparent_temp = current.get("apparent_temperature")
            humidity = current.get("relative_humidity_2m")
            wind_speed = current.get("wind_speed_10m")
            wind_unit = current_units.get("wind_speed_10m", "km/h")
            is_day = current.get("is_day", 1) == 1

            location_str = f"{geo['name']}"
            if geo.get("admin1"):
                location_str += f", {geo['admin1']}"
            if geo.get("country"):
                location_str += f", {geo['country']}"

            external_apis = [
                {
                    "name": "Open-Meteo Geocoding API",
                    "url": geo.get("_request_url", ""),
                    "method": "GET",
                    "status_code": 200,
                    "request": {
                        "method": "GET",
                        "url": geo.get("_request_url", ""),
                        "city": city_name
                    },
                    "response": {
                        "status_code": 200,
                        "name": geo["name"],
                        "latitude": lat,
                        "longitude": lon,
                        "timezone": geo["timezone"],
                        "country": geo.get("country", "")
                    }
                },
                {
                    "name": "Open-Meteo Forecast API",
                    "url": weather_url,
                    "method": "GET",
                    "status_code": 200,
                    "request": {
                        "method": "GET",
                        "url": weather_url,
                        "latitude": lat,
                        "longitude": lon,
                        "unit": unit
                    },
                    "response": {
                        "status_code": 200,
                        "temperature": f"{temp_val}{temp_unit}",
                        "condition": condition_desc,
                        "humidity": f"{humidity}%",
                        "wind_speed": f"{wind_speed} {wind_unit}"
                    }
                }
            ]

            return {
                "status": "success",
                "city": geo["name"],
                "country": geo.get("country", ""),
                "location": location_str,
                "timezone": geo["timezone"],
                "local_time": formatted_time,
                "iso_time": iso_time,
                "is_day": is_day,
                "weather": {
                    "condition": condition_desc,
                    "weather_code": wcode,
                    "temperature": f"{temp_val}{temp_unit}",
                    "temperature_numeric": temp_val,
                    "apparent_temperature": f"{apparent_temp}{temp_unit}",
                    "relative_humidity": f"{humidity}%",
                    "precipitation": f"{current.get('precipitation', 0)} mm",
                    "wind_speed": f"{wind_speed} {wind_unit}"
                },
                "external_apis": external_apis
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to retrieve weather data: {e}"
        }


def main():
    parser = argparse.ArgumentParser(description="Query local time and weather for any city without API keys.")
    parser.add_argument("city", nargs="?", default="Tokyo", help="Name of the city (e.g., 'Paris', 'New York', 'Tokyo')")
    parser.add_argument("--unit", choices=["celsius", "fahrenheit"], default="celsius", help="Temperature unit")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    result = get_city_weather_and_time(args.city, unit=args.unit)

    if args.json:
        print(json.dumps(result, indent=2))
        return

    if result.get("status") == "error":
        print(f"Error: {result.get('message')}")
        sys.exit(1)

    w = result["weather"]
    print(f"==================================================")
    print(f"🌍 Location:   {result['location']}")
    print(f"🕒 Local Time: {result['local_time']} ({result['timezone']})")
    print(f"⛅ Condition:  {w['condition']}")
    print(f"🌡️ Temp:       {w['temperature']} (Feels like {w['apparent_temperature']})")
    print(f"💧 Humidity:   {w['relative_humidity']}")
    print(f"💨 Wind Speed: {w['wind_speed']}")
    print(f"==================================================")


if __name__ == "__main__":
    main()
