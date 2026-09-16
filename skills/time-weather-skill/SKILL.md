---
name: time-weather-skill
description: Weather forecasts and current time for global cities.
---

# Time and Weather Skill

This skill allows the agent to look up the current local time, timezone, and weather forecast/conditions for any city worldwide using the public Open-Meteo API. No API key or registration is required.

## 🎯 Trigger Queries & Capabilities
Activate this skill when the user asks:
- "What is the time and weather in [City]?"
- "What is the current time in Tokyo / London / Paris?"
- "How is the weather right now in New York?"
- "Is it raining in Berlin?"
- "Give me the temperature and humidity for San Francisco."

---

## 🛠️ Public Endpoints Used (No API Key Required)

1. **Geocoding API (City to Latitude, Longitude & Timezone):**
   ```http
   GET https://geocoding-api.open-meteo.com/v1/search?name={city_name}&count=1&language=en&format=json
   ```
   - Parameters:
     - `name`: Name of the city (e.g., `Tokyo`, `London`, `San Francisco`)
     - `count`: `1` (top matching result)
     - `language`: `en`
     - `format`: `json`
   - Returns: `latitude`, `longitude`, `timezone` (e.g., `Asia/Tokyo`, `Europe/London`), `country`, `name`.

2. **Forecast & Current Weather API:**
   ```http
   GET https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,apparent_temperature,is_day,precipitation,weather_code,wind_speed_10m&timezone=auto
   ```
   - Parameters:
     - `latitude`: Float latitude from geocoding.
     - `longitude`: Float longitude from geocoding.
     - `current`: Comma-separated list of metrics (`temperature_2m`, `relative_humidity_2m`, `apparent_temperature`, `is_day`, `precipitation`, `weather_code`, `wind_speed_10m`).
     - `temperature_unit`: Optional `fahrenheit` (default is `celsius`).
     - `timezone`: `auto` (resolves to the local timezone of the coordinates).
   - Returns: Current metrics and `time` stamped in the city's local timezone.

---

## 📋 Standard Operating Procedure (SOP)

When answering questions about the time or weather of a city:
1. **Identify the City**: Extract the target city name and optional unit preference (Celsius vs Fahrenheit).
2. **Execute the Tool Script**:
   Run the bundled environment tool from the repository root:
   ```bash
   python skills/time-weather-skill/scripts/env_tools.py "<City>" [--unit fahrenheit] [--json]
   ```
3. **Format the Response**:
   Present the information clearly with:
   - **City & Country**: Identified location and administrative area.
   - **Current Local Time & Date**: Accurate local time including timezone (e.g., `Monday, September 14, 2026, 12:30 PM EDT`).
   - **Weather Condition**: Readable description (e.g., `Clear sky`, `Overcast`, `Light rain`).
   - **Temperature**: Current temperature and feels-like temperature.
   - **Atmospheric Metrics**: Relative humidity (%), precipitation (mm), and wind speed (km/h).

---

## 🌧️ WMO Weather Code Reference Table
- `0`: Clear sky
- `1, 2, 3`: Mainly clear, Partly cloudy, Overcast
- `45, 48`: Fog, Depositing rime fog
- `51, 53, 55`: Light, Moderate, Dense drizzle
- `61, 63, 65`: Slight, Moderate, Heavy rain
- `71, 73, 75`: Slight, Moderate, Heavy snow fall
- `80, 81, 82`: Rain showers
- `95, 96, 99`: Thunderstorm (with or without hail)

---

## 💻 Script CLI Reference

```bash
# Query default or specific city:
python skills/time-weather-skill/scripts/env_tools.py "Tokyo"

# Query in Fahrenheit:
python skills/time-weather-skill/scripts/env_tools.py "Chicago" --unit fahrenheit

# Machine-readable JSON output:
python skills/time-weather-skill/scripts/env_tools.py "Paris" --json
```
