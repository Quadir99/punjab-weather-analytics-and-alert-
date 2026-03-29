from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests


BASE_URL = "https://api.openweathermap.org/data/2.5"
DATA_DIR = Path("data")
HISTORY_FILE = DATA_DIR / "weather_history.csv"
GEOJSON_FILE = DATA_DIR / "punjab_districts.geojson"
TELEGRAM_STATE_FILE = DATA_DIR / "telegram_alert_state.json"
HISTORY_COLUMNS = [
    "Fetched_At",
    "City",
    "Crop_Focus",
    "Temp",
    "Humidity",
    "Pressure",
    "Visibility",
    "Wind_Speed",
    "Clouds",
    "Weather_Desc",
    "Lat",
    "Lon",
    "Forecast_Rain_Events",
    "Forecast_Min_Temp",
    "Forecast_Max_Temp",
    "Forecast_Min_Visibility",
    "Forecast_Note",
    "Risk_Score",
    "Alert_Band",
    "Smart_Alerts",
    "Advisory",
]

LOCATIONS = {
    "Ludhiana": {"lat": 30.9010, "lon": 75.8573, "crop_focus": "Wheat"},
    "Amritsar": {"lat": 31.6340, "lon": 74.8723, "crop_focus": "Wheat"},
    "Bathinda": {"lat": 30.2110, "lon": 74.9455, "crop_focus": "Cotton"},
    "Patiala": {"lat": 30.33, "lon": 76.40, "crop_focus": "Paddy"},
    "Jalandhar": {"lat": 31.33, "lon": 75.57, "crop_focus": "Maize"},
    "Mansa": {"lat": 29.99, "lon": 75.40, "crop_focus": "Cotton"},
    "Barnala": {"lat": 30.37, "lon": 75.55, "crop_focus": "Wheat"},
}

CROP_GUIDANCE = {
    "Wheat": {
        "heat_temp": 33,
        "message": "Protect grain filling by shifting irrigation to evening hours.",
    },
    "Paddy": {
        "heat_temp": 35,
        "message": "Watch water availability and avoid fertilizer application before rainfall.",
    },
    "Cotton": {
        "heat_temp": 36,
        "message": "Monitor leaf stress and postpone spraying during strong afternoon winds.",
    },
    "Maize": {
        "heat_temp": 34,
        "message": "Check tasseling-stage moisture demand and avoid moisture shock.",
    },
}


@dataclass
class ForecastSummary:
    rain_events: int
    min_temp: float | None
    max_temp: float | None
    min_visibility: float | None
    forecast_note: str


def format_visibility(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{int(value)} m"


def _safe_get(url: str, timeout: int = 20) -> dict:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()


def fetch_current_weather(api_key: str) -> pd.DataFrame:
    records: list[dict] = []

    for city, meta in LOCATIONS.items():
        url = (
            f"{BASE_URL}/weather?lat={meta['lat']}&lon={meta['lon']}"
            f"&appid={api_key}&units=metric"
        )
        payload = _safe_get(url)
        weather_desc = payload.get("weather", [{}])[0].get("description", "")
        records.append(
            {
                "City": city,
                "Crop_Focus": meta["crop_focus"],
                "Temp": payload.get("main", {}).get("temp"),
                "Humidity": payload.get("main", {}).get("humidity"),
                "Pressure": payload.get("main", {}).get("pressure"),
                "Visibility": payload.get("visibility"),
                "Wind_Speed": payload.get("wind", {}).get("speed"),
                "Clouds": payload.get("clouds", {}).get("all"),
                "Weather_Desc": weather_desc,
                "Lat": meta["lat"],
                "Lon": meta["lon"],
            }
        )

    return pd.DataFrame(records)


def summarize_forecast(api_key: str, city: str, lat: float, lon: float) -> ForecastSummary:
    url = (
        f"{BASE_URL}/forecast?lat={lat}&lon={lon}"
        f"&appid={api_key}&units=metric"
    )
    payload = _safe_get(url)
    items = payload.get("list", [])
    if not items:
        return ForecastSummary(0, None, None, None, "Forecast unavailable")

    temps = [item.get("main", {}).get("temp") for item in items if item.get("main")]
    temps = [temp for temp in temps if temp is not None]
    visibilities = [item.get("visibility") for item in items if item.get("visibility") is not None]
    rain_events = 0

    for item in items[:8]:
        description = item.get("weather", [{}])[0].get("description", "").lower()
        rain_volume = item.get("rain", {}).get("3h", 0)
        if "rain" in description or rain_volume:
            rain_events += 1

    max_temp = max(temps) if temps else None
    min_temp = min(temps) if temps else None
    min_visibility = min(visibilities) if visibilities else None

    if rain_events >= 3:
        note = f"{city} may see recurring rainfall in the next 24 hours."
    elif max_temp is not None and max_temp >= 36:
        note = f"{city} shows a high heat build-up in the next 24 hours."
    elif min_visibility is not None and min_visibility < 1000:
        note = f"{city} may face poor visibility in the next 24 hours."
    else:
        note = f"{city} has relatively stable conditions in the next 24 hours."

    return ForecastSummary(rain_events, min_temp, max_temp, min_visibility, note)


def build_enriched_dataset(current_df: pd.DataFrame, api_key: str) -> pd.DataFrame:
    enriched_rows: list[dict] = []

    for _, row in current_df.iterrows():
        forecast = summarize_forecast(api_key, row["City"], row["Lat"], row["Lon"])
        risk_score, alert_band, smart_alerts, advisory = assess_conditions(
            temp=row["Temp"],
            pressure=row["Pressure"],
            visibility=row["Visibility"],
            humidity=row["Humidity"],
            weather_desc=row["Weather_Desc"],
            wind_speed=row["Wind_Speed"],
            crop_focus=row["Crop_Focus"],
            forecast=forecast,
        )
        enriched_rows.append(
            {
                **row.to_dict(),
                "Forecast_Rain_Events": forecast.rain_events,
                "Forecast_Min_Temp": forecast.min_temp,
                "Forecast_Max_Temp": forecast.max_temp,
                "Forecast_Min_Visibility": forecast.min_visibility,
                "Forecast_Note": forecast.forecast_note,
                "Risk_Score": risk_score,
                "Alert_Band": alert_band,
                "Smart_Alerts": smart_alerts,
                "Advisory": advisory,
                "Crop_Recommendation": generate_crop_recommendation(
                    crop_focus=row["Crop_Focus"],
                    temp=row["Temp"],
                    humidity=row["Humidity"],
                    visibility=row["Visibility"],
                    wind_speed=row["Wind_Speed"],
                    forecast=forecast,
                ),
            }
        )

    return pd.DataFrame(enriched_rows)


def assess_conditions(
    *,
    temp: float | None,
    pressure: float | None,
    visibility: float | None,
    humidity: float | None,
    weather_desc: str,
    wind_speed: float | None,
    crop_focus: str,
    forecast: ForecastSummary,
) -> tuple[int, str, str, str]:
    score = 10
    alerts: list[str] = []
    description = (weather_desc or "").lower()
    crop_profile = CROP_GUIDANCE.get(crop_focus, CROP_GUIDANCE["Wheat"])

    if temp is not None:
        temp_excess = temp - crop_profile["heat_temp"]
        if temp_excess > 0:
            score += min(30, int(round(temp_excess * 8)))
            alerts.append("Heat stress risk")
        elif temp >= crop_profile["heat_temp"] - 1:
            score += 6
            alerts.append("Near heat threshold")

    if forecast.max_temp is not None:
        forecast_excess = forecast.max_temp - crop_profile["heat_temp"]
        if forecast_excess > 0:
            score += min(20, int(round(forecast_excess * 4)))
            alerts.append("Forecast heat spike")
        elif forecast.max_temp >= crop_profile["heat_temp"] - 1:
            score += 4
            alerts.append("Forecast warming")

    if pressure is not None and pressure < 1008 and "rain" in description:
        score += 18
        alerts.append("Storm pressure signal")
    elif pressure is not None and pressure < 1010 and "rain" in description:
        score += 10
        alerts.append("Rain pressure signal")

    if visibility is not None:
        if visibility < 500:
            score += 20
            alerts.append("Severe low visibility")
        elif visibility < 1000:
            score += 14
            alerts.append("Low visibility")
        elif visibility < 4000:
            score += 8
            alerts.append("Reduced visibility")

    if humidity is not None:
        if humidity > 90:
            score += 12
            alerts.append("High humidity disease risk")
        elif humidity > 80:
            score += 8
            alerts.append("Elevated humidity")
        elif humidity < 25 and temp is not None and temp > crop_profile["heat_temp"]:
            score += 6
            alerts.append("Dry heat stress")

    if wind_speed is not None:
        if wind_speed > 10:
            score += 10
            alerts.append("High wind exposure")
        elif wind_speed > 7:
            score += 6
            alerts.append("Wind exposure")

    if forecast.rain_events >= 4:
        score += 12
        alerts.append("Rain buildup")
    elif forecast.rain_events >= 2:
        score += 6
        alerts.append("Possible rainfall window")

    if forecast.min_visibility is not None:
        if forecast.min_visibility < 1000:
            score += 8
            alerts.append("Forecast visibility drop")
        elif forecast.min_visibility < 4000:
            score += 4
            alerts.append("Forecast haze risk")

    if temp is not None and humidity is not None and temp > 34 and humidity > 70:
        score += 8
        alerts.append("Heat-humidity stress")

    if (
        temp is not None
        and forecast.max_temp is not None
        and forecast.max_temp - temp >= 3
    ):
        score += 6
        alerts.append("Rapid heat escalation")

    score = max(0, min(100, score))

    if score >= 70:
        band = "High"
    elif score >= 40:
        band = "Moderate"
    else:
        band = "Low"

    if not alerts:
        alerts_text = "Stable weather window"
    else:
        alerts_text = " | ".join(alerts)

    advisory_parts = [crop_profile["message"], forecast.forecast_note]
    if visibility is not None and visibility < 1000:
        advisory_parts.insert(0, "Transport and spraying should be delayed until visibility improves.")
    advisory = " ".join(advisory_parts)

    return score, band, alerts_text, advisory


def generate_crop_recommendation(
    *,
    crop_focus: str,
    temp: float | None,
    humidity: float | None,
    visibility: float | None,
    wind_speed: float | None,
    forecast: ForecastSummary,
) -> str:
    crop_profile = CROP_GUIDANCE.get(crop_focus, CROP_GUIDANCE["Wheat"])
    recommendations: list[str] = []

    if temp is not None and temp >= crop_profile["heat_temp"]:
        recommendations.append("Shift irrigation to late evening to reduce heat stress.")
    if forecast.max_temp is not None and forecast.max_temp >= crop_profile["heat_temp"] + 2:
        recommendations.append("Prepare for hotter conditions over the next 24 hours.")
    if visibility is not None and visibility < 1000:
        recommendations.append("Delay transport and spraying until visibility improves.")
    if wind_speed is not None and wind_speed > 7:
        recommendations.append("Avoid chemical spraying during strong winds.")
    if humidity is not None and humidity > 80:
        recommendations.append("Monitor fungal and leaf-disease pressure in humid fields.")
    if forecast.rain_events >= 2:
        recommendations.append("Hold fertilizer application if rainfall is likely soon.")

    if not recommendations:
        recommendations.append(crop_profile["message"])

    return " ".join(recommendations)


def persist_snapshot(df: pd.DataFrame) -> None:
    if df.empty:
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_df = df.copy()
    snapshot_df.insert(0, "Fetched_At", datetime.now(timezone.utc).isoformat())
    snapshot_df.to_csv(
        HISTORY_FILE,
        mode="a",
        index=False,
        header=not HISTORY_FILE.exists(),
    )


def load_history(limit: int = 250) -> pd.DataFrame:
    if not HISTORY_FILE.exists():
        return pd.DataFrame()

    history_df = _read_history_csv()
    if history_df.empty:
        return history_df

    history_df["Fetched_At"] = pd.to_datetime(history_df["Fetched_At"], errors="coerce")
    history_df = history_df.dropna(subset=["Fetched_At"])
    history_df = history_df.sort_values("Fetched_At")

    if limit > 0:
        history_df = history_df.groupby("City", group_keys=False).tail(limit)

    return history_df


def _read_history_csv() -> pd.DataFrame:
    rows: list[dict] = []

    with HISTORY_FILE.open("r", encoding="utf-8", newline="") as history_file:
        reader = csv.reader(history_file)
        header = next(reader, None)
        if not header:
            return pd.DataFrame(columns=HISTORY_COLUMNS)

        for raw_row in reader:
            if not raw_row:
                continue

            # Older and newer app versions wrote different column counts.
            trimmed_row = raw_row[: len(HISTORY_COLUMNS)]
            if len(trimmed_row) < len(HISTORY_COLUMNS):
                trimmed_row += [""] * (len(HISTORY_COLUMNS) - len(trimmed_row))

            rows.append(dict(zip(HISTORY_COLUMNS, trimmed_row)))

    history_df = pd.DataFrame(rows, columns=HISTORY_COLUMNS)
    numeric_columns = [
        "Temp",
        "Humidity",
        "Pressure",
        "Visibility",
        "Wind_Speed",
        "Clouds",
        "Lat",
        "Lon",
        "Forecast_Rain_Events",
        "Forecast_Min_Temp",
        "Forecast_Max_Temp",
        "Forecast_Min_Visibility",
        "Risk_Score",
    ]
    for column in numeric_columns:
        history_df[column] = pd.to_numeric(history_df[column], errors="coerce")

    return history_df


def get_city_trend(history_df: pd.DataFrame, city: str) -> pd.DataFrame:
    if history_df.empty:
        return pd.DataFrame()
    city_df = history_df[history_df["City"] == city].copy()
    return city_df.sort_values("Fetched_At")


def latest_summary_metrics(df: pd.DataFrame) -> dict[str, float | int | str]:
    if df.empty:
        return {
            "avg_temp": 0.0,
            "avg_risk": 0.0,
            "high_risk_count": 0,
            "top_city": "N/A",
        }

    top_row = df.sort_values("Risk_Score", ascending=False).iloc[0]
    return {
        "avg_temp": float(df["Temp"].mean()),
        "avg_risk": float(df["Risk_Score"].mean()),
        "high_risk_count": int((df["Alert_Band"] == "High").sum()),
        "top_city": str(top_row["City"]),
    }


def exportable_columns() -> Iterable[str]:
    return (
        "City",
        "Crop_Focus",
        "Temp",
        "Humidity",
        "Pressure",
        "Visibility",
        "Wind_Speed",
        "Weather_Desc",
        "Risk_Score",
        "Alert_Band",
        "Smart_Alerts",
        "Advisory",
        "Crop_Recommendation",
        "Forecast_Rain_Events",
        "Forecast_Max_Temp",
        "Forecast_Note",
        "Predicted_Risk_24h",
        "Yield_Protection_Index",
        "Prediction_Confidence",
        "Prediction_Note",
    )


def add_prediction_features(current_df: pd.DataFrame, history_df: pd.DataFrame) -> pd.DataFrame:
    if current_df.empty:
        return current_df

    enriched_rows: list[dict] = []
    for _, row in current_df.iterrows():
        city = row["City"]
        city_history = pd.DataFrame()
        if not history_df.empty:
            city_history = history_df[history_df["City"] == city].sort_values("Fetched_At")

        prediction = predict_city_outlook(row, city_history)
        enriched_rows.append({**row.to_dict(), **prediction})

    return pd.DataFrame(enriched_rows)


def get_telegram_candidates(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "Alert_Band" not in df.columns:
        return pd.DataFrame()
    return df[df["Alert_Band"].isin(["Moderate", "High"])].copy()


def build_telegram_message(row: pd.Series) -> str:
    return (
        "Punjab Weather Alert\n"
        f"District: {row['City']}\n"
        f"Alert Band: {row['Alert_Band']}\n"
        f"Risk Score: {row['Risk_Score']}/100\n"
        f"Temperature: {row['Temp']:.1f} C\n"
        f"Visibility: {format_visibility(row['Visibility'])}\n"
        f"Alerts: {row['Smart_Alerts']}\n"
        f"Advisory: {row['Advisory']}\n"
        f"Recommendation: {row['Crop_Recommendation']}"
    )


def send_telegram_alerts(df: pd.DataFrame, bot_token: str | None, chat_id: str | None) -> dict[str, object]:
    bot_token = (bot_token or "").strip()
    chat_id = (chat_id or "").strip()
    candidates = get_telegram_candidates(df)
    if candidates.empty:
        return {"sent": 0, "skipped": 0, "status": "No moderate/high alerts to send."}

    if not bot_token or not chat_id:
        return {"sent": 0, "skipped": len(candidates), "status": "Telegram bot settings are missing."}

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    previous_state = _load_telegram_state()
    current_state = previous_state.copy()
    sent = 0
    skipped = 0

    for _, row in candidates.iterrows():
        signature = _build_alert_signature(row)
        city = str(row["City"])
        if previous_state.get(city) == signature:
            skipped += 1
            continue

        message = build_telegram_message(row)
        response = requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            data={"chat_id": chat_id, "text": message},
            timeout=20,
        )
        response.raise_for_status()
        current_state[city] = signature
        sent += 1

    _save_telegram_state(current_state)
    status = f"Telegram alerts sent: {sent}, skipped duplicates: {skipped}."
    return {"sent": sent, "skipped": skipped, "status": status}


def send_telegram_test_message(bot_token: str | None, chat_id: str | None) -> str:
    bot_token = (bot_token or "").strip()
    chat_id = (chat_id or "").strip()
    if not bot_token or not chat_id:
        return "Telegram bot settings are missing."

    response = requests.post(
        f"https://api.telegram.org/bot{bot_token}/sendMessage",
        data={
            "chat_id": chat_id,
            "text": (
                "Punjab Weather Alert Test\n"
                "Your Telegram bot is connected successfully.\n"
                "Future moderate/high alert bands will trigger weather warnings here."
            ),
        },
        timeout=20,
    )
    response.raise_for_status()
    return "Telegram test message sent successfully."


def _build_alert_signature(row: pd.Series) -> str:
    temp_value = row.get("Temp")
    rounded_temp = round(float(temp_value), 1) if pd.notna(temp_value) else "na"
    return "|".join(
        [
            str(row.get("Alert_Band", "")),
            str(row.get("Risk_Score", "")),
            str(row.get("Forecast_Rain_Events", "")),
            str(rounded_temp),
            str(row.get("Smart_Alerts", "")),
        ]
    )


def _load_telegram_state() -> dict[str, str]:
    if not TELEGRAM_STATE_FILE.exists():
        return {}
    try:
        return json.loads(TELEGRAM_STATE_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _save_telegram_state(state: dict[str, str]) -> None:
    TELEGRAM_STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def predict_city_outlook(current_row: pd.Series, city_history: pd.DataFrame) -> dict[str, float | str]:
    recent_history = city_history.tail(8).copy()
    history_count = len(recent_history)

    temp_values = recent_history["Temp"].dropna() if "Temp" in recent_history else pd.Series(dtype=float)
    risk_values = recent_history["Risk_Score"].dropna() if "Risk_Score" in recent_history else pd.Series(dtype=float)
    visibility_values = (
        recent_history["Visibility"].dropna() if "Visibility" in recent_history else pd.Series(dtype=float)
    )

    temp_delta = 0.0
    risk_delta = 0.0
    visibility_penalty = 0.0

    if len(temp_values) >= 2:
        temp_delta = float(temp_values.iloc[-1] - temp_values.iloc[0])
    if len(risk_values) >= 2:
        risk_delta = float(risk_values.iloc[-1] - risk_values.iloc[0])
    if len(visibility_values) >= 2 and visibility_values.min() < 1500:
        visibility_penalty = 8.0

    predicted_risk = float(current_row["Risk_Score"])
    predicted_risk += temp_delta * 2.2
    predicted_risk += risk_delta * 0.35
    predicted_risk += visibility_penalty
    predicted_risk += float(current_row.get("Forecast_Rain_Events", 0)) * 2.0

    forecast_max_temp = current_row.get("Forecast_Max_Temp")
    if pd.notna(forecast_max_temp) and pd.notna(current_row.get("Temp")):
        predicted_risk += max(0.0, float(forecast_max_temp) - float(current_row["Temp"])) * 1.5

    predicted_risk = max(0.0, min(100.0, predicted_risk))
    yield_protection_index = max(0.0, min(100.0, 100.0 - predicted_risk + 8.0))

    if history_count >= 8:
        confidence = "High"
    elif history_count >= 4:
        confidence = "Moderate"
    else:
        confidence = "Low"

    if predicted_risk >= 70:
        note = "Protective actions should be prioritized in the next 24 hours."
    elif predicted_risk >= 40:
        note = "Conditions suggest moderate stress buildup over the next 24 hours."
    else:
        note = "Short-range trend looks relatively stable for field operations."

    return {
        "Predicted_Risk_24h": round(predicted_risk, 1),
        "Yield_Protection_Index": round(yield_protection_index, 1),
        "Prediction_Confidence": confidence,
        "Prediction_Note": note,
    }
