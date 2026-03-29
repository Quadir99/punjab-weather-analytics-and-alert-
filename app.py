import os
import json
from io import BytesIO
from datetime import datetime

import folium
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from streamlit_folium import st_folium

from analytics_core import (
    GEOJSON_FILE,
    LOCATIONS,
    add_prediction_features,
    build_enriched_dataset,
    exportable_columns,
    fetch_current_weather,
    format_visibility,
    get_city_trend,
    latest_summary_metrics,
    load_history,
    persist_snapshot,
    send_telegram_alerts,
    send_telegram_test_message,
)


load_dotenv()
API_KEY = (os.getenv("API_KEY") or "").strip()
TELEGRAM_BOT_TOKEN = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
TELEGRAM_CHAT_ID = (os.getenv("TELEGRAM_CHAT_ID") or "").strip()
if not API_KEY:
    raise ValueError("API_KEY not found in .env file")

st.set_page_config(
    page_title="Punjab Smart Weather Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background:
            linear-gradient(180deg, rgba(248,250,252,0.78) 0%, rgba(248,250,252,0.92) 100%),
            url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1600 900'><defs><linearGradient id='sky' x1='0' y1='0' x2='0' y2='1'><stop offset='0%25' stop-color='%23dbeafe'/><stop offset='55%25' stop-color='%23f8fafc'/><stop offset='100%25' stop-color='%23fef3c7'/></linearGradient></defs><rect width='1600' height='900' fill='url(%23sky)'/><circle cx='1260' cy='150' r='95' fill='%23fde68a' fill-opacity='0.85'/><ellipse cx='250' cy='170' rx='170' ry='55' fill='white' fill-opacity='0.75'/><ellipse cx='370' cy='150' rx='120' ry='42' fill='white' fill-opacity='0.82'/><ellipse cx='540' cy='185' rx='150' ry='50' fill='white' fill-opacity='0.68'/><path d='M0 630 C180 560 320 560 500 625 S860 700 1080 625 S1400 560 1600 630 L1600 900 L0 900 Z' fill='%23bbf7d0' fill-opacity='0.8'/><path d='M0 690 C200 620 420 640 620 700 S980 760 1240 700 S1460 650 1600 685 L1600 900 L0 900 Z' fill='%2384cc16' fill-opacity='0.32'/><path d='M0 760 C220 710 430 735 650 790 S1100 845 1600 770 L1600 900 L0 900 Z' fill='%2365a30d' fill-opacity='0.24'/><g stroke='%2394a3b8' stroke-opacity='0.18' stroke-width='2' fill='none'><path d='M118 640 l28 -22 l38 30 l44 -40 l52 26'/><path d='M1260 610 l32 -18 l34 26 l48 -36 l40 20'/></g></svg>");
        background-attachment: fixed;
        background-size: cover;
        background-position: center top;
    }
    [data-testid="stHeader"] {
        background: rgba(255,255,255,0);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(255,255,255,0.88) 0%, rgba(239,246,255,0.92) 100%);
        backdrop-filter: blur(10px);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=600, show_spinner=False)
def fetch_dashboard_data(api_key: str) -> pd.DataFrame:
    current_df = fetch_current_weather(api_key)
    return build_enriched_dataset(current_df, api_key)


def refresh_dashboard_data() -> None:
    df = fetch_dashboard_data(API_KEY)
    history_df = load_history(limit=120)
    df = add_prediction_features(df, history_df)
    st.session_state.weather_df = df
    st.session_state.last_refresh = datetime.now()
    persist_snapshot(df)
    try:
        alert_result = send_telegram_alerts(df, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
        st.session_state.telegram_alert_status = str(alert_result["status"])
    except requests.RequestException as exc:
        st.session_state.telegram_alert_status = f"Telegram alert send failed: {exc}"


def build_pdf_report(df: pd.DataFrame, generated_at: datetime | None) -> bytes | None:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.pdfgen import canvas
    except ImportError:
        return None

    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    y = height - 40

    avg_temp = df["Temp"].mean() if not df.empty else 0
    avg_risk = df["Risk_Score"].mean() if not df.empty else 0
    high_risk_count = int((df["Alert_Band"] == "High").sum()) if not df.empty else 0
    top_city = df.sort_values("Risk_Score", ascending=False).iloc[0]["City"] if not df.empty else "N/A"

    pdf.setFillColor(colors.HexColor("#0f172a"))
    pdf.rect(0, height - 95, width, 95, fill=1, stroke=0)
    pdf.setFillColor(colors.white)
    pdf.setFont("Helvetica-Bold", 20)
    pdf.drawString(40, height - 42, "Punjab Smart Weather Advisory Report")
    pdf.setFont("Helvetica", 11)
    pdf.drawString(40, height - 62, "District-level weather intelligence, risk scoring, and crop guidance")

    y = height - 115
    pdf.setFillColor(colors.black)
    timestamp_text = generated_at.strftime("%Y-%m-%d %H:%M:%S") if generated_at else "N/A"
    pdf.setFont("Helvetica", 10)
    pdf.drawString(40, y, f"Generated at: {timestamp_text}")
    y -= 20

    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(40, y, "Executive Summary")
    y -= 14
    pdf.setFont("Helvetica", 10)
    summary_lines = [
        f"Average Temperature: {avg_temp:.1f} C",
        f"Average Risk Score: {avg_risk:.0f}/100",
        f"High-Risk Districts: {high_risk_count}",
        f"Most Exposed District: {top_city}",
    ]
    for line in summary_lines:
        pdf.drawString(48, y, line)
        y -= 12
    y -= 6

    pdf.setStrokeColor(colors.HexColor("#cbd5e1"))
    pdf.line(40, y, width - 40, y)
    y -= 20

    for _, row in df.iterrows():
        if y < 120:
            pdf.showPage()
            y = height - 40
            pdf.setFont("Helvetica-Bold", 14)
            pdf.drawString(40, y, "District Advisory Details")
            y -= 20

        if row["Alert_Band"] == "High":
            band_color = colors.HexColor("#b91c1c")
            band_bg = colors.HexColor("#fee2e2")
        elif row["Alert_Band"] == "Moderate":
            band_color = colors.HexColor("#c2410c")
            band_bg = colors.HexColor("#ffedd5")
        else:
            band_color = colors.HexColor("#166534")
            band_bg = colors.HexColor("#dcfce7")

        pdf.setFillColor(band_bg)
        pdf.roundRect(36, y - 42, width - 72, 56, 8, fill=1, stroke=0)
        pdf.setFillColor(colors.black)
        pdf.setFont("Helvetica-Bold", 11)
        pdf.drawString(40, y, f"{row['City']} | Risk {row['Risk_Score']} | {row['Alert_Band']}")
        pdf.setFillColor(band_color)
        pdf.setFont("Helvetica-Bold", 9)
        pdf.drawRightString(width - 44, y, row["Alert_Band"].upper())
        y -= 14
        pdf.setFillColor(colors.black)
        pdf.setFont("Helvetica", 9)
        lines = [
            f"Crop: {row['Crop_Focus']} | Temp: {row['Temp']:.1f} C | Visibility: {format_visibility(row['Visibility'])}",
            f"Alerts: {row['Smart_Alerts']}",
            f"Recommendation: {row['Crop_Recommendation']}",
        ]
        for line in lines:
            pdf.drawString(48, y, line[:110])
            y -= 12
        y -= 6

    pdf.save()
    return buffer.getvalue()


def create_map(df: pd.DataFrame) -> folium.Map:
    map_punjab = folium.Map(location=[31.0, 75.7], zoom_start=7, tiles="CartoDB positron")

    for _, row in df.iterrows():
        if row["Alert_Band"] == "High":
            color = "red"
        elif row["Alert_Band"] == "Moderate":
            color = "orange"
        else:
            color = "green"

        popup_text = (
            f"<b>{row['City']}</b><br>"
            f"Crop Focus: {row['Crop_Focus']}<br>"
            f"Temperature: {row['Temp']:.1f} C<br>"
            f"Humidity: {row['Humidity']}%<br>"
            f"Visibility: {format_visibility(row['Visibility'])}<br>"
            f"Risk Score: {row['Risk_Score']}<br>"
            f"Forecast: {row['Forecast_Note']}<br>"
            f"Advisory: {row['Advisory']}"
        )

        folium.CircleMarker(
            location=[row["Lat"], row["Lon"]],
            radius=10 + (row["Risk_Score"] / 20),
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            popup=folium.Popup(popup_text, max_width=320),
            tooltip=f"{row['City']} | {row['Alert_Band']} risk",
        ).add_to(map_punjab)

    return map_punjab


def create_choropleth(df: pd.DataFrame) -> folium.Map:
    district_map = folium.Map(location=[31.0, 75.7], zoom_start=7, tiles="CartoDB positron")
    with GEOJSON_FILE.open("r", encoding="utf-8") as geojson_file:
        geojson_data = json.load(geojson_file)

    choropleth_df = df[["City", "Risk_Score", "Predicted_Risk_24h"]].copy()
    choropleth_df = choropleth_df.rename(columns={"City": "district"})

    folium.Choropleth(
        geo_data=geojson_data,
        name="Current Risk",
        data=choropleth_df,
        columns=["district", "Risk_Score"],
        key_on="feature.properties.district",
        fill_color="YlOrRd",
        fill_opacity=0.75,
        line_opacity=0.5,
        legend_name="District Risk Score",
    ).add_to(district_map)

    for _, row in choropleth_df.iterrows():
        city_meta = LOCATIONS[row["district"]]
        tooltip_html = (
            f"<b>{row['district']}</b><br>"
            f"Current risk: {row['Risk_Score']:.0f}<br>"
            f"Predicted 24h risk: {row['Predicted_Risk_24h']:.0f}"
        )
        folium.Marker(
            location=[city_meta["lat"], city_meta["lon"]],
            icon=folium.DivIcon(
                html=(
                    "<div style=\"font-size:10px;font-weight:700;color:#1f2937;"
                    "background:#ffffffd9;padding:2px 4px;border-radius:4px;\">"
                    f"{row['district']}</div>"
                )
            ),
            tooltip=tooltip_html,
        ).add_to(district_map)

    folium.LayerControl().add_to(district_map)
    return district_map


st.markdown(
    """
    <div style="
        background: linear-gradient(135deg, #eff6ff 0%, #f8fafc 45%, #fef3c7 100%);
        border: 1px solid rgba(191, 219, 254, 0.9);
        border-radius: 28px;
        padding: 28px 24px 24px 24px;
        margin-bottom: 18px;
        box-shadow: 0 16px 40px rgba(15, 23, 42, 0.08);
        text-align: center;
    ">
        <h1 style="
            margin: 0 0 10px 0;
            color: #0f172a;
            font-size: 2.2rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            line-height: 1.1;
        ">
            Punjab Smart Weather and Agri Intelligence
        </h1>
        <p style="
            max-width: 860px;
            margin: 0 auto;
            color: #334155;
            font-size: 1.02rem;
            line-height: 1.7;
        ">
            A decision-support dashboard for district weather risk, crop advisories,
            visibility conditions, and short-term forecast intelligence.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if "weather_df" not in st.session_state:
    st.session_state.weather_df = pd.DataFrame()
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = None
if "telegram_alert_status" not in st.session_state:
    st.session_state.telegram_alert_status = "Telegram alerts not sent yet."

with st.sidebar:
    st.header("Control Panel")
    selected_cities = st.multiselect(
        "Districts",
        options=list(LOCATIONS.keys()),
        default=list(LOCATIONS.keys()),
    )
    selected_risk_bands = st.multiselect(
        "Risk Bands",
        options=["High", "Moderate", "Low"],
        default=["High", "Moderate", "Low"],
    )
    if st.button("Refresh Intelligence", use_container_width=True):
        fetch_dashboard_data.clear()
        refresh_dashboard_data()
    if st.button("Send Test Telegram Alert", use_container_width=True):
        try:
            st.session_state.telegram_alert_status = send_telegram_test_message(
                TELEGRAM_BOT_TOKEN,
                TELEGRAM_CHAT_ID,
            )
        except requests.RequestException as exc:
            st.session_state.telegram_alert_status = f"Telegram test failed: {exc}"

    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        st.success("Telegram alerts are enabled for moderate/high alert bands.")
    else:
        st.info("Add TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env to enable Telegram alerts.")


if st.session_state.weather_df.empty:
    with st.spinner("Building the weather intelligence snapshot..."):
        refresh_dashboard_data()

weather_df = st.session_state.weather_df.copy()
history_df = load_history(limit=60)

if selected_cities:
    weather_df = weather_df[weather_df["City"].isin(selected_cities)]
weather_df = weather_df[weather_df["Alert_Band"].isin(selected_risk_bands)]

if st.session_state.last_refresh:
    st.write(
        f"Last refreshed: {st.session_state.last_refresh.strftime('%Y-%m-%d %H:%M:%S')}"
    )
st.caption(f"Telegram status: {st.session_state.telegram_alert_status}")

if weather_df.empty:
    st.warning("No districts match the selected filters.")
    st.stop()

ranked_df = weather_df.sort_values(["Risk_Score", "Forecast_Rain_Events"], ascending=[False, False])
summary = latest_summary_metrics(weather_df)
st.markdown(
    """
    <style>
    .kpi-band {
        background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 20px;
        padding: 10px 8px;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
        margin-bottom: 18px;
    }
    .kpi-tile {
        padding: 12px 14px;
        min-height: 102px;
        border-right: 1px solid rgba(148, 163, 184, 0.16);
    }
    .kpi-last {
        border-right: none;
    }
    .kpi-tile .label {
        font-size: 0.76rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 8px;
        opacity: 0.72;
    }
    .kpi-tile .value {
        font-size: 1.85rem;
        font-weight: 800;
        line-height: 1.1;
        margin-bottom: 6px;
    }
    .kpi-tile .subtext {
        font-size: 0.82rem;
        line-height: 1.3;
        opacity: 0.8;
    }
    .kpi-tile .icon {
        font-size: 1.2rem;
        margin-bottom: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

summary_cards = [
    {
        "icon": "🌡️",
        "label": "Average Temperature",
        "value": f"{summary['avg_temp']:.1f} C",
        "subtext": "Live district average",
        "color": "#9a3412",
    },
    {
        "icon": "⚠️",
        "label": "Average Risk Score",
        "value": f"{summary['avg_risk']:.0f}/100",
        "subtext": "Combined weather stress",
        "color": "#1d4ed8",
    },
    {
        "icon": "🚨",
        "label": "High Risk Districts",
        "value": str(summary["high_risk_count"]),
        "subtext": "Active alert bracket count",
        "background": "linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%)",
        "color": "#b91c1c",
    },
    {
        "icon": "📍",
        "label": "Most Exposed District",
        "value": str(summary["top_city"]),
        "subtext": "Highest current risk",
        "color": "#6d28d9",
    },
    {
        "icon": "🔮",
        "label": "Avg Predicted 24h Risk",
        "value": f"{weather_df['Predicted_Risk_24h'].mean():.0f}/100",
        "subtext": "Short-range projected stress",
        "color": "#be123c",
    },
    {
        "icon": "🌾",
        "label": "Avg Yield Protection Index",
        "value": f"{weather_df['Yield_Protection_Index'].mean():.0f}/100",
        "subtext": "Estimated crop safety buffer",
        "color": "#047857",
    },
    {
        "icon": "✅",
        "label": "High Confidence Districts",
        "value": str(int((weather_df["Prediction_Confidence"] == "High").sum())),
        "subtext": "Prediction support strength",
        "color": "#334155",
    },
]

st.markdown("### Executive Insights")
first_row = summary_cards[:4]
second_row = summary_cards[4:]

for row_cards in (first_row, second_row):
    row_cols = st.columns(len(row_cards))
    for idx, card in enumerate(row_cards):
        with row_cols[idx]:
            tile_class = "kpi-tile kpi-last" if idx == len(row_cards) - 1 else "kpi-tile"
            st.markdown(
                f"""
                <div class="kpi-band">
                    <div class="{tile_class}" style="color:{card['color']};">
                        <div class="icon">{card['icon']}</div>
                        <div class="label">{card['label']}</div>
                        <div class="value">{card['value']}</div>
                        <div class="subtext">{card['subtext']}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

st.markdown("### District Status Cards")
card_cols = st.columns(3)
for idx, (_, row) in enumerate(weather_df.sort_values("Risk_Score", ascending=False).iterrows()):
    card_col = card_cols[idx % 3]
    if row["Alert_Band"] == "High":
        accent = "#b91c1c"
        bg = "#fee2e2"
    elif row["Alert_Band"] == "Moderate":
        accent = "#c2410c"
        bg = "#ffedd5"
    else:
        accent = "#166534"
        bg = "#dcfce7"
    with card_col:
        st.markdown(
            f"""
            <div style="background:{bg}; border-left:6px solid {accent}; padding:12px; border-radius:10px; margin-bottom:12px;">
                <div style="font-weight:700; font-size:18px;">{row['City']}</div>
                <div style="margin-top:4px;">Temp: {row['Temp']:.1f} C</div>
                <div>Risk Score: {row['Risk_Score']}</div>
                <div>Band: {row['Alert_Band']}</div>
                <div style="margin-top:6px; font-size:13px;">{row['Advisory']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("### Executive Snapshot")
left_col, right_col = st.columns([1.15, 1.0])

with left_col:
    score_df = ranked_df[
        ["City", "Crop_Focus", "Risk_Score", "Alert_Band", "Smart_Alerts", "Forecast_Note"]
    ]
    st.dataframe(score_df, use_container_width=True, hide_index=True)

with right_col:
    top_city = ranked_df.iloc[0]
    top_city_forecast_temp = (
        "N/A"
        if pd.isna(top_city["Forecast_Max_Temp"])
        else f"{top_city['Forecast_Max_Temp']:.1f} C"
    )
    st.markdown(
        f"""
        #### Priority Advisory
        **District:** {top_city['City']}  
        **Crop Focus:** {top_city['Crop_Focus']}  
        **Risk Band:** {top_city['Alert_Band']}  
        **Advisory:** {top_city['Advisory']}
        """
    )
    st.markdown(
        f"""
        #### Forecast Watch
        **District:** {top_city['City']}  
        Next-24h rain events: **{int(top_city['Forecast_Rain_Events'])}**  
        Forecast maximum temperature: **{top_city_forecast_temp}**
        """
    )

st.markdown("---")
st.markdown("### District Trend Explorer")
trend_city = st.selectbox("Choose a district for historical trend view", list(weather_df["City"]))
trend_df = get_city_trend(history_df, trend_city)

trend_left, trend_right = st.columns(2)
with trend_left:
    if trend_df.empty:
        st.info("Trend history will appear after more refresh cycles are stored.")
    else:
        temp_chart = trend_df.set_index("Fetched_At")[["Temp", "Humidity"]]
        st.line_chart(temp_chart, use_container_width=True)

with trend_right:
    if trend_df.empty:
        st.info("Visibility and risk history will populate automatically over time.")
    else:
        visibility_chart = trend_df.set_index("Fetched_At")[["Visibility", "Risk_Score"]]
        st.line_chart(visibility_chart, use_container_width=True)

st.markdown("### Predictive Intelligence")
prediction_left, prediction_right = st.columns([1.15, 1.0])
with prediction_left:
    prediction_df = weather_df.sort_values("Predicted_Risk_24h", ascending=False)[
        [
            "City",
            "Predicted_Risk_24h",
            "Yield_Protection_Index",
            "Prediction_Confidence",
            "Prediction_Note",
        ]
    ]
    st.dataframe(prediction_df, use_container_width=True, hide_index=True)

with prediction_right:
    top_prediction = weather_df.sort_values("Predicted_Risk_24h", ascending=False).iloc[0]
    st.markdown(
        f"""
        #### 24-Hour Risk Outlook
        **District:** {top_prediction['City']}  
        **Predicted Risk:** {top_prediction['Predicted_Risk_24h']:.0f}/100  
        **Yield Protection Index:** {top_prediction['Yield_Protection_Index']:.0f}/100  
        **Model Confidence:** {top_prediction['Prediction_Confidence']}  
        **Interpretation:** {top_prediction['Prediction_Note']}
        """
    )

st.markdown("### Crop Recommendation Panel")
rec_city = st.selectbox(
    "Choose a district for detailed crop recommendation",
    list(weather_df.sort_values("City")["City"]),
    key="crop_recommendation_city",
)
rec_row = weather_df[weather_df["City"] == rec_city].iloc[0]
rec_cols = st.columns([1, 1])
with rec_cols[0]:
    st.markdown(
        f"""
        #### Advisory Summary
        **District:** {rec_row['City']}  
        **Crop Focus:** {rec_row['Crop_Focus']}  
        **Temperature:** {rec_row['Temp']:.1f} C  
        **Visibility:** {format_visibility(rec_row['Visibility'])}  
        **Risk Band:** {rec_row['Alert_Band']}
        """
    )
with rec_cols[1]:
    st.markdown(
        f"""
        #### Recommended Actions
        {rec_row['Crop_Recommendation']}
        """
    )

st.markdown("---")
st.markdown("### Regional Map Intelligence")
map_df = weather_df.sort_values("Risk_Score", ascending=False)
st_folium(create_map(map_df), width=1100, height=520)

st.markdown("### District Choropleth")
st.caption(
    "District polygons are stored locally for a stable offline demo. "
    "Colors represent current district risk score."
)
st_folium(create_choropleth(map_df), width=1100, height=520)

st.markdown("### District Intelligence Table")
display_df = ranked_df[
    list(exportable_columns())
].copy()
display_df["Visibility"] = display_df["Visibility"].apply(format_visibility)
display_df["Forecast_Max_Temp"] = display_df["Forecast_Max_Temp"].apply(
    lambda value: "N/A" if pd.isna(value) else f"{value:.1f} C"
)
display_df["Predicted_Risk_24h"] = display_df["Predicted_Risk_24h"].apply(
    lambda value: f"{value:.1f}"
)
display_df["Yield_Protection_Index"] = display_df["Yield_Protection_Index"].apply(
    lambda value: f"{value:.1f}"
)
st.dataframe(display_df, use_container_width=True, hide_index=True)

csv_data = display_df.to_csv(index=False).encode("utf-8")
st.markdown(
    """
    <div style="
        background: linear-gradient(135deg, #eff6ff 0%, #fef3c7 100%);
        border: 1px solid #dbeafe;
        border-radius: 18px;
        padding: 18px 16px 10px 16px;
        margin-bottom: 12px;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
    ">
        <div style="text-align:center; font-size:1.05rem; font-weight:700; color:#1e3a8a; margin-bottom:6px;">
            Export Your Advisory Snapshot
        </div>
        <div style="text-align:center; color:#475569; font-size:0.95rem; margin-bottom:4px;">
            Save the live advisory snapshot as a spreadsheet or a presentation-ready PDF.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

export_cols = st.columns([1.2, 2.4, 2.4, 1.2])
with export_cols[0]:
    st.empty()
with export_cols[1]:
    st.download_button(
        "📊 Download CSV Report",
        data=csv_data,
        file_name="punjab_weather_advisory_report.csv",
        mime="text/csv",
        use_container_width=True,
    )
with export_cols[2]:
    pdf_data = build_pdf_report(weather_df.sort_values("Risk_Score", ascending=False), st.session_state.last_refresh)
    if pdf_data is not None:
        st.download_button(
            "📄 Download PDF Report",
            data=pdf_data,
            file_name="punjab_weather_advisory_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    else:
        st.info("Install `reportlab` to enable PDF export.")
with export_cols[3]:
    st.empty()
