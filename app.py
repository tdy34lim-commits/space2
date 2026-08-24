from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pyproj import Transformer


st.set_page_config(page_title="서울 온열질환 구급출동 지도", page_icon="🌡️", layout="wide")

DATA_DIR = Path(__file__).parent / "data"
CSV_PATH = DATA_DIR / "heat_illness_combined-2.csv"

SEASON_ORDER = ["봄", "여름", "가을", "겨울"]

# Simplified outer boundary of Seoul, kept inline so no boundary file is needed.
SEOUL_BOUNDARY = [(126.893376, 37.452806), (126.882746, 37.464391), (126.8527, 37.481819), (126.814629, 37.474649), (126.821564, 37.502156), (126.822119, 37.540677), (126.764495, 37.555275), (126.793441, 37.58452), (126.802581, 37.605033), (126.853632, 37.571791), (126.885245, 37.593894), (126.900304, 37.611191), (126.91228, 37.644315), (126.947565, 37.659215), (126.985718, 37.646093), (126.994017, 37.666783), (127.015415, 37.701455), (127.048632, 37.694063), (127.081105, 37.696137), (127.096447, 37.669689), (127.09457, 37.644571), (127.112203, 37.632643), (127.118477, 37.607613), (127.116664, 37.594017), (127.101144, 37.576071), (127.104929, 37.556422), (127.12861, 37.56616), (127.177155, 37.5812), (127.183795, 37.545574), (127.145439, 37.516065), (127.141048, 37.505407), (127.16139, 37.500201), (127.148683, 37.484043), (127.13083, 37.467745), (127.104341, 37.462174), (127.087855, 37.444894), (127.070905, 37.430191), (127.052325, 37.428297), (127.031196, 37.465626), (127.003675, 37.46772), (126.97458, 37.454413), (126.94022, 37.435712), (126.920275, 37.440466), (126.902988, 37.434068), (126.893376, 37.452806)]


@st.cache_data
def load_data() -> pd.DataFrame:
    """Load one row per emergency dispatch and prepare display fields."""
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig", low_memory=False)
    df["신고연도"] = pd.to_numeric(df["신고연도"], errors="coerce").astype("Int64")
    df["환자연령"] = pd.to_numeric(df["환자연령"], errors="coerce")
    df["시간단위기온"] = pd.to_numeric(df["시간단위기온"], errors="coerce")
    df["ADM_CD"] = df["ADM_CD"].astype("string").str.replace(r"\.0$", "", regex=True)

    # Requested bins are intentionally non-overlapping: 0–29, 30–49, and 50+.
    df["연령대"] = pd.cut(
        df["환자연령"],
        bins=[-1, 29, 49, float("inf")],
        labels=["0~29세", "30~49세", "50세 이상"],
    ).astype("string").fillna("미상")

    # Exclude the one record without an administrative-dong code and records
    # with an unknown patient age before any filters or displayed counts.
    df = df.dropna(subset=["ADM_CD", "환자연령"])

    report_date = df["신고일자"].astype("string").str.replace(r"\.0$", "", regex=True)
    report_time = df["신고시각"].astype("string").str.replace(r"\.0$", "", regex=True).str.zfill(6)
    df["발생시각표시"] = (
        report_date.str.slice(0, 4)
        + "-"
        + report_date.str.slice(4, 6)
        + "-"
        + report_date.str.slice(6, 8)
        + " "
        + report_time.str.slice(0, 2)
        + ":"
        + report_time.str.slice(2, 4)
    )
    df["지역표시"] = (
        df["시군구명"].astype("string").fillna("")
        + " "
        + df["ADM_NM"].astype("string").fillna("")
    ).str.strip()

    # The incident point coordinates use EPSG:5186; Plotly maps need WGS84 longitude/latitude.
    x = pd.to_numeric(df["GIS_X좌표"], errors="coerce").to_numpy()
    y = pd.to_numeric(df["GIS_Y좌표"], errors="coerce").to_numpy()
    lon, lat = Transformer.from_crs("EPSG:5186", "EPSG:4326", always_xy=True).transform(x, y)
    df["lon"] = lon
    df["lat"] = lat
    return df.dropna(subset=["lon", "lat"])


def build_map(filtered: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if not filtered.empty:
        point_data = filtered.copy()
        point_data["기온표시"] = point_data["시간단위기온"].map(
            lambda x: f"{x:.1f}°C" if pd.notna(x) else "정보 없음"
        )
        fig.add_trace(
            go.Scattermapbox(
                lon=point_data["lon"],
                lat=point_data["lat"],
                mode="markers",
                marker={"size": 11, "color": "#e31a1c", "opacity": 0.82},
                customdata=point_data[["발생시각표시", "지역표시", "기온표시"]],
                hovertemplate=(
                    "<b>온열질환 구급출동</b><br>발생 시기: %{customdata[0]}"
                    "<br>지역: %{customdata[1]}<br>발생 당시 기온: %{customdata[2]}<extra></extra>"
                ),
                name="개별 출동 지점",
            )
        )

    # Draw the Seoul outline above the base map and below the point hover layer.
    fig.add_trace(
        go.Scattermapbox(
            lon=[point[0] for point in SEOUL_BOUNDARY],
            lat=[point[1] for point in SEOUL_BOUNDARY],
            mode="lines",
            line={"color": "#111111", "width": 3},
            hoverinfo="skip",
            name="서울시 경계",
        )
    )

    fig.update_layout(
        mapbox={"style": "carto-positron", "center": {"lat": 37.5665, "lon": 126.9780}, "zoom": 9.15},
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        height=710,
        showlegend=False,
    )
    return fig


st.title("서울 온열질환 구급출동 현황")
st.caption("2020–2022년 출동 지점 기준 · 붉은 원 하나가 온열질환 구급출동 1건을 의미합니다.")

data = load_data()

# Filters sit in the upper-right area above the map.
spacer, year_col, season_col, age_col = st.columns([2.0, 1.15, 1.15, 1.15])
with year_col:
    selected_year = st.selectbox("연도", ["전체"] + sorted(data["신고연도"].dropna().astype(int).unique().tolist()))
with season_col:
    available_seasons = [s for s in SEASON_ORDER if s in set(data["계절구분명"].dropna())]
    selected_season = st.selectbox("계절", ["전체"] + available_seasons)
with age_col:
    selected_age = st.selectbox("연령대", ["전체", "0~29세", "30~49세", "50세 이상"])

filtered = data.copy()
if selected_year != "전체":
    filtered = filtered[filtered["신고연도"] == int(selected_year)]
if selected_season != "전체":
    filtered = filtered[filtered["계절구분명"] == selected_season]
if selected_age != "전체":
    filtered = filtered[filtered["연령대"] == selected_age]

metric1, metric2, metric3 = st.columns(3)
metric1.metric("필터 적용 출동 건수", f"{filtered['구급보고서번호'].nunique():,}건")
metric2.metric("발생 행정동 수", f"{filtered['ADM_CD'].nunique():,}개")
metric3.metric("평균 발생 당시 기온", f"{filtered['시간단위기온'].mean():.1f}°C" if not filtered.empty else "–")

st.plotly_chart(build_map(filtered), use_container_width=True, config={"scrollZoom": True})
st.info("행정동 코드가 없거나 연령이 미상인 사례는 분석에서 제외했습니다. 붉은 원에 마우스를 올리면 개별 출동의 발생 시기·지역·기온을 확인할 수 있습니다.")
