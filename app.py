from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pyproj import Transformer


st.set_page_config(page_title="서울 온열질환 구급출동 지도", page_icon="🌡️", layout="wide")

DATA_DIR = Path(__file__).parent / "data"
CSV_PATH = DATA_DIR / "heat_illness_combined-2.csv"
GEOJSON_PATH = DATA_DIR / "seoul_admdong.geojson"

SEASON_ORDER = ["봄", "여름", "가을", "겨울"]


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

    # The incident point coordinates are EPSG:5181; Plotly maps need WGS84 longitude/latitude.
    x = pd.to_numeric(df["GIS_X좌표"], errors="coerce").to_numpy()
    y = pd.to_numeric(df["GIS_Y좌표"], errors="coerce").to_numpy()
    lon, lat = Transformer.from_crs("EPSG:5181", "EPSG:4326", always_xy=True).transform(x, y)
    df["lon"] = lon
    df["lat"] = lat
    return df.dropna(subset=["ADM_CD", "lon", "lat"])


@st.cache_data
def load_boundaries() -> dict:
    with GEOJSON_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def label_values(values: pd.Series, limit: int = 4) -> str:
    items = [str(value) for value in values.dropna().unique()]
    if not items:
        return "정보 없음"
    if len(items) <= limit:
        return ", ".join(items)
    return ", ".join(items[:limit]) + " 외"


def build_summary(filtered: pd.DataFrame, boundaries: dict) -> pd.DataFrame:
    base = pd.DataFrame(
        [
            {"ADM_CD": str(feature["properties"]["ADM_CD"]), "ADM_NM": feature["properties"]["ADM_NM"]}
            for feature in boundaries["features"]
        ]
    )
    summary = (
        filtered.groupby("ADM_CD", as_index=False)
        .agg(
            출동건수=("구급보고서번호", "nunique"),
            발생연도=("신고연도", label_values),
            계절=("계절구분명", label_values),
            평균기온=("시간단위기온", "mean"),
        )
    )
    result = base.merge(summary, on="ADM_CD", how="left")
    result["출동건수"] = result["출동건수"].fillna(0).astype(int)
    result["발생연도"] = result["발생연도"].fillna("발생 없음")
    result["계절"] = result["계절"].fillna("발생 없음")
    result["평균기온표시"] = result["평균기온"].map(lambda x: f"{x:.1f}°C" if pd.notna(x) else "정보 없음")
    return result


def build_map(summary: pd.DataFrame, filtered: pd.DataFrame, boundaries: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Choroplethmapbox(
            geojson=boundaries,
            locations=summary["ADM_CD"],
            z=summary["출동건수"],
            featureidkey="properties.ADM_CD",
            colorscale=[[0, "#fff5f0"], [0.35, "#fcbba1"], [0.7, "#fb6a4a"], [1, "#cb181d"]],
            zmin=0,
            zmax=max(1, int(summary["출동건수"].max())),
            marker_line_color="rgba(120, 120, 120, 0.65)",
            marker_line_width=0.5,
            colorbar={"title": "출동 건수", "thickness": 14, "x": 0.98, "y": 0.5},
            customdata=summary[["ADM_NM", "출동건수", "발생연도", "계절", "평균기온표시"]],
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>출동 건수: %{customdata[1]}건"
                "<br>발생 연도: %{customdata[2]}<br>계절: %{customdata[3]}"
                "<br>발생 당시 평균 기온: %{customdata[4]}<extra></extra>"
            ),
            name="행정동 출동 건수",
        )
    )

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
                marker={"size": 10, "color": "#e31a1c", "opacity": 0.8},
                customdata=point_data[["ADM_NM", "신고연도", "계절구분명", "연령대", "기온표시"]],
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>발생 연도: %{customdata[1]}"
                    "<br>계절: %{customdata[2]}<br>연령대: %{customdata[3]}"
                    "<br>발생 당시 기온: %{customdata[4]}<extra></extra>"
                ),
                name="개별 출동 지점",
            )
        )

    fig.update_layout(
        mapbox={"style": "carto-positron", "center": {"lat": 37.5665, "lon": 126.9780}, "zoom": 9.6},
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        height=710,
        legend={"orientation": "h", "y": 0.02, "x": 0.01, "bgcolor": "rgba(255,255,255,0.8)"},
    )
    return fig


st.title("서울 온열질환 구급출동 현황")
st.caption("2020–2022년 출동 지점 기준 · 행정동 색상은 출동 건수, 빨간 점은 개별 출동 지점")

data = load_data()
boundaries = load_boundaries()

# Filters sit in the upper-right area above the map.
spacer, year_col, season_col, age_col = st.columns([2.0, 1.15, 1.15, 1.15])
with year_col:
    selected_year = st.selectbox("연도", ["전체"] + sorted(data["신고연도"].dropna().astype(int).unique().tolist()))
with season_col:
    available_seasons = [s for s in SEASON_ORDER if s in set(data["계절구분명"].dropna())]
    selected_season = st.selectbox("계절", ["전체"] + available_seasons)
with age_col:
    selected_age = st.selectbox("연령대", ["전체", "0~29세", "30~49세", "50세 이상", "미상"])

filtered = data.copy()
if selected_year != "전체":
    filtered = filtered[filtered["신고연도"] == int(selected_year)]
if selected_season != "전체":
    filtered = filtered[filtered["계절구분명"] == selected_season]
if selected_age != "전체":
    filtered = filtered[filtered["연령대"] == selected_age]

summary = build_summary(filtered, boundaries)
metric1, metric2, metric3 = st.columns(3)
metric1.metric("필터 적용 출동 건수", f"{filtered['구급보고서번호'].nunique():,}건")
metric2.metric("발생 행정동 수", f"{(summary['출동건수'] > 0).sum():,}개")
metric3.metric("평균 발생 당시 기온", f"{filtered['시간단위기온'].mean():.1f}°C" if not filtered.empty else "–")

st.plotly_chart(build_map(summary, filtered, boundaries), use_container_width=True, config={"scrollZoom": True})
st.info("연령대는 중복 없이 0–29세, 30–49세, 50세 이상으로 분류했습니다. 행정동에 마우스를 올리면 필터 결과의 연도·계절·평균 기온을, 빨간 점에 올리면 해당 출동의 기온을 확인할 수 있습니다.")
