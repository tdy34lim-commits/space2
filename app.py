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
SEASON_COLORS = {"봄": "#2ca25f", "여름": "#e31a1c", "가을": "#ffcc00", "겨울": "#3182bd"}
TIME_BAND_ORDER = ["00~06시", "06~12시", "12~18시", "18~24시"]

@st.cache_data
def load_data() -> pd.DataFrame:
    """Load one row per emergency dispatch and prepare display fields."""
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig", low_memory=False)
    df["신고연도"] = pd.to_numeric(df["신고연도"], errors="coerce").astype("Int64")
    df["신고월"] = pd.to_numeric(df["신고월"], errors="coerce").astype("Int64")
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
    report_hour = pd.to_numeric(report_time.str.slice(0, 2), errors="coerce")
    df["시간대"] = pd.cut(
        report_hour,
        bins=[-1, 5, 11, 17, 23],
        labels=TIME_BAND_ORDER,
    ).astype("string")
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


def make_hover_data(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["기온표시"] = result["시간단위기온"].map(
        lambda x: f"{x:.1f}°C" if pd.notna(x) else "정보 없음"
    )
    return result


def build_incident_map(filtered: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if not filtered.empty:
        point_data = make_hover_data(filtered)
        for season in SEASON_ORDER:
            season_data = point_data[point_data["계절구분명"] == season]
            if season_data.empty:
                continue
            fig.add_trace(
                go.Scattermapbox(
                    lon=season_data["lon"],
                    lat=season_data["lat"],
                    mode="markers",
                    marker={
                        "size": 7.15,
                        "color": SEASON_COLORS[season],
                        "opacity": 0.82,
                    },
                    customdata=season_data[["발생시각표시", "지역표시", "기온표시", "계절구분명"]],
                    hovertemplate=(
                        "<b>온열질환 구급출동</b><br>발생 시기: %{customdata[0]}"
                        "<br>지역: %{customdata[1]}<br>발생 당시 기온: %{customdata[2]}"
                        "<br>계절: %{customdata[3]}<extra></extra>"
                    ),
                    name=season,
                )
            )

    fig.update_layout(
        mapbox={"style": "carto-positron", "center": {"lat": 37.5665, "lon": 126.9780}, "zoom": 9.15},
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        height=710,
        showlegend=False,
        annotations=[
            {
                "xref": "paper",
                "yref": "paper",
                "x": 0.985,
                "y": 0.985,
                "xanchor": "right",
                "yanchor": "top",
                "showarrow": False,
                "align": "left",
                "text": (
                    "<b>원 색상 · 계절</b><br>"
                    "<span style='color:#2ca25f'>●</span> 봄&nbsp;&nbsp;"
                    "<span style='color:#e31a1c'>●</span> 여름<br>"
                    "<span style='color:#ffcc00'>●</span> 가을&nbsp;&nbsp;"
                    "<span style='color:#3182bd'>●</span> 겨울"
                ),
                "bgcolor": "rgba(255, 255, 255, 0.92)",
                "bordercolor": "#bdbdbd",
                "borderwidth": 1,
                "borderpad": 7,
                "font": {"size": 12, "color": "#222"},
            }
        ],
    )
    return fig


def most_common(values: pd.Series) -> str:
    values = values.dropna()
    return values.mode().iat[0] if not values.empty else "정보 없음"


def aggregate_by_dong(filtered: pd.DataFrame) -> pd.DataFrame:
    if filtered.empty:
        return pd.DataFrame(columns=["ADM_CD", "지역표시", "출동건수", "lon", "lat", "대표계절", "평균기온"])
    return (
        filtered.groupby(["ADM_CD", "지역표시"], as_index=False)
        .agg(
            출동건수=("구급보고서번호", "nunique"),
            lon=("lon", "mean"),
            lat=("lat", "mean"),
            대표계절=("계절구분명", most_common),
            평균기온=("시간단위기온", "mean"),
        )
        .sort_values("출동건수", ascending=False)
    )


def build_dong_map(filtered: pd.DataFrame) -> go.Figure:
    summary = aggregate_by_dong(filtered)
    fig = go.Figure()
    if not summary.empty:
        summary["평균기온표시"] = summary["평균기온"].map(
            lambda x: f"{x:.1f}°C" if pd.notna(x) else "정보 없음"
        )
        max_count = summary["출동건수"].max()
        sizes = 10 + 20 * (summary["출동건수"] / max_count) ** 0.5
        fig.add_trace(
            go.Scattermapbox(
                lon=summary["lon"],
                lat=summary["lat"],
                mode="markers",
                marker={"size": sizes, "color": "#7a0177", "opacity": 0.78},
                customdata=summary[["지역표시", "출동건수", "대표계절", "평균기온표시"]],
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>출동 건수: %{customdata[1]}건"
                    "<br>가장 많은 계절: %{customdata[2]}<br>평균 발생 당시 기온: %{customdata[3]}<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        mapbox={"style": "carto-positron", "center": {"lat": 37.5665, "lon": 126.9780}, "zoom": 9.15},
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        height=710,
        showlegend=False,
        annotations=[
            {
                "xref": "paper",
                "yref": "paper",
                "x": 0.985,
                "y": 0.985,
                "xanchor": "right",
                "yanchor": "top",
                "showarrow": False,
                "text": "<b>행정동 집계</b><br>원 크기 = 출동 건수",
                "bgcolor": "rgba(255,255,255,0.92)",
                "bordercolor": "#bdbdbd",
                "borderwidth": 1,
                "borderpad": 7,
                "font": {"size": 12, "color": "#222"},
            }
        ],
    )
    return fig


def build_top10_chart(filtered: pd.DataFrame) -> go.Figure:
    top10 = aggregate_by_dong(filtered).head(10).sort_values("출동건수")
    fig = go.Figure(
        go.Bar(
            x=top10["출동건수"],
            y=top10["지역표시"],
            orientation="h",
            marker_color="#d7301f",
            text=top10["출동건수"],
            textposition="outside",
            hovertemplate="%{y}<br>출동 건수: %{x}건<extra></extra>",
        )
    )
    fig.update_layout(
        title="출동 건수 상위 10개 행정동",
        margin={"l": 0, "r": 35, "t": 42, "b": 25},
        height=360,
        xaxis_title="출동 건수",
        yaxis_title=None,
        plot_bgcolor="white",
    )
    return fig


def build_time_trend_chart(filtered: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for band, color in zip(TIME_BAND_ORDER, ["#6a3d9a", "#1f78b4", "#e31a1c", "#ff7f00"]):
        counts = (
            filtered[filtered["시간대"] == band]
            .groupby("신고월")["구급보고서번호"]
            .nunique()
            .reindex(range(1, 13), fill_value=0)
        )
        fig.add_trace(go.Scatter(x=list(range(1, 13)), y=counts, mode="lines+markers", name=band, line={"color": color}))
    fig.update_layout(
        title="월별·시간대별 출동 추이",
        margin={"l": 0, "r": 10, "t": 42, "b": 25},
        height=360,
        xaxis={"title": "신고월", "tickmode": "linear", "dtick": 1},
        yaxis_title="출동 건수",
        legend={"orientation": "h", "y": 1.12, "x": 0},
        plot_bgcolor="white",
    )
    return fig


st.title("서울 온열질환 구급출동 현황")
st.caption("2020–2022년 온열질환 구급출동 · 행정동 코드 또는 연령 정보가 없는 사례는 제외")

data = load_data()

# Filters sit in the upper-right area above the map.
view_col, year_col, time_col, age_col = st.columns([1.6, 1.15, 1.15, 1.15])
with view_col:
    selected_view = st.radio("지도 보기", ["개별 출동", "행정동 집계"], horizontal=True)
with year_col:
    selected_year = st.selectbox("연도", ["전체"] + sorted(data["신고연도"].dropna().astype(int).unique().tolist()))
with time_col:
    available_time_bands = [band for band in TIME_BAND_ORDER if band in set(data["시간대"].dropna())]
    selected_time_band = st.selectbox("시간대", ["전체"] + available_time_bands)
with age_col:
    selected_age = st.selectbox("연령대", ["전체", "0~29세", "30~49세", "50세 이상"])

filtered = data.copy()
if selected_year != "전체":
    filtered = filtered[filtered["신고연도"] == int(selected_year)]
if selected_time_band != "전체":
    filtered = filtered[filtered["시간대"] == selected_time_band]
if selected_age != "전체":
    filtered = filtered[filtered["연령대"] == selected_age]

metric1, metric2, metric3 = st.columns(3)
metric1.metric("필터 적용 출동 건수", f"{filtered['구급보고서번호'].nunique():,}건")
metric2.metric("발생 행정동 수", f"{filtered['ADM_CD'].nunique():,}개")
metric3.metric("평균 발생 당시 기온", f"{filtered['시간단위기온'].mean():.1f}°C" if not filtered.empty else "–")

if selected_view == "개별 출동":
    st.plotly_chart(build_incident_map(filtered), use_container_width=True, config={"scrollZoom": True})
else:
    st.plotly_chart(build_dong_map(filtered), use_container_width=True, config={"scrollZoom": True})

left_chart, right_chart = st.columns(2)
with left_chart:
    st.plotly_chart(build_top10_chart(filtered), use_container_width=True)
with right_chart:
    st.plotly_chart(build_time_trend_chart(filtered), use_container_width=True)

download_columns = ["구급보고서번호", "발생시각표시", "지역표시", "신고연도", "신고월", "시간대", "연령대", "계절구분명", "시간단위기온"]
st.download_button(
    "필터 적용 출동 데이터 CSV 다운로드",
    data=filtered[download_columns].to_csv(index=False).encode("utf-8-sig"),
    file_name="온열질환_구급출동_필터결과.csv",
    mime="text/csv",
)
st.info("개별 출동 보기에서는 계절별 색의 원을, 행정동 집계 보기에서는 출동 건수에 비례한 보라색 원을 확인할 수 있습니다.")
