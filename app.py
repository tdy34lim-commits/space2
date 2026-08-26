from __future__ import annotations

import json
import math
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pyproj import Transformer
from shapely.geometry import Polygon
from shapely.ops import unary_union


st.set_page_config(page_title="서울 온열질환 발생 분포 및 소방서 권역", page_icon="🌡️", layout="wide")

DATA_DIR = Path(__file__).parent / "data"
CSV_PATH = DATA_DIR / "heat_illness_combined-2.csv"
FIRE_STATIONS_PATH = DATA_DIR / "fire_stations.csv"
SEOUL_BOUNDARIES_URL = (
    "https://raw.githubusercontent.com/raqoon886/Local_HangJeongDong/"
    "master/hangjeongdong_%EC%84%9C%EC%9A%B8%ED%8A%B9%EB%B3%84%EC%8B%9C.geojson"
)

SEASON_ORDER = ["봄", "여름", "가을", "겨울"]
SEASON_COLORS = {"봄": "#2ca25f", "여름": "#e31a1c", "가을": "#FFD000", "겨울": "#3182bd"}
TIME_BAND_ORDER = ["00~06시", "06~12시", "12~18시", "18~24시"]
SEOUL_BOUNDARY = [(126.893376, 37.452806), (126.882746, 37.464391), (126.8527, 37.481819), (126.814629, 37.474649), (126.821564, 37.502156), (126.822119, 37.540677), (126.764495, 37.555275), (126.793441, 37.58452), (126.802581, 37.605033), (126.853632, 37.571791), (126.885245, 37.593894), (126.900304, 37.611191), (126.91228, 37.644315), (126.947565, 37.659215), (126.985718, 37.646093), (126.994017, 37.666783), (127.015415, 37.701455), (127.048632, 37.694063), (127.081105, 37.696137), (127.096447, 37.669689), (127.09457, 37.644571), (127.112203, 37.632643), (127.118477, 37.607613), (127.116664, 37.594017), (127.101144, 37.576071), (127.104929, 37.556422), (127.12861, 37.56616), (127.177155, 37.5812), (127.183795, 37.545574), (127.145439, 37.516065), (127.141048, 37.505407), (127.16139, 37.500201), (127.148683, 37.484043), (127.13083, 37.467745), (127.104341, 37.462174), (127.087855, 37.444894), (127.070905, 37.430191), (127.052325, 37.428297), (127.031196, 37.465626), (127.003675, 37.46772), (126.97458, 37.454413), (126.94022, 37.435712), (126.920275, 37.440466), (126.902988, 37.434068), (126.893376, 37.452806)]

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


@st.cache_data
def load_fire_stations() -> pd.DataFrame:
    stations = pd.read_csv(FIRE_STATIONS_PATH, encoding="utf-8-sig")
    x = pd.to_numeric(stations["X좌표"], errors="coerce").to_numpy()
    y = pd.to_numeric(stations["Y좌표"], errors="coerce").to_numpy()
    lon, lat = Transformer.from_crs("EPSG:5186", "EPSG:4326", always_xy=True).transform(x, y)
    stations["lon"] = lon
    stations["lat"] = lat
    return stations.dropna(subset=["lon", "lat"])


def circle_coordinates(lon: float, lat: float, radius_m: float = 1000, steps: int = 48) -> tuple[list[float], list[float]]:
    """Approximate a 1 km circle in longitude/latitude for the web map."""
    lat_delta = radius_m / 111_320
    lon_delta = radius_m / (111_320 * math.cos(math.radians(lat)))
    angles = [2 * math.pi * step / steps for step in range(steps + 1)]
    return (
        [lon + lon_delta * math.cos(angle) for angle in angles],
        [lat + lat_delta * math.sin(angle) for angle in angles],
    )


def square_coordinates(lon: float, lat: float, side_m: float = 350) -> tuple[list[float], list[float]]:
    """Return a visible black square centered on a station location."""
    half_lat = side_m / 2 / 111_320
    half_lon = side_m / 2 / (111_320 * math.cos(math.radians(lat)))
    return (
        [lon - half_lon, lon + half_lon, lon + half_lon, lon - half_lon, lon - half_lon],
        [lat - half_lat, lat - half_lat, lat + half_lat, lat + half_lat, lat - half_lat],
    )


def clipped_circle_polygons(lon: float, lat: float, radius_m: float) -> list[tuple[list[float], list[float]]]:
    """Return only the parts of a station radius that fall within Seoul."""
    circle_lon, circle_lat = circle_coordinates(lon, lat, radius_m=radius_m)
    clipped = Polygon(zip(circle_lon, circle_lat)).intersection(Polygon(SEOUL_BOUNDARY))
    if clipped.is_empty:
        return []
    polygons = [clipped] if clipped.geom_type == "Polygon" else list(clipped.geoms)
    return [
        ([point[0] for point in polygon.exterior.coords], [point[1] for point in polygon.exterior.coords])
        for polygon in polygons
        if polygon.geom_type == "Polygon"
    ]


def polygon_parts(geometry):
    """Yield Polygon members from Polygon, MultiPolygon, or GeometryCollection values."""
    if geometry.is_empty:
        return
    if geometry.geom_type == "Polygon":
        yield geometry
    elif hasattr(geometry, "geoms"):
        for member in geometry.geoms:
            yield from polygon_parts(member)


@st.cache_data
def station_coverage_regions(stations: pd.DataFrame) -> list[tuple[list[float], list[float], str]]:
    """Partition 2 km service areas so each polygon carries every covering station name."""
    city_boundary = Polygon(SEOUL_BOUNDARY)
    regions: list[tuple[object, frozenset[str]]] = []

    for _, station in stations.iterrows():
        circle_lon, circle_lat = circle_coordinates(station["lon"], station["lat"], radius_m=2000)
        coverage = Polygon(zip(circle_lon, circle_lat)).intersection(city_boundary)
        if coverage.is_empty:
            continue

        existing_union = unary_union([geometry for geometry, _ in regions]) if regions else None
        updated: list[tuple[object, frozenset[str]]] = []
        for geometry, names in regions:
            for part in polygon_parts(geometry.difference(coverage)):
                updated.append((part, names))
            for part in polygon_parts(geometry.intersection(coverage)):
                updated.append((part, names | frozenset([station["소방서명"]])))

        uncovered = coverage if existing_union is None else coverage.difference(existing_union)
        for part in polygon_parts(uncovered):
            updated.append((part, frozenset([station["소방서명"]])))
        regions = updated

    return [
        (
            [point[0] for point in polygon.exterior.coords],
            [point[1] for point in polygon.exterior.coords],
            "<br>".join(sorted(names)),
        )
        for polygon, names in regions
        if polygon.area > 1e-10
    ]


@st.cache_data
def load_boundaries() -> dict:
    """Load public administrative-dong polygons without adding a large file to GitHub."""
    try:
        with urlopen(SEOUL_BOUNDARIES_URL, timeout=30) as response:
            boundaries = json.load(response)
    except (URLError, TimeoutError, json.JSONDecodeError) as error:
        raise RuntimeError("행정동 경계 데이터를 불러오지 못했습니다. 잠시 후 다시 시도해 주세요.") from error

    for feature in boundaries["features"]:
        props = feature["properties"]
        code = str(props.get("adm_cd8") or props.get("adm_cd") or props.get("ADM_CD"))
        props["ADM_CD"] = f"{code}0" if len(code) == 7 else code
        dong_name = str(props.get("adm_nm") or props.get("ADM_NM")).split()[-1]
        district_name = str(props.get("sggnm") or "")
        props["지역표시"] = f"{district_name} {dong_name}".strip()
    return boundaries


def build_incident_map(filtered: pd.DataFrame, fire_stations: pd.DataFrame | None = None) -> go.Figure:
    fig = go.Figure()
    if not filtered.empty:
        point_data = make_hover_data(filtered)
        for season in SEASON_ORDER:
            season_data = point_data[point_data["계절구분명"] == season]
            if season_data.empty:
                continue
            fig.add_trace(
                go.Scattermap(
                    lon=season_data["lon"],
                    lat=season_data["lat"],
                    mode="markers",
                    marker={"size": 7.54, "color": "#FFFFFF", "opacity": 0.95},
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scattermap(
                    lon=season_data["lon"],
                    lat=season_data["lat"],
                    mode="markers",
                    marker={
                        "size": 5.58,
                        "color": SEASON_COLORS[season],
                        "opacity": 0.95 if season == "가을" else 0.55,
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

    if fire_stations is not None:
        for circle_lon, circle_lat, station_names in station_coverage_regions(fire_stations):
            fig.add_trace(
                go.Scattermap(
                    lon=circle_lon,
                    lat=circle_lat,
                    mode="lines",
                    fill="toself",
                    fillcolor="rgba(220, 38, 38, 0.07)",
                    line={"color": "rgba(220, 38, 38, 0.16)", "width": 1},
                    text=[station_names] * len(circle_lon),
                    hovertemplate="<b>반경 2km 내 소방서</b><br>%{text}<extra></extra>",
                    showlegend=False,
                )
            )

    fig.add_trace(
        go.Scattermap(
            lon=[point[0] for point in SEOUL_BOUNDARY],
            lat=[point[1] for point in SEOUL_BOUNDARY],
            mode="lines",
            line={"color": "#1f1f1f", "width": 1},
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.update_layout(
        map={"style": "carto-positron", "center": {"lat": 37.5665, "lon": 126.9780}, "zoom": 9.15},
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
                    "<span style='color:#FFD000'>●</span> 가을&nbsp;&nbsp;"
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


def build_dong_map(filtered: pd.DataFrame, boundaries: dict) -> go.Figure:
    summary = aggregate_by_dong(filtered)
    boundary_base = pd.DataFrame(
        [
            {"ADM_CD": str(feature["properties"]["ADM_CD"]), "지역표시": feature["properties"]["지역표시"]}
            for feature in boundaries["features"]
        ]
    )
    boundary_codes = set(boundary_base["ADM_CD"])
    region_to_code = boundary_base.drop_duplicates("지역표시").set_index("지역표시")["ADM_CD"]
    summary["경계코드"] = summary["ADM_CD"].where(summary["ADM_CD"].isin(boundary_codes))
    summary["경계코드"] = summary["경계코드"].fillna(summary["지역표시"].map(region_to_code))
    summary = summary.dropna(subset=["경계코드"]).rename(columns={"경계코드": "경계_ADM_CD"})
    map_data = boundary_base.merge(summary, left_on="ADM_CD", right_on="경계_ADM_CD", how="left", suffixes=("", "_집계"))
    map_data["출동건수"] = map_data["출동건수"].fillna(0).astype(int)
    map_data["대표계절"] = map_data["대표계절"].fillna("발생 없음")
    map_data["평균기온표시"] = map_data["평균기온"].map(
        lambda x: f"{x:.1f}°C" if pd.notna(x) else "정보 없음"
    )
    fig = go.Figure()
    fig.add_trace(
        go.Choroplethmap(
            geojson=boundaries,
            locations=map_data["ADM_CD"],
            z=map_data["출동건수"],
            featureidkey="properties.ADM_CD",
            zmin=0,
            zmax=3,
            colorscale=[[0, "#ffffff"], [0.34, "#fee0d2"], [0.67, "#fc9272"], [1, "#de2d26"]],
            marker_line_color="rgba(120,120,120,0.45)",
            marker_line_width=0.45,
            colorbar={"title": "출동 건수", "tickvals": [0, 1, 2, 3], "ticktext": ["0", "1", "2", "3+"], "thickness": 13, "x": 0.98},
            customdata=map_data[["지역표시", "출동건수", "대표계절", "평균기온표시"]],
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>출동 건수: %{customdata[1]}건"
                "<br>가장 많은 계절: %{customdata[2]}<br>평균 발생 당시 기온: %{customdata[3]}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        map={"style": "carto-positron", "center": {"lat": 37.5665, "lon": 126.9780}, "zoom": 9.15},
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        height=710,
        showlegend=False,
    )
    return fig


def build_summary_heatmap(filtered: pd.DataFrame, age_group: str, scale_max: int, show_scale: bool) -> go.Figure:
    selected = filtered[filtered["연령대"] == age_group]
    counts = (
        selected.groupby(["시간대", "신고월"])["구급보고서번호"]
        .nunique()
        .unstack(fill_value=0)
        .reindex(index=TIME_BAND_ORDER, columns=range(1, 13), fill_value=0)
    )
    labels = counts.where(counts > 0, "")
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=counts.values,
            x=[f"{month}월" for month in counts.columns],
            y=counts.index,
            text=labels.values,
            texttemplate="%{text}",
            textfont={"size": 12},
            colorscale=[[0, "#ffffff"], [1, "#d7301f"]],
            zmin=0,
            zmax=max(1, scale_max),
            showscale=show_scale,
            colorbar={"title": "건수", "thickness": 12} if show_scale else None,
            hovertemplate="연령대: " + age_group + "<br>월: %{x}<br>시간대: %{y}<br>출동 건수: %{z}건<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"{age_group}: 월 × 시간대",
        margin={"l": 20, "r": 10, "t": 45, "b": 35},
        height=360,
        xaxis={"side": "bottom", "tickangle": 0},
        yaxis={"autorange": "reversed"},
        plot_bgcolor="white",
    )
    return fig


st.title("서울 온열질환 발생 분포 및 소방서 권역")
st.caption("2020–2022년 온열질환 구급출동 · 행정동 코드 또는 연령 정보가 없는 사례는 제외")

data = load_data()
fire_stations = load_fire_stations()

# Filters sit in the upper-right area above the map.
view_col, year_col, time_col, age_col = st.columns([1.6, 1.15, 1.15, 1.15])
with view_col:
    selected_view = st.radio("지도 보기", ["개별 출동", "행정동 집계", "요약 그래프"], horizontal=True)
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

show_fire_stations = False
if selected_view == "개별 출동":
    show_fire_stations = st.toggle("소방서 반경 2km 표시", value=False)

metric1, metric2, metric3 = st.columns(3)
metric1.metric("필터 적용 출동 건수", f"{filtered['구급보고서번호'].nunique():,}건")
metric2.metric("발생 행정동 수", f"{filtered['ADM_CD'].nunique():,}개")
metric3.metric("평균 발생 당시 기온", f"{filtered['시간단위기온'].mean():.1f}°C" if not filtered.empty else "–")

if selected_view == "개별 출동":
    station_overlay = fire_stations if show_fire_stations else None
    st.plotly_chart(build_incident_map(filtered, station_overlay), use_container_width=True, config={"scrollZoom": True})
elif selected_view == "행정동 집계":
    try:
        boundaries = load_boundaries()
    except RuntimeError as error:
        st.error(str(error))
        st.stop()
    st.plotly_chart(build_dong_map(filtered, boundaries), use_container_width=True, config={"scrollZoom": True})
else:
    st.subheader("연령대별 월·시간대 출동 분포")
    heatmap_scale = 0
    for age_group in ["0~29세", "30~49세", "50세 이상"]:
        count_table = filtered[filtered["연령대"] == age_group].groupby(["시간대", "신고월"])["구급보고서번호"].nunique()
        heatmap_scale = max(heatmap_scale, int(count_table.max()) if not count_table.empty else 0)
    heatmap_columns = st.columns(3)
    for index, age_group in enumerate(["0~29세", "30~49세", "50세 이상"]):
        with heatmap_columns[index]:
            st.plotly_chart(
                build_summary_heatmap(filtered, age_group, heatmap_scale, show_scale=index == 2),
                use_container_width=True,
            )

st.info("개별 출동 보기에서는 계절별 색의 원을, 행정동 집계 보기에서는 0건(흰색)부터 3건 이상(붉은색)까지의 색상으로 출동 건수를 확인할 수 있습니다.")
