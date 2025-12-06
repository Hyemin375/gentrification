# app.py
import os
import glob
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import altair as alt


# =========================================
# 행정동 코드 → 이름 매핑
# =========================================
REGION_CODE_MAP = {
    "공릉동": "11300",
    "성수동":   "11400",
    "신사동":     "10900",
    "연남동":   "12400",
    "을지로":   "10400",
    "익선동":    "13300",
    "한남동":    "13100",
}
REGION_NAME_MAP = {v: k for k, v in REGION_CODE_MAP.items()}

# =========================================
# 디렉토리 경로
# =========================================
ANALYSIS_DIR = "analysis"
REALTIME_GOLD_DIR = "realtime_gold"
REALTIME_PROCESSED_DIR = "realtime_processed"
STRUCTURED_DIR = "structured"  # SGI 정형 데이터 CSV 디렉토리


# 워드클라우드 한글 폰트 경로 (환경에 맞게 필요 시 수정)
KOREAN_FONT_PATH = "C:/Windows/Fonts/malgun.ttf"


# =========================================
# 데이터 로딩 함수
# =========================================
@st.cache_data
def load_analysis(path: str) -> pd.DataFrame:
    files = glob.glob(os.path.join(path, "*.csv"))
    if not files:
        return pd.DataFrame()

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    # 날짜 컬럼 파싱
    if "period_start" in df.columns:
        df["period_start"] = pd.to_datetime(df["period_start"], errors="coerce")
    if "period_end" in df.columns:
        df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")
    if "processed_at" in df.columns:
        df["processed_at"] = pd.to_datetime(df["processed_at"], errors="coerce")

    # 행정동 코드 → 이름 (매핑 안되면 원래 값 유지)
    if "administrative_dong" in df.columns:
        df["administrative_dong"] = (
            df["administrative_dong"].astype(str).map(REGION_NAME_MAP)
            .fillna(df["administrative_dong"].astype(str))
        )

    return df


@st.cache_data
def load_realtime_gold(path: str) -> pd.DataFrame:
    files = glob.glob(os.path.join(path, "*.csv"))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    # 날짜 처리
    if "postdate" in df.columns:
        df["postdate"] = pd.to_datetime(df["postdate"], errors="coerce")
    else:
        if set(["year", "month", "day"]).issubset(df.columns):
            df["postdate"] = pd.to_datetime(
                df["year"].astype(int).astype(str)
                + "-"
                + df["month"].astype(int).astype(str).str.zfill(2)
                + "-"
                + df["day"].astype(int).astype(str).str.zfill(2),
                errors="coerce",
            )
        else:
            df["postdate"] = pd.NaT

    if "administrative_dong" in df.columns:
        df["administrative_dong"] = (
            df["administrative_dong"].astype(str).map(REGION_NAME_MAP)
            .fillna(df["administrative_dong"].astype(str))
        )

    return df


@st.cache_data
def load_realtime_processed(path: str) -> pd.DataFrame:
    files = glob.glob(os.path.join(path, "*.csv"))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    if "processed_at" in df.columns:
        df["processed_at"] = pd.to_datetime(df["processed_at"], errors="coerce")

    if "administrative_dong" in df.columns:
        df["administrative_dong"] = (
            df["administrative_dong"].astype(str).map(REGION_NAME_MAP)
            .fillna(df["administrative_dong"].astype(str))
        )

    return df


@st.cache_data
def load_structured(path: str) -> pd.DataFrame:
    """SGI 정형 데이터 로딩 (region_code, time_id, SGI_score 등)"""
    files = glob.glob(os.path.join(path, "*.csv"))
    if not files:
        return pd.DataFrame()

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    # time_id(YYYYMM) → datetime (해당 월 1일)
    if "time_id" in df.columns:
        df["time_id"] = pd.to_datetime(df["time_id"], format="%Y%m", errors="coerce")

    # region_code → region_name (Unstructured와 매칭 위해)
    if "region_code" in df.columns:
        df["region_code"] = df["region_code"].astype(str)
        df["region_name"] = (
            df["region_code"].map(REGION_NAME_MAP)
            .fillna(df["region_code"])
        )
    else:
        df["region_name"] = "UNKNOWN"

    return df


# =========================================
# 워드클라우드 관련
# =========================================
def extract_keyword_freq(df: pd.DataFrame, col: str = "doc_keywords", top_n: int = 100):
    if col not in df.columns:
        return {}
    s = df[col].dropna().astype(str)
    s = s.str.replace(r"[\[\]\(\)\"']", "", regex=True)

    tokens = s.str.split("[,;]").explode().dropna()
    tokens = tokens.astype(str).str.strip()
    tokens = tokens[tokens != ""]
    tokens = tokens.apply(lambda x: x.split(":")[0].strip() if ":" in x else x)

    if tokens.empty:
        return {}
    return tokens.value_counts().head(top_n).to_dict()


def plot_wordcloud(freq_dict: dict):
    if not freq_dict:
        st.write("표시할 키워드 없음")
        return

    font_path = KOREAN_FONT_PATH if os.path.exists(KOREAN_FONT_PATH) else None

    wc = WordCloud(
        width=900,
        height=450,
        background_color="white",
        font_path=font_path,
    ).generate_from_frequencies(freq_dict)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)


# =========================================
# Unstructured 탭 (UGI 기간별 시각화)
# =========================================
def analysis_tab():
    df = load_analysis(ANALYSIS_DIR)
    if df.empty:
        st.error("analysis 디렉토리에 CSV 파일이 없습니다.")
        return

    st.title("Unstructured – UGI Trend")

    # 필터
    st.subheader("필터")

    dongs = sorted(df["administrative_dong"].dropna().unique().tolist())
    selected_dong = st.selectbox("행정동", dongs)

    filtered = df[df["administrative_dong"] == selected_dong].copy()
    filtered = filtered.dropna(subset=["UGI", "period_start"])

    if filtered.empty:
        st.write("선택한 조건에 해당하는 데이터가 없습니다.")
        return

    filtered = filtered.sort_values("period_start")

    # 최신 UGI
    latest_row = filtered.tail(1)
    latest_ugi = float(latest_row["UGI"].iloc[0])
    st.subheader("현재 UGI")
    st.markdown(f"### {latest_ugi:.3f}")

    # 기간별 UGI 라인차트
    st.subheader("기간별 UGI 추이")

    plot_df = filtered[["period_start", "UGI"]].dropna()

    if not plot_df.empty:
        chart = (
            alt.Chart(plot_df)
            .mark_line(point=True)
            .encode(
                x=alt.X(
                    "period_start:T",
                    axis=alt.Axis(
                        title="기간",
                        format="%Y\n%b",  # 예: 2024 / Dec
                        labelAngle=0,
                    ),
                ),
                y=alt.Y("UGI:Q", title="UGI"),
            ).interactive()
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.write("UGI 데이터가 없습니다.")


# =========================================
# Structured 탭 (SGI 기간별 시각화)
# =========================================
def structured_tab():
    df = load_structured(STRUCTURED_DIR)
    if df.empty:
        st.error("structured 디렉토리에 CSV 파일이 없습니다.")
        return

    st.title("Structured – SGI Trend")

    st.subheader("필터")

    regions = sorted(df["region_name"].dropna().unique().tolist())
    selected_region = st.selectbox("지역 선택", regions, key="structured_region")


    filtered = df[df["region_name"] == selected_region].copy()
    filtered = filtered.dropna(subset=["SGI_score", "time_id"])

    if filtered.empty:
        st.write("선택한 조건에 해당하는 데이터가 없습니다.")
        return

    filtered = filtered.sort_values("time_id")

    # 최신 SGI
    latest_row = filtered.tail(1)
    latest_sgi = float(latest_row["SGI_score"].iloc[0])
    st.subheader("현재 SGI")
    st.markdown(f"### {latest_sgi:.3f}")

    # 기간별 SGI 라인차트
    st.subheader("기간별 SGI 추이")

    plot_df = filtered[["time_id", "SGI_score"]].dropna()

    if not plot_df.empty:
        chart = (
            alt.Chart(plot_df)
            .mark_line(point=True)
            .encode(
                x=alt.X(
                    "time_id:T",
                    axis=alt.Axis(
                        title="기간",
                        format="%Y\n%b",
                        labelAngle=0,
                    ),
                ),
                y=alt.Y("SGI_score:Q", title="SGI Score"),
            ).interactive()
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.write("SGI 데이터가 없습니다.")


# =========================================
# Overall 탭 (UGI 0.6 + SGI 0.4 통합 지수)
# =========================================
def overall_tab():
    un_df = load_analysis(ANALYSIS_DIR)
    st_df = load_structured(STRUCTURED_DIR)

    if un_df.empty:
        st.error("analysis 디렉토리에 CSV 파일이 없습니다. (UGI)")
        return
    if st_df.empty:
        st.error("structured 디렉토리에 CSV 파일이 없습니다. (SGI)")
        return

    # 월 단위 키 생성
    un_df = un_df.dropna(subset=["UGI", "period_start", "administrative_dong"]).copy()
    st_df = st_df.dropna(subset=["SGI_score", "time_id", "region_name"]).copy()

    un_df["month"] = un_df["period_start"].dt.to_period("M").dt.to_timestamp()
    st_df["month"] = st_df["time_id"].dt.to_period("M").dt.to_timestamp()

    # 이름 기준으로 맞추기: administrative_dong ↔ region_name
    un_df = un_df.rename(columns={"administrative_dong": "region_name"})

    # 양쪽에 모두 존재하는 지역만 선택
    common_regions = sorted(
        set(un_df["region_name"].unique()) & set(st_df["region_name"].unique())
    )
    if not common_regions:
        st.error("UGI와 SGI가 동시에 존재하는 공통 지역이 없습니다.")
        return

    st.title("Overall – UGI(0.6) + SGI(0.4)")

    st.subheader("필터")
    selected_region = st.selectbox("지역 선택", common_regions, key="overall_region")


    un_f = un_df[un_df["region_name"] == selected_region]
    st_f = st_df[st_df["region_name"] == selected_region]

    if un_f.empty or st_f.empty:
        st.write("선택한 지역에 대해 UGI 또는 SGI 데이터가 없습니다.")
        return

    # 월 기준 inner join
    merged = pd.merge(
        un_f[["month", "UGI"]],
        st_f[["month", "SGI_score"]],
        on="month",
        how="inner",
    )

    if merged.empty:
        st.write("선택한 지역에 대해 겹치는 월 데이터가 없습니다.")
        return

    merged = merged.sort_values("month")

    # 통합 지수 계산: 0.6 * UGI + 0.4 * SGI
    merged["Overall_index"] = 0.6 * merged["UGI"] + 0.4 * merged["SGI_score"]

    # 최신 값 표시
    latest = merged.tail(1).iloc[0]
    st.subheader("현재 통합 지수")
    st.markdown(f"### {latest['Overall_index']:.3f}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption("최신 UGI")
        st.markdown(f"**{latest['UGI']:.3f}**")
    with col2:
        st.caption("최신 SGI")
        st.markdown(f"**{latest['SGI_score']:.3f}**")
    with col3:
        st.caption("최신 Overall")
        st.markdown(f"**{latest['Overall_index']:.3f}**")

    # 라인 차트: UGI, SGI, Overall 모두 표시
    st.subheader("기간별 지수 추이 (UGI / SGI / Overall)")

    plot_df = merged.rename(columns={"month": "date"})
    plot_long = plot_df.melt(
        id_vars="date",
        value_vars=["UGI", "SGI_score", "Overall_index"],
        var_name="metric",
        value_name="value",
    )

    chart = (
        alt.Chart(plot_long)
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "date:T",
                axis=alt.Axis(
                    title="기간",
                    format="%Y\n%b",
                    labelAngle=0,
                ),
            ),
            y=alt.Y("value:Q", title="지수 값"),
            color=alt.Color(
                "metric:N",
                title="지표",
                scale=alt.Scale(
                    domain=["UGI", "SGI_score", "Overall_index"],
                    range=["#1f77b4", "#ff7f0e", "#2ca02c"],
                ),
                legend=alt.Legend(labelExpr="datum.label == 'SGI_score' ? 'SGI' : datum.label")
            ),
            tooltip=[
                alt.Tooltip("date:T", title="날짜", format="%Y-%m"),
                alt.Tooltip("metric:N", title="지표"),
                alt.Tooltip("value:Q", title="값", format=".3f"),
            ],
        )
        .interactive()
    )

    st.altair_chart(chart, use_container_width=True)


# =========================================
# Realtime 탭
# =========================================
def realtime_tab():
    gold_df = load_realtime_gold(REALTIME_GOLD_DIR)
    proc_df = load_realtime_processed(REALTIME_PROCESSED_DIR)

    if gold_df.empty:
        st.error("realtime_gold 디렉토리에 CSV 파일이 없습니다.")
        return
    if proc_df.empty:
        st.error("realtime_processed 디렉토리에 CSV 파일이 없습니다.")
        return

    st.title("Realtime Blog Signals")

    # 필터
    st.subheader("필터")

    dongs = sorted(gold_df["administrative_dong"].dropna().unique())
    selected_dong = st.selectbox("행정동 선택", dongs)

    gold_filtered = gold_df[gold_df["administrative_dong"] == selected_dong]
    proc_filtered = proc_df[proc_df["administrative_dong"] == selected_dong]

    # 키워드
    st.subheader("주요 키워드 (doc_keywords 기준)")

    freq_dict = extract_keyword_freq(gold_filtered, "doc_keywords")
    plot_wordcloud(freq_dict)

    # 요약
    st.subheader("요약")

    col1, col2 = st.columns(2)

    with col1:
        st.caption("SENTIMENT")
        if not gold_filtered.empty and "sentiment_label" in gold_filtered.columns:
            dominant = gold_filtered["sentiment_label"].value_counts().idxmax()
            st.markdown(f"### {dominant}")
        else:
            st.markdown("데이터 없음")

    with col2:
        st.caption("POST DENSITY")
        if not proc_filtered.empty and "post_density" in proc_filtered.columns:
            if "processed_at" in proc_filtered.columns:
                latest = proc_filtered.sort_values("processed_at").tail(1)
            else:
                latest = proc_filtered.tail(1)
            val = int(round(float(latest["post_density"].iloc[0])))
            st.markdown(f"### {val}")
        else:
            st.markdown("데이터 없음")


# =========================================
# 메인 앱
# =========================================
def main():
    st.set_page_config(page_title="Blog Dashboard", layout="wide")

    tab_overall, tab_unstructured, tab_structured, tab_realtime = st.tabs(
        ["Overall", "Unstructured", "Structured", "Realtime"]
    )

    with tab_overall:
        overall_tab()

    with tab_unstructured:
        analysis_tab()

    with tab_structured:
        structured_tab()

    with tab_realtime:
        realtime_tab()


if __name__ == "__main__":
    main()
