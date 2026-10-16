# 대시보드: 노트북 한 화면(약 700px)에 스크롤 없이 표시
#   맨 위 : 제목, 현장, 기준 시각
#   1줄  : 현재 전력 | 15분 예상 수요(계산식 + ML) | ESS | 현장 연결
#   2줄  : 15분 수요 막대(최근 3시간, 예측 점 포함) | 전력 추이(최근 30분)
#   3줄  : 제어 이력 | 알림 | 오늘 ML 성적표
# ml.* 테이블은 ml/serve.py 가 만듦. 아직 없으면 ML 부분만 "대기"로 표시

import html
import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import altair as alt
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text

SITE = "plant1"
REFRESH = 5          # 화면 갱신 주기 (초)
OFFLINE_SEC = 30     # 30초 넘게 안 오면 끊김
GAP_SEC = 3          # 3초 넘게 빈 곳을 빈 구간으로 셈
WARN = 0.9           # 목표의 90% 이상이면 경고
SOC_MIN = 20
V_MIN = 342
V_MAX = 418
HZ_MIN = 59.8
HZ_MAX = 60.2
PF_MIN = 0.90
ML_OLD_SEC = 90      # ML 예측이 90초 넘게 안 들어오면 serve.py 멈춘 걸로 봄

CHART_H = 220        # 차트 높이
CHART_BOX_H = 290    # 차트 칸 높이 (제목 + 차트 + 여백)
LOG_BOX_H = 232      # 기록 칸 높이
ROWS = 6             # 기록 표에 보여줄 줄 수

GREEN = "#2e9e5b"
AMBER = "#d39b1e"
RED = "#d64545"
GRAY = "#7a7f87"
BLUE = "#4c78a8"
PURPLE = "#9467bd"   # ML

ACTIONS = {"shed": "부하 차단", "restore": "부하 복귀", "ess_on": "ESS 방전", "ess_off": "ESS 정지"}
KINDS = {
    "no_data": "수신 끊김", "data_resumed": "수신 재개", "edge_offline": "현장 오프라인",
    "delay": "수신 지연", "duplicate": "중복 수신", "kwh_backward": "전력량 역행",
    "voltage": "전압 이탈", "frequency": "주파수 이탈", "power_factor": "역률 저하",
    "peak_exceeded": "피크 초과",
}

st.set_page_config(page_title="공장 전력 모니터링", layout="wide")

st.markdown("""
<style>
[data-testid="stHeader"] {display: none;}
.block-container, [data-testid="stMainBlockContainer"] {padding: 12px 20px 0 20px; max-width: 100%;}
[data-testid="stVerticalBlock"] {gap: 10px;}

/* 직접 만든 HTML 아래 여백 (안 주면 다음 칸이랑 겹침) */
.spacer {padding-bottom: 14px;}

.topbar {display: flex; justify-content: space-between; align-items: baseline;}
.topbar .title {font-size: 19px; font-weight: 700;}
.topbar .meta {font-size: 13px; opacity: .7;}

.tile {border: 1px solid rgba(128,128,128,.35); border-radius: 6px; padding: 10px 14px;
       height: 136px; box-sizing: border-box;}
.t-label {font-size: 13px; opacity: .8; display: flex; justify-content: space-between; align-items: center;}
.t-value {font-size: 28px; font-weight: 700; line-height: 1.2; margin-top: 4px;}
.t-sub {font-size: 13px; opacity: .75; margin-top: 6px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}
.t-extra {font-size: 13px; margin-top: 3px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}
.t-extra b {color: #9467bd;}
.badge {font-size: 12px; font-weight: 700; padding: 1px 9px; border-radius: 10px; color: #fff;}
.bar {height: 6px; background: rgba(128,128,128,.25); border-radius: 3px; margin-top: 6px; overflow: hidden;}
.bar > div {height: 100%;}

.panel-title {font-size: 14px; font-weight: 600; margin: 0 0 2px 0;}
.panel-sub {font-size: 12px; opacity: .65; font-weight: 400; margin-left: 6px;}

table.log {width: 100%; border-collapse: collapse; font-size: 13px; table-layout: fixed; margin: 0;}
table.log th, table.log td {padding: 3px 6px !important; line-height: 1.4; border: none !important;
                            white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}
table.log th {text-align: left; font-weight: 600; opacity: .7;
              border-bottom: 1px solid rgba(128,128,128,.35) !important;}
table.log td {border-bottom: 1px solid rgba(128,128,128,.15) !important;}
.empty {font-size: 13px; opacity: .6; padding: 8px 6px;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def engine():
    return create_engine(os.environ["DB_URL"], pool_pre_ping=True)


def now():
    return datetime.now(ZoneInfo("Asia/Seoul")).replace(tzinfo=None)


def query(sql, **params):
    with engine().connect() as conn:
        return pd.read_sql(text(sql), conn, params=params)


def badge(label, color):
    return f'<span class="badge" style="background:{color}">{label}</span>'


# 상단 타일 HTML. bar 를 주면 값 아래에 막대 표시
def tile(label, value, sub, badge_html="", bar=None, bar_color=GRAY, extra=""):
    bar_html = ""
    if bar is not None:
        width = max(0.0, min(bar, 1.0)) * 100
        bar_html = f'<div class="bar"><div style="width:{width:.0f}%;background:{bar_color}"></div></div>'
    return (f'<div class="spacer"><div class="tile notranslate" translate="no">'
            f'<div class="t-label"><span>{label}</span>{badge_html}</div>'
            f'<div class="t-value">{value}</div>{bar_html}'
            f'<div class="t-sub">{sub}</div>'
            f'<div class="t-extra">{extra}</div></div></div>')


def panel_title(title, sub=""):
    st.markdown(f'<p class="panel-title notranslate" translate="no">{title}'
                f'<span class="panel-sub">{sub}</span></p>', unsafe_allow_html=True)


def empty(msg):
    st.markdown(f'<div class="empty">{msg}</div>', unsafe_allow_html=True)


# 기록 표 HTML. widths = 칸 너비 (%)
def table(headers, rows, widths):
    if not rows:
        return '<div class="empty">기록 없음</div>'

    cols = ""
    for w in widths:
        cols += f'<col style="width:{w}">'
    head = ""
    for h in headers:
        head += f"<th>{h}</th>"
    body = ""
    for row in rows:
        body += "<tr>"
        for v in row:
            body += f"<td>{html.escape(str(v))}</td>"
        body += "</tr>"
    return (f'<table class="log notranslate" translate="no"><colgroup>{cols}</colgroup>'
            f'<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>')


def show_tiles(t, r, d, has_ml):
    cols = st.columns(4)

    # 1. 현재 전력 + 전력 품질
    ok = V_MIN <= r["voltage"] <= V_MAX and HZ_MIN <= r["hz"] <= HZ_MAX and r["pf"] >= PF_MIN
    cols[0].markdown(tile(
        "현재 전력", f"{r['kw']:,.0f} kW",
        f"{r['voltage']:.0f} V · 역률 {r['pf']:.2f} · {r['hz']:.2f} Hz",
        badge("품질 정상", GREEN) if ok else badge("품질 이상", RED),
    ), unsafe_allow_html=True)

    # 2. 15분 예상 수요 (계산식) + 그 아래 ML 예측 한 줄
    if d is None:
        cols[1].markdown(tile("15분 예상 수요", "-", "구간 데이터 없음"), unsafe_allow_html=True)
    else:
        ratio = d["projected_kw"] / d["target_kw"]
        if ratio >= 1:
            state, color = "위험", RED
        elif ratio >= WARN:
            state, color = "경고", AMBER
        else:
            state, color = "정상", GREEN
        ws = d["window_start"]

        # 이번 구간의 최신 ML 예측 (90초 넘게 오래된 건 안 씀)
        ml = None
        if has_ml:
            f = query("SELECT ts, ml_kw FROM ml.forecast WHERE window_start = :ws ORDER BY ts DESC LIMIT 1", ws=ws)
            if not f.empty and (pd.Timestamp(t) - f.iloc[0]["ts"]).total_seconds() <= ML_OLD_SEC:
                ml = f.iloc[0]["ml_kw"]

        if not has_ml:
            ml_text = '<span style="opacity:.6">ML 예측 대기 · serve.py 미실행</span>'
        elif ml is None:
            ml_text = '<span style="opacity:.6">ML 예측 대기 · 구간 시작 2분 후부터</span>'
        else:
            ml_text = (f'ML 예측 <b>{ml:,.0f} kW</b>'
                       f'<span style="opacity:.75"> · 목표 대비 {ml / d["target_kw"]:.0%}'
                       f' · 계산식과 {ml - d["projected_kw"]:+,.0f} kW</span>')

        cols[1].markdown(tile(
            "15분 예상 수요 (계산식)", f"{d['projected_kw']:,.0f} kW",
            f"목표 {d['target_kw']:,.0f} kW 대비 {ratio:.0%} · 구간 {ws:%H:%M}~{ws + timedelta(minutes=15):%H:%M}",
            badge(state, color), ratio, color, ml_text,
        ), unsafe_allow_html=True)

    # 3. ESS
    cols[2].markdown(tile(
        "ESS 충전율", f"{r['soc']:.0f} %",
        f"방전 {r['ess_kw']:.0f} kW · 방전 하한 {SOC_MIN}%",
        badge("방전 중", AMBER) if r["ess_kw"] > 0 else badge("대기", GRAY),
        r["soc"] / 100, RED if r["soc"] <= SOC_MIN else GREEN,
    ), unsafe_allow_html=True)

    # 4. 현장 연결 + 최근 1시간 빈 구간, 재전송 건수
    lag = (pd.Timestamp(t) - r["ts"]).total_seconds()
    h = query("""
        SELECT count(*) FILTER (WHERE buffered) AS resent,
               coalesce(sum(gap) FILTER (WHERE gap > :g), 0) AS gap_sec
        FROM (
            SELECT buffered,
                   extract(epoch FROM ts - lag(ts) OVER (ORDER BY ts)) AS gap
            FROM raw.meter_reading
            WHERE ts >= :s
        ) x""", s=t - timedelta(hours=1), g=GAP_SEC).iloc[0]
    online = lag < OFFLINE_SEC
    cols[3].markdown(tile(
        "현장 연결", "정상" if online else "끊김",
        f"{lag:.0f}초 전 수신 · 1시간 빈 구간 {float(h['gap_sec']):.0f}초 · 재전송 {int(h['resent'])}건",
        badge("ONLINE", GREEN) if online else badge("OFFLINE", RED),
    ), unsafe_allow_html=True)


# 15분 수요 막대 (최근 3시간). 흐린 막대 = 진행 중 구간
# 점 = 그 구간 동안 30초마다 낸 예측의 평균 (ML 성적표 오차와 같은 기준)
def show_demand_chart(t, target, has_ml):
    sub = "최근 3시간 · 빨강 = 목표 초과 · 흐림 = 진행 중 · 점선 = 목표"
    if has_ml:
        sub += (f' · <span style="color:{PURPLE}">●</span> ML'
                f' · <span style="color:{GRAY}">◆</span> 계산식 (구간 예측 평균)')
    panel_title("15분 수요", sub)

    since = t - timedelta(hours=3)
    df = query("""
        SELECT window_start,
               CASE WHEN final THEN avg_kw ELSE projected_kw END AS kw,
               target_kw, final
        FROM ops.demand_15min
        WHERE window_start >= :s
        ORDER BY window_start""", s=since)
    if df.empty:
        empty("데이터 없음")
        return
    df["구간"] = df["window_start"].dt.strftime("%H:%M")

    if has_ml:
        pred = query("""
            SELECT window_start, avg(formula_kw) AS formula_kw, avg(ml_kw) AS ml_kw
            FROM ml.forecast
            WHERE window_start >= :s
            GROUP BY window_start""", s=since)
        df = df.merge(pred, on="window_start", how="left")
    else:
        df["formula_kw"] = float("nan")
        df["ml_kw"] = float("nan")

    # 목표선, 예측 점이 위로 안 넘치게 y축을 10% 크게
    y_max = max(df[["kw", "formula_kw", "ml_kw"]].astype(float).max().max(), target or 0) * 1.1
    x = alt.X("구간:O", title=None, axis=alt.Axis(labelAngle=0))
    y_scale = alt.Scale(domain=[0, y_max])
    tooltip = [alt.Tooltip("구간:O"),
               alt.Tooltip("kw:Q", format=",.0f", title="실제 kW"),
               alt.Tooltip("formula_kw:Q", format=",.0f", title="계산식 예측"),
               alt.Tooltip("ml_kw:Q", format=",.0f", title="ML 예측")]

    base = alt.Chart(df)
    chart = base.mark_bar(size=22).encode(
        x=x,
        y=alt.Y("kw:Q", title="kW", scale=y_scale),
        color=alt.condition("datum.kw >= datum.target_kw", alt.value(RED), alt.value(BLUE)),
        opacity=alt.condition("datum.final", alt.value(1.0), alt.value(0.45)),
        tooltip=tooltip,
    )
    if target is not None:
        chart = chart + alt.Chart(pd.DataFrame({"y": [target]})).mark_rule(strokeDash=[5, 4], color=GRAY).encode(y="y:Q")
    if df["ml_kw"].notna().any():
        formula_pts = base.transform_filter("isValid(datum.formula_kw)").mark_point(
            shape="diamond", size=70, filled=True, color=GRAY, opacity=1,
        ).encode(x=x, y=alt.Y("formula_kw:Q", scale=y_scale), tooltip=tooltip)
        ml_pts = base.transform_filter("isValid(datum.ml_kw)").mark_point(
            shape="circle", size=70, filled=True, color=PURPLE, opacity=1,
        ).encode(x=x, y=alt.Y("ml_kw:Q", scale=y_scale), tooltip=tooltip)
        chart = chart + formula_pts + ml_pts

    chart = chart.properties(height=CHART_H).configure_axis(labelFontSize=12, titleFontSize=12)
    st.altair_chart(chart, use_container_width=True)


def show_power_chart(t, target):
    panel_title("전력 추이", "최근 30분 · 점선 = 목표")

    df = query("SELECT ts, kw FROM raw.meter_reading WHERE ts >= :s ORDER BY ts", s=t - timedelta(minutes=30))
    if df.empty:
        empty("데이터 없음")
        return

    y_min = df["kw"].min() * 0.95
    y_max = max(df["kw"].max(), target or 0) * 1.03

    chart = alt.Chart(df).mark_line(strokeWidth=1.4).encode(
        x=alt.X("ts:T", title=None, axis=alt.Axis(format="%H:%M", tickCount=6)),
        y=alt.Y("kw:Q", title="kW", scale=alt.Scale(domain=[y_min, y_max])),
        tooltip=[alt.Tooltip("ts:T", format="%H:%M:%S", title="시각"),
                 alt.Tooltip("kw:Q", format=",.0f", title="kW")],
    )
    if target is not None:
        chart = chart + alt.Chart(pd.DataFrame({"y": [target]})).mark_rule(strokeDash=[5, 4], color=GRAY).encode(y="y:Q")

    chart = chart.properties(height=CHART_H).configure_axis(labelFontSize=12, titleFontSize=12)
    st.altair_chart(chart, use_container_width=True)


def show_control_log():
    panel_title("제어 이력", f"최근 {ROWS}건")

    df = query("SELECT ts, action, target, reason, projected_kw FROM ops.control_event ORDER BY ts DESC LIMIT :n",
               n=ROWS)
    rows = []
    for _, e in df.iterrows():
        rows.append([f"{e['ts']:%H:%M:%S}", ACTIONS.get(e["action"], e["action"]), e["target"],
                     e["reason"], f"{e['projected_kw']:,.0f}"])
    st.markdown(table(["시각", "동작", "대상", "사유", "예상 kW"], rows, ["15%", "17%", "13%", "40%", "15%"]),
                unsafe_allow_html=True)


def show_alerts():
    panel_title("알림", f"최근 {ROWS}건")

    df = query("SELECT ts, kind, message FROM ops.alert ORDER BY ts DESC LIMIT :n", n=ROWS)
    rows = []
    for _, a in df.iterrows():
        rows.append([f"{a['ts']:%H:%M:%S}", KINDS.get(a["kind"], a["kind"]), a["message"]])
    st.markdown(table(["시각", "종류", "내용"], rows, ["15%", "20%", "65%"]), unsafe_allow_html=True)


# 오늘 마감된 구간들의 계산식 vs ML 오차
def show_ml_score(t, has_ml):
    panel_title("ML 성적표", "오늘 · 확정 구간 기준")
    if not has_ml:
        empty("ML 서빙 대기 · ml/serve.py 실행 필요")
        return

    s = query("""
        SELECT count(*) AS n,
               avg(formula_mae) AS formula_mae,
               avg(ml_mae) AS ml_mae,
               count(*) FILTER (WHERE ml_mae < formula_mae) AS ml_wins
        FROM ml.window_eval
        WHERE window_start >= :s""", s=t.replace(hour=0, minute=0, second=0, microsecond=0)).iloc[0]
    n = int(s["n"])
    if n == 0:
        empty("오늘 평가된 구간 없음 · 첫 구간 마감 후 표시")
        return

    f_mae = float(s["formula_mae"])
    m_mae = float(s["ml_mae"])
    wins = int(s["ml_wins"])
    gain = (f_mae - m_mae) / f_mae if f_mae > 0 else 0.0

    last = query("SELECT window_start, formula_mae, ml_mae FROM ml.window_eval ORDER BY window_start DESC LIMIT 1").iloc[0]
    last_win = "ML" if last["ml_mae"] < last["formula_mae"] else "계산식"

    rows = [
        ["평가 구간", f"{n}개"],
        ["계산식 평균 오차", f"{f_mae:,.1f} kW"],
        ["ML 평균 오차", f"{m_mae:,.1f} kW"],
        ["ML 오차 개선", f"{gain:+.1%}"],
        ["ML 승률", f"{wins / n:.0%} ({wins}/{n})"],
        ["최근 구간", f"{last['window_start']:%H:%M} · {last_win} 승"],
    ]
    st.markdown(table(["항목", "값"], rows, ["50%", "50%"]), unsafe_allow_html=True)


@st.fragment(run_every=REFRESH)
def live():
    t = now()

    st.markdown(f'<div class="spacer"><div class="topbar notranslate" translate="no">'
                f'<span class="title">공장 전력 모니터링 · 피크 제어</span>'
                f'<span class="meta">{SITE} · {t:%Y-%m-%d %H:%M:%S} 기준 · {REFRESH}초 갱신</span></div></div>',
                unsafe_allow_html=True)

    last = query("SELECT * FROM raw.meter_reading ORDER BY ts DESC LIMIT 1")
    if last.empty:
        st.info("아직 수신된 데이터가 없습니다.")
        return
    r = last.iloc[0]

    dem = query("SELECT * FROM ops.demand_15min ORDER BY window_start DESC LIMIT 1")
    d = None
    target = None
    if not dem.empty:
        d = dem.iloc[0]
        target = float(d["target_kw"])

    # ml/serve.py 가 한 번이라도 돌아서 ml 테이블이 있는지
    has_ml = bool(query("SELECT to_regclass('ml.forecast') IS NOT NULL "
                        "AND to_regclass('ml.window_eval') IS NOT NULL AS ok").iloc[0]["ok"])

    show_tiles(t, r, d, has_ml)

    # 칸 높이를 고정해서 좌우 높이를 맞추고 내용이 밑으로 안 넘치게
    cols = st.columns([3, 2])
    with cols[0].container(border=True, height=CHART_BOX_H):
        show_demand_chart(t, target, has_ml)
    with cols[1].container(border=True, height=CHART_BOX_H):
        show_power_chart(t, target)

    cols = st.columns([4, 4, 3])
    with cols[0].container(border=True, height=LOG_BOX_H):
        show_control_log()
    with cols[1].container(border=True, height=LOG_BOX_H):
        show_alerts()
    with cols[2].container(border=True, height=LOG_BOX_H):
        show_ml_score(t, has_ml)


live()
