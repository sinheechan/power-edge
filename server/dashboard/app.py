# 대시보드: 관제실 모니터 한 화면(1600x900 기준)에 스크롤 없이 표시
#   왼쪽 : 메뉴 (모니터링 / 이력 조회 / 기준값)
#   모니터링
#     맨 위 : 제목, 현장, 연결 상태, 기준 시각
#     1줄  : 핵심 수치 밴드 (현재 전력 | 15분 예상 | ML 예측 | 목표 | ESS | 구간 잔여)
#     2줄  : 상태 배너 (지금 상태 + 최근 조치)
#     3줄  : 전력 추이(30분) | 15분 수요(3시간) | 전력 품질·통신 표
#     4줄  : 실시간 계측 데이터 표 | 이벤트 로그 표 | ML 성적표
#   이력 조회 : 날짜별 15분 구간 표, 이벤트 표, CSV 내려받기
#   기준값   : 경보 기준, 차단 부하 순서
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
DELAY_SEC = 5        # 수신 지연 경고 기준
GAP_SEC = 3          # 3초 넘게 빈 곳을 빈 구간으로 셈
WARN = 0.9           # 목표의 90% 이상이면 경고
SOC_MIN = 20
V_MIN = 342
V_MAX = 418
HZ_MIN = 59.8
HZ_MAX = 60.2
PF_MIN = 0.90
ML_OLD_SEC = 90      # ML 예측이 90초 넘게 안 들어오면 serve.py 멈춘 걸로 봄

# 차단 부하 (edge/config.py 의 LOAD_KW, SHED_ORDER 와 같게 유지)
LOADS = [("load3", 100), ("load2", 120), ("load1", 150)]   # 차단 순서대로
ESS_KW = 200
SAFE = 0.8           # 목표의 80% 미만이면 복귀

CHART_H = 268        # 차트 높이
CHART_BOX_H = 350    # 차트 칸 높이 (제목 + 차트 + 여백)
LOG_BOX_H = 334      # 표 칸 높이
ROWS = 9             # 표에 보여줄 줄 수

# 색 (어두운 관제 화면 기준)
BG = "#0d1117"
PANEL = "#151b24"
LINE = "#263041"
TEXT = "#e6edf3"
MUTED = "#8b97a8"
GREEN = "#2fb170"
AMBER = "#e0a526"
RED = "#ef5350"
GRAY = "#9aa4b2"
BLUE = "#4f8fe8"
PURPLE = "#a78bfa"   # ML

ACTIONS = {"shed": "부하 차단", "restore": "부하 복귀", "ess_on": "ESS 방전", "ess_off": "ESS 정지"}
KINDS = {
    "no_data": "수신 끊김", "data_resumed": "수신 재개", "edge_offline": "현장 오프라인",
    "delay": "수신 지연", "duplicate": "중복 수신", "kwh_backward": "전력량 역행",
    "voltage": "전압 이탈", "frequency": "주파수 이탈", "power_factor": "역률 저하",
    "peak_exceeded": "피크 초과",
}
# 알림 종류별 심각도 색
SEVERITY = {
    "no_data": RED, "edge_offline": RED, "peak_exceeded": RED, "voltage": RED, "frequency": RED,
    "delay": AMBER, "duplicate": AMBER, "kwh_backward": AMBER, "power_factor": AMBER,
    "data_resumed": GREEN,
}

PAGES = {"monitor": "모니터링", "history": "이력 조회", "settings": "기준값"}
ICONS = {
    "monitor": '<path d="M3 12h4l3-8 4 16 3-8h4"/>',
    "history": '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 3"/>',
    "settings": '<path d="M4 6h10M18 6h2M4 12h4M12 12h8M4 18h12M20 18h0"/>'
                '<circle cx="16" cy="6" r="2"/><circle cx="10" cy="12" r="2"/><circle cx="18" cy="18" r="2"/>',
}

st.set_page_config(page_title="공장 전력 모니터링", layout="wide")

st.markdown(f"""
<style>
[data-testid="stHeader"], [data-testid="stToolbar"] {{display: none;}}
.stApp {{background: {BG};}}
.block-container, [data-testid="stMainBlockContainer"] {{padding: 10px 18px 0 96px; max-width: 100%;}}
[data-testid="stVerticalBlock"] {{gap: 8px;}}
[class*="st-key-panel_"] {{background: {PANEL}; border-color: {LINE} !important; border-radius: 6px;}}

/* 직접 만든 HTML 아래 여백 (안 주면 다음 칸이랑 겹침) */
.spacer {{padding-bottom: 10px;}}

/* 왼쪽 메뉴 */
.rail {{position: fixed; top: 0; left: 0; bottom: 0; width: 78px; background: #0a0e14;
        border-right: 1px solid {LINE}; z-index: 1000; display: flex; flex-direction: column;
        align-items: center; padding-top: 12px; gap: 4px;}}
.rail .logo {{font-size: 11px; font-weight: 800; color: {BLUE}; letter-spacing: .5px; text-align: center;
              line-height: 1.15; margin-bottom: 14px; border: 2px solid {BLUE}; border-radius: 8px; padding: 6px 4px;}}
.rail a {{width: 78px; padding: 10px 0 8px; display: flex; flex-direction: column; align-items: center;
          gap: 4px; color: {MUTED} !important; text-decoration: none !important; font-size: 11px;
          border-left: 3px solid transparent;}}
.rail a:hover {{color: {TEXT} !important; background: #121822;}}
.rail a.on {{color: {TEXT} !important; border-left-color: {GREEN}; background: #121822;}}
.rail svg {{width: 24px; height: 24px; fill: none; stroke: currentColor; stroke-width: 1.8;
            stroke-linecap: round; stroke-linejoin: round;}}

/* 맨 위 줄 */
.topbar {{display: flex; justify-content: space-between; align-items: center;}}
.topbar .title {{font-size: 18px; font-weight: 700; color: {TEXT};}}
.topbar .title small {{font-size: 12px; font-weight: 400; color: {MUTED}; margin-left: 8px;}}
.topbar .meta {{font-size: 13px; color: {MUTED}; display: flex; gap: 14px; align-items: center;}}
.topbar .meta b {{color: {TEXT}; font-variant-numeric: tabular-nums;}}

/* 핵심 수치 밴드 */
.band {{display: grid; grid-template-columns: repeat(6, 1fr); background: linear-gradient(180deg, #13407a, #0f3463);
        border: 1px solid #1d4f8f; border-radius: 6px;}}
.kpi {{padding: 8px 14px 9px; border-right: 1px solid rgba(255,255,255,.12); min-width: 0;}}
.kpi:last-child {{border-right: none;}}
.k-label {{font-size: 12px; color: #b9cbe6; display: flex; justify-content: space-between; align-items: center;}}
.k-value {{font-size: 30px; font-weight: 700; color: #fff; line-height: 1.15; font-style: italic;
           font-variant-numeric: tabular-nums; white-space: nowrap;}}
.k-value .unit {{font-size: 14px; font-weight: 500; font-style: normal; color: #b9cbe6; margin-left: 4px;}}
.k-sub {{font-size: 12px; color: #b9cbe6; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}}

/* 상태 배너 */
.banner {{display: flex; align-items: center; gap: 12px; padding: 7px 12px; border-radius: 6px;
          border: 1px solid; font-size: 13px; color: {TEXT};}}
.banner .lv {{font-weight: 800; font-size: 13px; padding: 2px 12px; border-radius: 4px; color: #fff;}}
.banner .msg {{flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}}
.banner .act {{color: {MUTED}; white-space: nowrap;}}
.banner .act b {{color: {TEXT}; font-weight: 600;}}

/* 칸 제목: 왼쪽 제목, 오른쪽 요약 칩 (샘플의 Min/Max 자리) */
.phead {{display: flex; justify-content: space-between; align-items: center; gap: 8px;
         border-bottom: 1px solid {LINE}; padding-bottom: 5px; margin-bottom: 4px;}}
.ptitle {{font-size: 13px; font-weight: 700; color: {TEXT}; white-space: nowrap;}}
.ptitle small {{font-weight: 400; color: {MUTED}; margin-left: 6px; font-size: 11px;}}
.chips {{display: flex; gap: 5px; flex-wrap: nowrap;}}
.chip {{font-size: 11px; color: {MUTED}; background: #0e131b; border: 1px solid {LINE}; border-radius: 3px;
        padding: 1px 6px; white-space: nowrap;}}
.chip b {{color: {TEXT}; font-weight: 600; font-variant-numeric: tabular-nums;}}
.legend {{font-size: 11px; color: {MUTED}; margin: 0 0 8px 2px;}}
.legend span {{margin-right: 10px;}}

.badge {{font-size: 11px; font-weight: 700; padding: 1px 8px; border-radius: 3px; color: #fff; white-space: nowrap;}}

table.log {{width: 100%; border-collapse: collapse; font-size: 12.5px; table-layout: fixed; margin: 0; color: {TEXT};}}
table.log th, table.log td {{padding: 4px 6px !important; line-height: 1.35; border: none !important;
                            white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}}
table.log th {{text-align: left; font-weight: 600; color: {MUTED}; background: #10161f;
              border-bottom: 1px solid {LINE} !important;}}
table.log td {{border-bottom: 1px solid #1c2431 !important; font-variant-numeric: tabular-nums;}}
table.log td.n, table.log th.n {{text-align: right;}}
table.log td.bad {{color: {RED}; font-weight: 700;}}
table.log tr:first-child td {{background: rgba(79,143,232,.07);}}
.empty {{font-size: 13px; color: {MUTED}; padding: 8px 6px;}}
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


def has_ml_tables():
    # ml/serve.py 가 한 번이라도 돌아서 ml 테이블이 있는지
    return bool(query("SELECT to_regclass('ml.forecast') IS NOT NULL "
                      "AND to_regclass('ml.window_eval') IS NOT NULL AS ok").iloc[0]["ok"])


def badge(label, color):
    return f'<span class="badge" style="background:{color}">{label}</span>'


def chip(label, value):
    return f'<span class="chip">{label} <b>{value}</b></span>'


def demand_state(ratio):
    if ratio >= 1:
        return "위험", RED
    if ratio >= WARN:
        return "경고", AMBER
    return "정상", GREEN


def panel_head(title, sub="", chips=""):
    st.markdown(f'<div class="phead notranslate" translate="no"><span class="ptitle">{title}<small>{sub}</small></span>'
                f'<span class="chips">{chips}</span></div>', unsafe_allow_html=True)


def empty(msg):
    st.markdown(f'<div class="empty">{msg}</div>', unsafe_allow_html=True)


# 표 HTML. widths = 칸 너비 (%), num = 오른쪽 정렬할 칸 번호
# 칸 값이 (값, 표시방식) 튜플이면: "bad" = 빨간 글씨, "html" = 이스케이프 없이 그대로
def table(headers, rows, widths, num=()):
    if not rows:
        return '<div class="empty">기록 없음</div>'

    cols = ""
    for w in widths:
        cols += f'<col style="width:{w}">'
    head = ""
    for i, h in enumerate(headers):
        head += f'<th class="n">{h}</th>' if i in num else f"<th>{h}</th>"
    body = ""
    for row in rows:
        body += "<tr>"
        for i, v in enumerate(row):
            mode = ""
            if isinstance(v, tuple):
                v, mode = v
            cls = []
            if i in num:
                cls.append("n")
            if mode == "bad":
                cls.append("bad")
            cell = v if mode == "html" else html.escape(str(v))
            body += f'<td class="{" ".join(cls)}">{cell}</td>'
        body += "</tr>"
    return (f'<table class="log notranslate" translate="no"><colgroup>{cols}</colgroup>'
            f'<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>')


# 어두운 화면용 차트 스타일
def dark(chart, height=CHART_H):
    return (chart.properties(height=height)
            .configure(background="transparent")
            .configure_view(stroke=None)
            .configure_axis(labelColor=MUTED, titleColor=MUTED, gridColor="#1f2836", domainColor=LINE,
                            tickColor=LINE, labelFontSize=11, titleFontSize=11))


def rail(page):
    links = ""
    for key, name in PAGES.items():
        on = "on" if key == page else ""
        links += (f'<a class="{on}" href="?page={key}" target="_self">'
                  f'<svg viewBox="0 0 24 24">{ICONS[key]}</svg>{name}</a>')
    st.markdown(f'<div class="rail notranslate" translate="no"><div class="logo">POWER<br>EDGE</div>{links}</div>',
                unsafe_allow_html=True)


def topbar(t, title, online=None, lag=None):
    conn = ""
    if online is not None:
        conn = (badge("ONLINE", GREEN) if online else badge("OFFLINE", RED)) + f" <span>{lag:.0f}초 전 수신</span>"
    st.markdown(f'<div class="spacer"><div class="topbar notranslate" translate="no">'
                f'<span class="title">{title}<small>공장 전력 피크 제어 · 현장 {SITE}</small></span>'
                f'<span class="meta">{conn}<span>{REFRESH}초 갱신</span><b>{t:%Y-%m-%d %H:%M:%S}</b></span>'
                f'</div></div>', unsafe_allow_html=True)


# 테두리 있는 칸. key 로 붙는 클래스(st-key-panel_*)에 배경색을 줌
def box(col, name, height="content"):
    return col.container(border=True, height=height, key=f"panel_{name}")


def kpi(label, value, unit, sub, badge_html="", color="#fff"):
    return (f'<div class="kpi"><div class="k-label"><span>{label}</span>{badge_html}</div>'
            f'<div class="k-value" style="color:{color}">{value}<span class="unit">{unit}</span></div>'
            f'<div class="k-sub">{sub}</div></div>')


# ── 모니터링 ─────────────────────────────────────────

# 이번 구간의 최신 ML 예측 (90초 넘게 오래된 건 안 씀)
def latest_ml(t, ws, has_ml):
    if not has_ml:
        return None
    f = query("SELECT ts, ml_kw FROM ml.forecast WHERE window_start = :ws ORDER BY ts DESC LIMIT 1", ws=ws)
    if not f.empty and (pd.Timestamp(t) - f.iloc[0]["ts"]).total_seconds() <= ML_OLD_SEC:
        return float(f.iloc[0]["ml_kw"])
    return None


def show_band(t, r, d, ml, has_ml):
    cells = kpi("현재 전력", f"{r['kw']:,.0f}", "kW", f"전류 {r['current']:,.0f} A · {r['voltage']:.0f} V")

    if d is None:
        cells += kpi("15분 예상 수요 (계산식)", "-", "kW", "구간 데이터 없음")
        cells += kpi("ML 예측", "-", "kW", "구간 데이터 없음")
        cells += kpi("목표 최대수요", "-", "kW", "")
        remain = "-"
        window = ""
    else:
        target = d["target_kw"]
        ratio = d["projected_kw"] / target
        state, color = demand_state(ratio)
        cells += kpi("15분 예상 수요 (계산식)", f"{d['projected_kw']:,.0f}", "kW", f"목표 대비 {ratio:.0%}",
                     badge(state, color), color if state != "정상" else "#fff")

        if ml is not None:
            ml_state, ml_color = demand_state(ml / target)
            cells += kpi("ML 예측 (LightGBM)", f"{ml:,.0f}", "kW",
                         f"목표 대비 {ml / target:.0%} · 계산식과 {ml - d['projected_kw']:+,.0f}",
                         badge(ml_state, ml_color), ml_color if ml_state != "정상" else "#fff")
        else:
            cells += kpi("ML 예측 (LightGBM)", "-", "kW",
                         "serve.py 미실행" if not has_ml else "구간 시작 2분 후부터")

        cells += kpi("목표 최대수요", f"{target:,.0f}", "kW", f"경고 기준 {target * WARN:,.0f} kW ({WARN:.0%})")

        end = d["window_start"] + timedelta(minutes=15)
        left = max(0, int((end - pd.Timestamp(t)).total_seconds()))
        remain = f"{left // 60:02d}:{left % 60:02d}"
        window = f"{d['window_start']:%H:%M} ~ {end:%H:%M} 구간"

    soc_color = RED if r["soc"] <= SOC_MIN else "#fff"
    ess_badge = badge("방전 중", AMBER) if r["ess_kw"] > 0 else badge("대기", "#5b6675")
    cells += kpi("ESS 충전율", f"{r['soc']:.0f}", "%", f"방전 {r['ess_kw']:.0f} / {ESS_KW} kW · 하한 {SOC_MIN}%",
                 ess_badge, soc_color)
    cells += kpi("구간 잔여 시간", remain, "", window)

    st.markdown(f'<div class="spacer"><div class="band notranslate" translate="no">{cells}</div></div>',
                unsafe_allow_html=True)


# 지금 상태를 한 줄로. 가장 나쁜 항목 기준
def show_banner(r, d, ml, online, quality_ok):
    level, color = "정상", GREEN
    msg = "15분 수요가 목표 안에서 유지되고 있습니다."

    if d is not None:
        worst = max(d["projected_kw"], ml or 0)
        state, c = demand_state(worst / d["target_kw"])
        who = "ML 예측" if ml is not None and ml >= d["projected_kw"] else "계산식 예상"
        if state != "정상":
            level, color = state, c
            msg = f"{who} {worst:,.0f} kW · 목표 {d['target_kw']:,.0f} kW 의 {worst / d['target_kw']:.0%}"
            if r["ess_kw"] > 0:
                msg += f" · ESS {r['ess_kw']:.0f} kW 방전 중"
    if not quality_ok and level == "정상":
        level, color, msg = "경고", AMBER, "전력 품질(전압·주파수·역률) 기준 이탈 항목이 있습니다."
    if not online:
        level, color, msg = "통신 끊김", RED, "현장 데이터 수신이 끊겼습니다. 엣지 버퍼에 저장 후 복구 시 재전송됩니다."

    ev = query("SELECT ts, action, target FROM ops.control_event ORDER BY ts DESC LIMIT 1")
    act = "최근 조치 없음"
    if not ev.empty:
        e = ev.iloc[0]
        act = f"최근 조치 <b>{e['ts']:%H:%M:%S} {ACTIONS.get(e['action'], e['action'])} ({html.escape(str(e['target']).replace('ess', 'ESS'))})</b>"

    st.markdown(f'<div class="spacer"><div class="banner notranslate" translate="no" style="border-color:{color};background:{color}1a">'
                f'<span class="lv" style="background:{color}">{level}</span>'
                f'<span class="msg">{msg}</span><span class="act">{act}</span></div></div>', unsafe_allow_html=True)


def show_power_chart(t, target):
    df = query("SELECT ts, kw FROM raw.meter_reading WHERE ts >= :s ORDER BY ts", s=t - timedelta(minutes=30))
    chips = ""
    if not df.empty:
        chips = chip("최대", f"{df['kw'].max():,.0f}") + chip("최소", f"{df['kw'].min():,.0f}") + \
                chip("평균", f"{df['kw'].mean():,.0f}")
    panel_head("전력 추이", "최근 30분 · kW", chips)
    st.markdown(f'<div class="legend notranslate" translate="no"><span style="color:{BLUE}">━</span>순시 전력 '
                f'<span></span><span style="color:{RED}">┅</span>목표 '
                f'<span></span><span style="color:{AMBER}">┅</span>경고 {WARN:.0%}</div>', unsafe_allow_html=True)
    if df.empty:
        empty("데이터 없음")
        return

    y_min = df["kw"].min() * 0.95
    y_max = max(df["kw"].max(), target or 0) * 1.03
    x = alt.X("ts:T", title=None, axis=alt.Axis(format="%H:%M", tickCount=6))
    y = alt.Y("kw:Q", title=None, scale=alt.Scale(domain=[y_min, y_max]))
    tooltip = [alt.Tooltip("ts:T", format="%H:%M:%S", title="시각"),
               alt.Tooltip("kw:Q", format=",.0f", title="kW")]

    base = alt.Chart(df)
    chart = (base.mark_area(color=BLUE, opacity=0.12).encode(x=x, y=y, y2=alt.datum(y_min))
             + base.mark_line(strokeWidth=1.6, color=BLUE).encode(x=x, y=y, tooltip=tooltip))
    if target is not None:
        lines = pd.DataFrame({"y": [target, target * WARN], "c": [RED, AMBER]})
        chart = chart + alt.Chart(lines).mark_rule(strokeDash=[5, 4], strokeWidth=1.2).encode(
            y="y:Q", color=alt.Color("c:N", scale=None))
    st.altair_chart(dark(chart), use_container_width=True)


# 15분 수요 막대 (최근 3시간). 흐린 막대 = 진행 중 구간
# 점 = 그 구간 동안 30초마다 낸 예측의 평균 (ML 성적표 오차와 같은 기준)
def show_demand_chart(t, target, has_ml):
    since = t - timedelta(hours=3)
    df = query("""
        SELECT window_start,
               CASE WHEN final THEN avg_kw ELSE projected_kw END AS kw,
               target_kw, final
        FROM ops.demand_15min
        WHERE window_start >= :s
        ORDER BY window_start""", s=since)

    chips = ""
    if not df.empty:
        over = int((df["kw"] >= df["target_kw"]).sum())
        chips = chip("최대", f"{df['kw'].max():,.0f}") + chip("초과", f"{over} / {len(df)} 구간")
    panel_head("15분 평균 수요", "최근 3시간 · kW", chips)

    legend = (f'<span style="color:{BLUE}">■</span>실적 <span style="color:{RED}">■</span>목표 초과 '
              f'<span style="opacity:.5">■</span>진행 중')
    if has_ml:
        legend += f' &nbsp;<span style="color:{PURPLE}">●</span>ML 예측 <span style="color:{GRAY}">◆</span>계산식 예측'
    st.markdown(f'<div class="legend notranslate" translate="no">{legend}</div>', unsafe_allow_html=True)
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
    x = alt.X("구간:O", title=None, axis=alt.Axis(labelAngle=0, labelOverlap=True))
    y_scale = alt.Scale(domain=[0, y_max])
    tooltip = [alt.Tooltip("구간:O"),
               alt.Tooltip("kw:Q", format=",.0f", title="실적 kW"),
               alt.Tooltip("formula_kw:Q", format=",.0f", title="계산식 예측"),
               alt.Tooltip("ml_kw:Q", format=",.0f", title="ML 예측")]

    base = alt.Chart(df)
    chart = base.mark_bar(size=14, cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        x=x,
        y=alt.Y("kw:Q", title=None, scale=y_scale),
        color=alt.condition("datum.kw >= datum.target_kw", alt.value(RED), alt.value(BLUE)),
        opacity=alt.condition("datum.final", alt.value(1.0), alt.value(0.45)),
        tooltip=tooltip,
    )
    if target is not None:
        chart = chart + alt.Chart(pd.DataFrame({"y": [target]})).mark_rule(
            strokeDash=[5, 4], color=RED, strokeWidth=1.2).encode(y="y:Q")
    if df["ml_kw"].notna().any():
        formula_pts = base.transform_filter("isValid(datum.formula_kw)").mark_point(
            shape="diamond", size=60, filled=True, color=GRAY, opacity=1, stroke=PANEL, strokeWidth=1.5,
        ).encode(x=x, y=alt.Y("formula_kw:Q", scale=y_scale), tooltip=tooltip)
        ml_pts = base.transform_filter("isValid(datum.ml_kw)").mark_point(
            shape="circle", size=60, filled=True, color=PURPLE, opacity=1, stroke=PANEL, strokeWidth=1.5,
        ).encode(x=x, y=alt.Y("ml_kw:Q", scale=y_scale), tooltip=tooltip)
        chart = chart + formula_pts + ml_pts
    st.altair_chart(dark(chart), use_container_width=True)


# 전력 품질 + 통신 상태 표. 품질이 모두 정상인지 돌려줌
def show_quality(t, r):
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

    def state(ok, warn=False):
        if ok:
            return (badge("정상", GREEN), "html")
        return (badge("주의", AMBER) if warn else badge("이탈", RED), "html")

    v_ok = V_MIN <= r["voltage"] <= V_MAX
    hz_ok = HZ_MIN <= r["hz"] <= HZ_MAX
    pf_ok = r["pf"] >= PF_MIN
    soc_ok = r["soc"] > SOC_MIN
    gap = float(h["gap_sec"])
    rows = [
        ["전압", f"{r['voltage']:.1f} V", f"{V_MIN}~{V_MAX} V", state(v_ok)],
        ["주파수", f"{r['hz']:.2f} Hz", f"{HZ_MIN}~{HZ_MAX} Hz", state(hz_ok)],
        ["역률", f"{r['pf']:.3f}", f"{PF_MIN:.2f} 이상", state(pf_ok, warn=True)],
        ["ESS 충전율", f"{r['soc']:.1f} %", f"{SOC_MIN}% 초과", state(soc_ok, warn=True)],
        ["수신 지연", f"{lag:.0f} 초", f"{DELAY_SEC}초 이내", state(lag <= DELAY_SEC, warn=lag < OFFLINE_SEC)],
        ["빈 구간 (1h)", f"{gap:.0f} 초", "0초", state(gap == 0, warn=True)],
        ["재전송 (1h)", f"{int(h['resent'])} 건", "복구 후 자동", (badge("재전송 완료", BLUE), "html")
         if int(h["resent"]) else state(True)],
    ]
    panel_head("전력 품질 · 통신", "현재값 / 기준")
    st.markdown(table(["항목", "현재", "기준", "상태"], rows, ["25%", "22%", "26%", "27%"], num=(1,)),
                unsafe_allow_html=True)
    return v_ok and hz_ok and pf_ok


# 최근 1초 계측값. 기준 밖 값은 빨간 글씨
def show_readings():
    df = query("""
        SELECT ts, kw, voltage, current, pf, hz, soc, ess_kw, buffered,
               extract(epoch FROM received_at - ts) AS delay
        FROM raw.meter_reading ORDER BY ts DESC LIMIT :n""", n=ROWS)
    panel_head("실시간 계측 데이터", "1초 주기 · SOC % · ESS kW · 지연 초", chip("표시", f"최근 {ROWS}건"))

    def mark(v, ok):
        return v if ok else (v, "bad")

    rows = []
    for _, m in df.iterrows():
        delay = m["delay"]
        rows.append([
            f"{m['ts']:%H:%M:%S}",
            f"{m['kw']:,.1f}",
            mark(f"{m['voltage']:.1f}", V_MIN <= m["voltage"] <= V_MAX),
            f"{m['current']:,.1f}",
            mark(f"{m['pf']:.3f}", m["pf"] >= PF_MIN),
            mark(f"{m['hz']:.2f}", HZ_MIN <= m["hz"] <= HZ_MAX),
            mark(f"{m['soc']:.1f}", m["soc"] > SOC_MIN),
            f"{m['ess_kw']:.0f}",
            "-" if pd.isna(delay) else f"{delay:.1f}",
            (badge("재전송", BLUE), "html") if m["buffered"] else "실시간",
        ])
    st.markdown(table(["시각", "kW", "V", "A", "역률", "Hz", "SOC", "ESS", "지연", "구분"], rows,
                      ["12%", "11%", "9%", "11%", "9%", "8%", "8%", "9%", "8%", "15%"],
                      num=(1, 2, 3, 4, 5, 6, 7, 8)),
                unsafe_allow_html=True)


EVENTS_SQL = """
    SELECT ts, '제어' AS cat, action AS kind, replace(target, 'ess', 'ESS') || ' · ' || reason AS msg, projected_kw AS kw
    FROM ops.control_event {where}
    UNION ALL
    SELECT ts, '알림', kind, message, value
    FROM ops.alert {where}
    ORDER BY ts DESC"""


def event_badge(cat, kind):
    if cat == "제어":
        return badge(ACTIONS.get(kind, kind), BLUE)
    return badge(KINDS.get(kind, kind), SEVERITY.get(kind, GRAY))


# 제어 이력 + 알림을 시간순으로 합친 표
def show_events():
    df = query(EVENTS_SQL.format(where="") + " LIMIT :n", n=ROWS)
    today = query("SELECT (SELECT count(*) FROM ops.control_event WHERE ts >= current_date) AS c, "
                  "(SELECT count(*) FROM ops.alert WHERE ts >= current_date) AS a").iloc[0]
    panel_head("이벤트 로그", "제어 + 알림",
               chip("오늘 제어", f"{int(today['c'])}건") + chip("오늘 알림", f"{int(today['a'])}건"))
    rows = []
    for _, e in df.iterrows():
        rows.append([f"{e['ts']:%H:%M:%S}", e["cat"], (event_badge(e["cat"], e["kind"]), "html"), e["msg"]])
    st.markdown(table(["시각", "구분", "종류", "내용"], rows, ["16%", "11%", "23%", "50%"]), unsafe_allow_html=True)


# 오늘 마감된 구간들의 계산식 vs ML 오차
def show_ml_score(t, has_ml):
    if not has_ml:
        panel_head("ML 예측 성적", "오늘 · 확정 구간")
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
        panel_head("ML 예측 성적", "오늘 · 확정 구간")
        empty("오늘 평가된 구간 없음 · 첫 구간 마감 후 표시")
        return

    f_mae = float(s["formula_mae"])
    m_mae = float(s["ml_mae"])
    wins = int(s["ml_wins"])
    gain = (f_mae - m_mae) / f_mae if f_mae > 0 else 0.0
    panel_head("ML 예측 성적", "오늘 · 확정 구간", chip("오차 개선", f"{gain:+.0%}"))

    last = query("SELECT window_start, formula_mae, ml_mae FROM ml.window_eval ORDER BY window_start DESC LIMIT 1").iloc[0]
    last_win = "ML" if last["ml_mae"] < last["formula_mae"] else "계산식"

    rows = [
        ["평가 구간", f"{n}개", ""],
        ["평균 오차 (MAE)", f"{f_mae:,.1f}", f"{m_mae:,.1f}"],
        ["오차 개선", "-", f"{gain:+.1%}"],
        ["구간별 승", f"{n - wins}", f"{wins} ({wins / n:.0%})"],
        ["최근 구간", f"{last['window_start']:%H:%M}", f"{last_win} 승"],
    ]
    st.markdown(table(["항목", "계산식", "ML"], rows, ["40%", "28%", "32%"], num=(1, 2)), unsafe_allow_html=True)


@st.fragment(run_every=REFRESH)
def monitor():
    t = now()

    last = query("SELECT * FROM raw.meter_reading ORDER BY ts DESC LIMIT 1")
    if last.empty:
        topbar(t, "실시간 모니터링")
        st.info("아직 수신된 데이터가 없습니다.")
        return
    r = last.iloc[0]
    lag = (pd.Timestamp(t) - r["ts"]).total_seconds()
    online = lag < OFFLINE_SEC

    dem = query("SELECT * FROM ops.demand_15min ORDER BY window_start DESC LIMIT 1")
    d = None
    target = None
    if not dem.empty:
        d = dem.iloc[0]
        target = float(d["target_kw"])

    has_ml = has_ml_tables()
    ml = latest_ml(t, d["window_start"], has_ml) if d is not None else None

    topbar(t, "실시간 모니터링", online, lag)
    show_band(t, r, d, ml, has_ml)
    banner = st.empty()

    # 칸 높이를 고정해서 좌우 높이를 맞추고 내용이 밑으로 안 넘치게
    cols = st.columns([5, 5, 3.4])
    with box(cols[0], "power", CHART_BOX_H):
        show_power_chart(t, target)
    with box(cols[1], "demand", CHART_BOX_H):
        show_demand_chart(t, target, has_ml)
    with box(cols[2], "quality", CHART_BOX_H):
        quality_ok = show_quality(t, r)

    with banner.container():
        show_banner(r, d, ml, online, quality_ok)

    cols = st.columns([5, 5, 3.4])
    with box(cols[0], "readings", LOG_BOX_H):
        show_readings()
    with box(cols[1], "events", LOG_BOX_H):
        show_events()
    with box(cols[2], "ml", LOG_BOX_H):
        show_ml_score(t, has_ml)


# ── 이력 조회 ────────────────────────────────────────

def history():
    t = now()
    topbar(t, "이력 조회")
    has_ml = has_ml_tables()

    c = st.columns([2, 2, 6], vertical_alignment="bottom")
    day = c[0].date_input("날짜", value=t.date(), max_value=t.date())
    only_over = c[1].toggle("초과·경고 구간만", value=False)
    s = datetime.combine(day, datetime.min.time())
    e = s + timedelta(days=1)

    if has_ml:
        df = query("""
            SELECT d.window_start, d.avg_kw, d.max_projected_kw, d.target_kw, d.exceeded, d.final,
                   f.ml_kw, w.formula_mae, w.ml_mae
            FROM ops.demand_15min d
            LEFT JOIN (SELECT window_start, avg(ml_kw) AS ml_kw FROM ml.forecast GROUP BY window_start) f
                   USING (window_start)
            LEFT JOIN ml.window_eval w USING (window_start)
            WHERE d.window_start >= :s AND d.window_start < :e
            ORDER BY d.window_start DESC""", s=s, e=e)
    else:
        df = query("""
            SELECT window_start, avg_kw, max_projected_kw, target_kw, exceeded, final,
                   NULL::real AS ml_kw, NULL::real AS formula_mae, NULL::real AS ml_mae
            FROM ops.demand_15min
            WHERE window_start >= :s AND window_start < :e
            ORDER BY window_start DESC""", s=s, e=e)
    ev = query(EVENTS_SQL.format(where="WHERE ts >= :s AND ts < :e"), s=s, e=e)

    # 요약 밴드
    n = len(df)
    over = int(df["exceeded"].fillna(False).sum()) if n else 0
    peak = f"{df['avg_kw'].max():,.0f}" if n else "-"
    n_ctrl = int((ev["cat"] == "제어").sum())
    n_alert = int((ev["cat"] == "알림").sum())
    gain = "-"
    if has_ml and df["ml_mae"].notna().any():
        f_mae, m_mae = df["formula_mae"].mean(), df["ml_mae"].mean()
        gain = f"{(f_mae - m_mae) / f_mae:+.0%}" if f_mae > 0 else "-"
    cells = (kpi("15분 구간", f"{n}", "개", f"{day:%Y-%m-%d}")
             + kpi("최대 15분 수요", peak, "kW", "확정 구간 평균 기준")
             + kpi("목표 초과", f"{over}", "구간", "요금 기준 초과 위험", color=RED if over else "#fff")
             + kpi("자동 제어", f"{n_ctrl}", "건", "ESS 방전 · 부하 차단/복귀")
             + kpi("알림", f"{n_alert}", "건", "통신 · 품질 · 피크")
             + kpi("ML 오차 개선", gain, "", "계산식 대비 MAE"))
    st.markdown(f'<div class="spacer"><div class="band notranslate" translate="no">{cells}</div></div>',
                unsafe_allow_html=True)

    if only_over and n:
        df = df[df["exceeded"].fillna(False) | (df["max_projected_kw"] >= df["target_kw"] * WARN)]

    cols = st.columns([6, 5])
    with box(cols[0], "windows"):
        panel_head("15분 구간 기록", "최신순", chip("표시", f"{len(df)}개"))
        view = pd.DataFrame({
            "구간": df["window_start"].dt.strftime("%H:%M") + "~" +
                    (df["window_start"] + timedelta(minutes=15)).dt.strftime("%H:%M"),
            "평균 kW": df["avg_kw"],
            "최대 예상 kW": df["max_projected_kw"],
            "목표 kW": df["target_kw"],
            "목표 대비": df["avg_kw"] / df["target_kw"] * 100,
            "ML 예측 kW": df["ml_kw"],
            "계산식 오차": df["formula_mae"],
            "ML 오차": df["ml_mae"],
            "상태": [("진행 중" if not f else "초과" if x else "정상")
                     for f, x in zip(df["final"], df["exceeded"].fillna(False))],
        })
        st.dataframe(view, hide_index=True, height=520, use_container_width=True, column_config={
            "평균 kW": st.column_config.NumberColumn(format="%.0f"),
            "최대 예상 kW": st.column_config.NumberColumn(format="%.0f"),
            "목표 kW": st.column_config.NumberColumn(format="%.0f"),
            "목표 대비": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=120),
            "ML 예측 kW": st.column_config.NumberColumn(format="%.0f"),
            "계산식 오차": st.column_config.NumberColumn(format="%.1f"),
            "ML 오차": st.column_config.NumberColumn(format="%.1f"),
        })
        st.download_button("15분 구간 CSV 내려받기", view.to_csv(index=False).encode("utf-8-sig"),
                           f"demand_{day:%Y%m%d}.csv", "text/csv")

    with box(cols[1], "event_hist"):
        panel_head("이벤트 기록", "제어 + 알림 · 최신순")
        cat = st.segmented_control("구분", ["전체", "제어", "알림"], default="전체", label_visibility="collapsed")
        if cat in ("제어", "알림"):
            ev = ev[ev["cat"] == cat]
        names = {**ACTIONS, **KINDS}
        view = pd.DataFrame({
            "시각": ev["ts"].dt.strftime("%H:%M:%S"),
            "구분": ev["cat"],
            "종류": ev["kind"].map(lambda k: names.get(k, k)),
            "내용": ev["msg"],
            "값": ev["kw"],
        })
        st.dataframe(view, hide_index=True, height=478, use_container_width=True, column_config={
            "내용": st.column_config.TextColumn(width="large"),
        })
        st.download_button("이벤트 CSV 내려받기", view.to_csv(index=False).encode("utf-8-sig"),
                           f"events_{day:%Y%m%d}.csv", "text/csv")


# ── 기준값 ──────────────────────────────────────────

def settings():
    t = now()
    topbar(t, "기준값")
    dem = query("SELECT target_kw FROM ops.demand_15min ORDER BY window_start DESC LIMIT 1")
    target = float(dem.iloc[0]["target_kw"]) if not dem.empty else None
    tgt = f"{target:,.0f} kW" if target else "-"
    warn = f"{target * WARN:,.0f} kW" if target else "-"

    cols = st.columns(2)
    with box(cols[0], "limits"):
        panel_head("경보 · 판정 기준", "엣지 controller 와 같은 값")
        rows = [
            ["목표 최대수요 (15분 평균)", tgt, "넘으면 피크 초과 · 요금 기준"],
            ["경고 기준", warn, f"예상 수요가 목표의 {WARN:.0%} 이상"],
            ["전압", f"{V_MIN}~{V_MAX} V", "380 V ±10%"],
            ["주파수", f"{HZ_MIN}~{HZ_MAX} Hz", "60 Hz ±0.2"],
            ["역률", f"{PF_MIN:.2f} 이상", "미만이면 역률 저하 알림"],
            ["ESS 방전 하한", f"{SOC_MIN} %", "이하면 방전 중지"],
            ["수신 끊김 판정", f"{OFFLINE_SEC} 초", "넘게 안 오면 OFFLINE"],
            ["빈 구간 판정", f"{GAP_SEC} 초", "계측 간격이 이보다 길면 누락"],
            ["ML 예측 유효 시간", f"{ML_OLD_SEC} 초", "넘으면 serve.py 정지로 봄"],
        ]
        st.markdown(table(["항목", "기준값", "설명"], rows, ["34%", "22%", "44%"]), unsafe_allow_html=True)

    with box(cols[1], "order"):
        panel_head("피크 제어 순서", "15분 예상 수요 기준 · controller.py")
        rows = [["1", "ESS 방전", f"{ESS_KW} kW", f"목표 {WARN:.0%} 이상 · SOC {SOC_MIN}% 초과"]]
        for i, (name, kw) in enumerate(LOADS, start=2):
            rows.append([str(i), f"{name} 차단", f"{kw} kW", "목표 100% 이상 · 1건씩"])
        rows.append(["-", "부하 복귀", "-", f"목표 {SAFE:.0%} 미만 · 차단 역순"])
        rows.append(["-", "ESS 정지", "-", f"목표 {SAFE:.0%} 미만 또는 SOC {SOC_MIN}% 이하"])
        st.markdown(table(["순서", "동작", "감소량", "조건"], rows, ["12%", "24%", "18%", "46%"], num=(2,)),
                    unsafe_allow_html=True)


page = st.query_params.get("page", "monitor")
if page not in PAGES:
    page = "monitor"
rail(page)
if page == "history":
    history()
elif page == "settings":
    settings()
else:
    monitor()
