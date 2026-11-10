"""엔진 실험실 (담당: 수학과 B)
이상 심어 보기 · 240회 실험 결과 · 기준값 k · 숫자 검증기 · robust Z 손 재현.
240회 실험과 k 곡선, 검증기 시험 결과는 HTML 시안에 저장된 값(data/snap.json)을 그대로 보여 줘요.
'이상 심어 보기'는 단순 robust Z 탐지기와 공용 엔진(시계열 모드, 4개 방법 합의)을 골라 돌려 볼 수 있어요.
공용 엔진 v1.1부터 시간 열이 없으면 자동으로 표 모드가 되므로, 행 순서가 시간인 데이터는 mode="timeseries"를 꼭 지정해요.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from engine import check_table, events_table, load_csv, report_text, run_engine  # noqa: E402

st.set_page_config(page_title="엔진 실험실", layout="wide")
NUM_RE = re.compile(r"(?:^|[^A-Za-z0-9_.])(-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?)")


@st.cache_data
def load_lab():
    return json.loads((ROOT / "data" / "snap.json").read_text(encoding="utf-8"))["lab"]


L = load_lab()


def numbers(text):
    return [float(m.replace(",", "")) for m in NUM_RE.findall(text)]


def verify(report, summary):
    allowed = {abs(x) for x in numbers(summary)}
    found = numbers(report)
    return found, [x for x in found if abs(x) not in allowed]


def robust_z(values, base_n):
    base = values[:base_n]
    m = np.median(base)
    s = 1.4826 * np.median(np.abs(base - m))
    return m, s, (values - m) / (s if s > 0 else 1)


def segments(mask, min_len):
    out, start = [], None
    for i, on in enumerate(list(mask) + [False]):
        if on and start is None:
            start = i
        if not on and start is not None:
            if i - start >= min_len:
                out.append((start, i - 1))
            start = None
    return out


def engine_segments(y, k):
    """공용 엔진을 시계열 모드로 돌려 (시작 행, 끝 행) 목록을 돌려줘요.
    시간 열이 없는 데이터라 mode를 지정하지 않으면 표 모드가 되어 이벤트가 0건이 돼요."""
    out = run_engine(pd.DataFrame({"값": y}), mode="timeseries", k=k)
    rn = out["prep"]["rowNums"]  # CSV 줄 번호(머리글이 1번) → 0부터 세는 행 번호는 -2
    return out, [(rn[e["start"]] - 2, rn[e["end"]] - 2) for e in out["result"]["events"]]


st.caption("엔진 실험실 · 담당: 수학과 B")
st.title("이 엔진을 어디까지 믿을 수 있나")
big = L["bench"]["big"]
a, b, c, d = st.columns(4)
a.metric("240회 재현율", f"{L['bench']['recall']:.1%}")
b.metric("정밀도", f"{L['bench']['precision']:.1%}")
c.metric("4σ 이상 F1", f"{big['f1']:.2f}")
d.metric("숫자 검증기 재현율", f"{L['verifier']['numRecall']:.1%}")

t_inject, t_csv, t_bench, t_k, t_ver, t_rob, t_ai = st.tabs(
    ["이상 심어 보기", "내 CSV로 돌려 보기", "240회 실험", "기준값 k", "숫자 검증기", "robust Z 손 재현", "AI 전송 설계"])

with t_inject:
    st.subheader("이상을 하나 심고 탐지기를 돌려요")
    st.caption("회색 = 심은 구간(정답), 빨강 = 잡은 이벤트 · 겹치면 정답, 안 겹치면 오탐")
    detector = st.radio("탐지기", ["robust Z 하나 (단순)", "공용 엔진 (시계열 모드)"], horizontal=True)
    use_engine = detector.startswith("공용")
    col1, col2, col3, col4 = st.columns(4)
    kind = col1.selectbox("이상 종류", ["튀는 값", "수준 이동", "변동성 증가", "서서히 흐름"])
    size = col2.slider("크기(σ)", 1.0, 8.0, 4.0, 0.5)
    if use_engine:
        k_val = col3.slider("민감도 k (크면 덜 민감)", 0.5, 3.0, 1.0, 0.1)
    else:
        k_val = col3.slider("기준 |z|", 2.0, 8.0, 4.0, 0.5)
    seed = col4.number_input("시드", 0, 9999, 42)
    r = np.random.default_rng(int(seed))
    n, start, length = 500, 320, 40
    y = 10 + r.standard_normal(n)
    if kind == "튀는 값":
        length = 1
        y[start] += size
    elif kind == "수준 이동":
        y[start: start + length] += size
    elif kind == "변동성 증가":
        y[start: start + length] = 10 + (y[start: start + length] - 10) * size
    else:
        y[start: start + length] += np.linspace(0, size, length)
    if use_engine:
        out, events = engine_segments(y, k_val)
        st.caption(f"엔진 모드: {out['mode']} · 분석 열: {', '.join(out['columns'])} · {out['notice']}")
    else:
        _, _, z = robust_z(y, 150)
        events = segments(np.abs(z) >= k_val, 1 if kind == "튀는 값" else 2)
    hit = any(s <= start + length - 1 and e >= start for s, e in events)
    fp = sum(1 for s, e in events if e < start or s > start + length - 1)
    fig = go.Figure(go.Scatter(y=y, name="값", line=dict(width=1)))
    fig.add_vrect(x0=start - 0.5, x1=start + length - 0.5, fillcolor="gray", opacity=0.3, line_width=0)
    for s, e in events:
        fig.add_vrect(x0=s - 0.5, x1=e + 0.5, fillcolor="crimson", opacity=0.3, line_width=0)
    fig.update_layout(height=320, xaxis_title="행", margin=dict(t=20))
    st.plotly_chart(fig, width="stretch")
    a, b = st.columns(2)
    a.metric("심은 이상", "잡음" if hit else "놓침")
    b.metric("오탐", f"{fp}건")
    if use_engine:
        with st.expander("모드를 지정하지 않으면? (auto → 표 모드)"):
            auto = run_engine(pd.DataFrame({"값": y}), k=k_val)
            st.write(f"자동 모드 결과: **{auto['mode']}** · 이벤트 {len(auto['result'].get('events', []))}건 · "
                     f"확인이 필요한 행 {len(auto['result'].get('outliers', []))}개")
            st.caption("이 데이터에는 시간 열이 없어서 자동이면 표 모드가 돼요. 행을 하나씩 따로 보므로 수준 이동·서서히 흐름 같은 '이어진 이상'을 이벤트로 묶지 못해요.")

with t_csv:
    st.subheader("내 CSV를 공용 엔진에 넣어 보기")
    st.caption("시간 열이 있으면 자동으로 시계열, 없으면 표 모드예요. 시간 열 없이 행 순서가 시간인 센서 데이터라면 '시계열'을 고르세요.")
    file = st.file_uploader("CSV 파일", type=["csv"])
    col1, col2 = st.columns(2)
    mode_ko = {"자동": "auto", "시계열 (행 순서 = 시간)": "timeseries", "표 (한 행 = 한 건)": "table"}
    mode_label = col1.radio("모드", list(mode_ko), horizontal=True)
    k_csv = col2.slider("민감도 k (크면 덜 민감)", 0.5, 3.0, 1.0, 0.1, key="k_csv")
    if file is not None:
        try:
            table = load_csv(file.getvalue())
            out = run_engine(table, mode=mode_ko[mode_label], k=k_csv, dataset_name=file.name)
        except ValueError as e:
            st.error(str(e))
        else:
            a, b, c = st.columns(3)
            a.metric("실제로 돈 모드", "시계열" if out["mode"] == "timeseries" else "표")
            b.metric("분석한 열", len(out["columns"]))
            c.metric("이벤트" if out["mode"] == "timeseries" else "확인이 필요한 행",
                     len(out["result"]["events"] if out["mode"] == "timeseries" else out["result"]["outliers"]))
            st.caption(out["notice"])
            if mode_ko[mode_label] == "auto" and out["mode"] == "table":
                st.info("시간 열을 찾지 못해 표 모드로 돌았어요. 센서처럼 행 순서가 시간인 데이터라면 위에서 '시계열'을 고르세요.")
            st.markdown("**데이터 점검**")
            st.dataframe(pd.DataFrame(check_table(out)), width="stretch", hide_index=True)
            st.markdown("**이벤트**" if out["mode"] == "timeseries" else "**확인이 필요한 행**")
            st.dataframe(pd.DataFrame(events_table(out)), width="stretch", hide_index=True)
            with st.expander("기본 리포트"):
                st.markdown(report_text(out))

with t_bench:
    st.subheader("이상 종류 × 크기별 재현율")
    st.caption(f"총 {L['bench']['runs']}회 · 기준값 k = {L['bench']['k']} · HTML 시안에 저장된 결과")
    rows = []
    for g in L["bench"]["grid"]:
        for cell in g["cells"]:
            row = {"이상 종류": g["name"], "단위": g.get("unit", "")}
            row.update({key: val for key, val in cell.items() if not isinstance(val, (list, dict))})
            rows.append(row)
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

with t_k:
    st.subheader("기준값 k에 따른 정밀도·재현율·F1")
    pts = pd.DataFrame(L["sweep"]["pts"]).set_index("k")[["precision", "recall", "f1"]]
    st.line_chart(pts)
    st.metric("최고 F1을 낸 k", L["sweep"]["best"]["k"], f"F1 {L['sweep']['best']['f1']:.3f}", delta_color="off")
    st.dataframe(pts.round(3), width="stretch")

with t_ver:
    st.subheader("숫자 검증기 직접 써 보기")
    st.caption("리포트 속 숫자가 계산값 요약에 있는지 하나씩 대조해요.")
    col1, col2 = st.columns(2)
    summary = col1.text_area("계산값 요약", '{"관측_빈도": 0.1061, "검증_A_E_건수": 1.022, "순보험료": 235.9}', height=140)
    report = col2.text_area("리포트", "관측 빈도는 0.1061이고 검증 A/E는 1.022다. 순보험료는 240.5유로다.", height=140)
    found, bad = verify(report, summary)
    if not found:
        st.info("검사할 숫자가 없어요.")
    elif bad:
        st.warning(f"숫자 {len(found)}개 중 {len(bad)}개가 계산값에 없어요: {bad}")
    else:
        st.success(f"숫자 {len(found)}개가 모두 계산값과 맞아요.")
    st.subheader("기존 시험 결과")
    V = L["verifier"]
    a, b, c = st.columns(3)
    a.metric("바꾼 숫자 중 잡은 수", f"{V['caught']} / {V['totalChanged']}")
    b.metric("변조 리포트 적발률", f"{V['reportRecall']:.0%}")
    c.metric("깨끗한 리포트 오경보", f"{V['falseAlarm']}건 / {V['nClean']}건")
    case_id = st.selectbox("시험 리포트", range(len(V["cases"])), format_func=lambda i: V["cases"][i]["id"] + (" · 변조" if V["cases"][i]["tampered"] else " · 원본"))
    st.markdown(V["cases"][case_id]["report"])

with t_rob:
    st.subheader("값 15개, 그중 하나가 튐")
    st.caption("엑셀로 같은 계산을 해서 맞춰 보는 예제예요.")
    text = st.text_input("값 (쉼표로 구분)", "10.2, 9.8, 10.1, 9.9, 10.0, 10.3, 9.7, 10.1, 9.9, 10.2, 10.0, 9.8, 18.5, 10.1, 9.9")
    try:
        vals = np.array([float(x) for x in text.split(",")])
        m, s, z = robust_z(vals, len(vals))
        a, b, c = st.columns(3)
        a.metric("중앙값", f"{m:.3f}")
        b.metric("MAD", f"{s / 1.4826:.3f}")
        c.metric("1.4826 × MAD", f"{s:.3f}")
        st.dataframe(pd.DataFrame({"값": vals, "robust z": np.round(z, 2)}), width="stretch")
        st.latex(r"z_i = \frac{x_i - \mathrm{median}(x)}{1.4826 \times \mathrm{MAD}}")
    except ValueError:
        st.error("숫자를 쉼표로 구분해 입력해 주세요.")

with t_ai:
    st.subheader("AI에게 보내는 것과 보내지 않는 것")
    col1, col2 = st.columns(2)
    col1.markdown("**보내지 않음**\n- 원본 행 전체\n- 이름·연락처·주민번호 열\n- 한 사람·한 계약 단위의 값")
    col2.markdown("**보냄**\n- 집계 통계와 이벤트 목록\n- 계산된 지표(A/E, AUC, RMSE)\n- 분야 기준 규칙과 출처")
    st.subheader("비용·안정성 장치 (설계안)")
    st.markdown("- 틀린 숫자 다시 쓰기 최대 2번\n- 같은 요약은 저장해 재사용\n- 접속당 호출 20번 상한")
    st.info("이 앱에는 아직 AI 호출을 연결하지 않았어요. API 키는 코드에 넣지 말고 Streamlit Secrets에 저장해요.")
