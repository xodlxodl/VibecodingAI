"""제조 PHM (담당: 산공)
3D 프린터 데이터: Szydlo T. et al. (2021), github.com/joanna-/3D-Printing-Data, CC BY 4.0
베어링·잔여수명·정비 주기: HTML 시안의 시뮬레이션 결과(data/snap.json)
"""
import io
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="제조 PHM", layout="wide")
ROOT = Path(__file__).parent.parent


@st.cache_data
def load_data():
    snap = json.loads((ROOT / "data" / "snap.json").read_text(encoding="utf-8"))
    prn = json.loads((ROOT / "data" / "prn_data.json").read_text(encoding="utf-8"))
    return snap, prn


SNAP, PD = load_data()
B, R, W = SNAP["bearing"], SNAP["rul"], SNAP["home"]["weibull"]

# ---------------------------------------------------------------- 3D 프린터 FDC
RUNS = [
    ("proper", "정상 출력", None),
    ("plastic", "필라멘트 소진", "plastic"),
    ("bowden", "필라멘트 튜브 빠짐", "bowden"),
    ("retraction", "되감기 설정 오류", "retraction"),
    ("unstick", "출력물이 판에서 떨어짐", "unstick"),
    ("arm", "헤드 팔 분리", "arm"),
]
RUN_NAME = {k: n for k, n, _ in RUNS}
# 창 열: [t, z, a0, a1, tilt, ten, osc, noz, sat]
FEATS = {
    "a0": dict(i=2, name="출력판 진동", unit="m/s²", floor=lambda r: 0.1 * r),
    "a1": dict(i=3, name="헤드 진동", unit="m/s²", floor=lambda r: 0.1 * r),
    "tilt": dict(i=4, name="헤드 기울기", unit="°", floor=lambda r: 0.5),
    "ten": dict(i=5, name="필라멘트 장력", unit="", floor=lambda r: max(30, 0.05 * r)),
    "osc": dict(i=6, name="장력 흔들림", unit="", floor=lambda r: max(5, 0.1 * r)),
}
TYPES = {
    "arm": "헤드 팔 분리",
    "bowden": "필라멘트 튜브 빠짐",
    "plastic": "필라멘트 소진",
    "retraction": "되감기 설정 오류",
    "unstick": "출력물이 판에서 떨어짐",
    "other": "미분류 이상",
}
OOS = ["plastic", "bowden", "unstick", "arm"]
LAYER_MAX = 6.5


def is_num(x):
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def num(x):
    return x if is_num(x) else 0


def med(values):
    return float(np.median(values)) if len(values) else float("nan")


def bin_of(z):
    return f"L{z:.2f}" if z < LAYER_MAX else f"T{math.floor(z)}"


def baseline():
    win, zmax = PD["runs"]["proper"]["win"], PD["ZMAX"]
    table = {}
    for key, fe in FEATS.items():
        groups = {}
        for w in win:
            if w[1] < zmax and is_num(w[fe["i"]]):
                groups.setdefault(bin_of(w[1]), []).append(w[fe["i"]])
        tb = {}
        for b, v in groups.items():
            if len(v) < 3:
                continue
            m = med(v)
            s = 1.4826 * med([abs(x - m) for x in v])
            tb[b] = dict(m=m, s=max(s, fe["floor"](abs(m))), n=len(v))
        table[key] = tb
    return table


def ref(BL, key, z):
    tb, b = BL[key], bin_of(z)
    if b in tb:
        return tb[b]
    if b[0] == "L":
        return None
    ks = [int(k[1:]) for k in tb if k[0] == "T"]
    best = ks[0]
    for k in ks:
        if abs(k - math.floor(z)) < abs(best - math.floor(z)):
            best = k
    return tb[f"T{best}"]


def scores(BL, win):
    out = []
    for w in win:
        if not w[1] < PD["ZMAX"]:
            out.append(None)
            continue
        o = {}
        for key, fe in FEATS.items():
            r = ref(BL, key, w[1])
            if r is None:
                o = None
                break
            o[key] = (w[fe["i"]] - r["m"]) / r["s"] if is_num(w[fe["i"]]) else 0
        out.append(o)
    return out


def alarms(Z, K, P):
    out = {}
    for key in FEATS:
        segs, s = [], -1
        for i in range(len(Z) + 1):
            on = i < len(Z) and Z[i] is not None and abs(Z[i][key]) >= K
            if on and s < 0:
                s = i
            if not on and s >= 0:
                if i - s >= P:
                    segs.append(dict(s=s, e=i - 1, dir=1 if Z[s][key] > 0 else -1))
                s = -1
        out[key] = segs
    return out


def classify(win, Z, A, onset_idx):
    first = {}
    for key in FEATS:
        g = next((x for x in A[key] if x["e"] >= onset_idx), None)
        if g:
            first[key] = g
    if not first:
        return None
    t0 = min(g["s"] for g in first.values())
    in_ev = [k for k in first if first[k]["s"] <= t0 + 18]
    order = sorted(in_ev, key=lambda k: (first[k]["s"], -abs(Z[first[k]["s"]][k])))
    d = lambda k: first[k]["dir"] if k in first else 0
    kind = "other"
    if d("tilt") < 0 and "tilt" in in_ev:
        kind = "arm"
    elif (
        "ten" in in_ev
        and d("ten") < 0
        and any(num(w[5]) < 300 for w in win[first["ten"]["s"]: first["ten"]["e"] + 1])
    ):
        j = first["ten"]["s"]
        while j <= first["ten"]["e"] and num(win[j][5]) >= 300:
            j += 1
        pre = med([num(w[5]) for w in win[max(0, first["ten"]["s"] - 6): first["ten"]["s"]]])
        k = j
        while k > 0 and num(win[k - 1][5]) < 0.7 * pre:
            k -= 1
        kind = "bowden" if j - k == 0 else "plastic"
    elif "osc" in in_ev and d("osc") < 0:
        kind = "retraction"
    elif "a0" in in_ev and d("a0") > 0 and not ("ten" in in_ev and d("ten") < 0):
        kind = "unstick"
    return dict(t0=t0, order=order, first=first, type=kind)


def merge_events(items):
    out = []
    for g in sorted(items, key=lambda x: x["s"]):
        if out and g["s"] <= out[-1]["e"] + 1:
            out[-1]["e"] = max(out[-1]["e"], g["e"])
        else:
            out.append(dict(s=g["s"], e=g["e"]))
    return out


@st.cache_data
def study(K, P):
    BL = baseline()
    runs = {}
    for key, _, _ in RUNS:
        win = PD["runs"][key]["win"]
        Z = scores(BL, win)
        A = alarms(Z, K, P)
        lab = PD["labels"].get(key)
        onset = None if lab is None else lab.get("visible", lab["t"])
        idx = 0
        if onset is not None:
            while idx < len(win) - 1 and win[idx][0] < onset:
                idx += 1
        pre_end = len(win) if key == "proper" else max(0, idx - 1)
        pre = [g for k in FEATS for g in A[k] if g["e"] < pre_end]
        fdc = None if key == "proper" else classify(win, Z, A, idx)
        runs[key] = dict(
            A=A, Z=Z, onset=onset, fdc=fdc,
            detect=win[fdc["t0"]][0] if fdc else None,
            false_alarms=merge_events(pre),
            pre_hours=sum(1 for z in Z[:pre_end] if z) * PD["W"] / 3600,
            unjudged=sum(1 for z in Z if not z),
        )
    faults = [r for r in RUNS if r[2]]
    correct = sum(1 for k, _, f in faults if runs[k]["fdc"] and runs[k]["fdc"]["type"] == f)
    return dict(BL=BL, runs=runs, correct=correct, n=len(faults))


def mmss(sec):
    if sec is None:
        return "-"
    return f"{int(sec // 60)}분 {int(round(sec % 60)):02d}초"


# ---------------------------------------------------------------- 직접 측정
def per_second_rms(df):
    t = df.iloc[:, 0].to_numpy(dtype=float)
    axes = df.iloc[:, 1:4].to_numpy(dtype=float)
    sec = np.floor(t - t[0]).astype(int)
    rows = []
    for s in np.unique(sec):
        part = axes[sec == s]
        if len(part) >= 10:
            rows.append((int(s), float(np.sqrt(np.nanvar(part, axis=0).sum()))))
    return rows


def windows10(rows):
    groups = {}
    for t, rms in rows:
        groups.setdefault(t // 10, []).append(rms)
    out = []
    for k, v in groups.items():
        if len(v) >= 6:
            v = sorted(v)
            out.append((k * 10, v[min(len(v) - 1, int(math.floor(0.95 * (len(v) - 1) + 0.5)))]))
    return out


def band_check(base, test, K, P):
    v = [x[1] for x in base]
    m = med(v)
    s = max(1.4826 * med([abs(x - m) for x in v]), 0.05 * m)
    lo, hi = m - K * s, m + K * s
    out, start = [], -1
    for i in range(len(test) + 1):
        on = i < len(test) and (test[i][1] > hi or test[i][1] < lo)
        if on and start < 0:
            start = i
        if not on and start >= 0:
            if i - start >= P:
                out.append((start, i - 1))
            start = -1
    hours = len(test) * 10 / 3600
    return dict(m=m, lo=lo, hi=hi, alarms=out, hours=hours, rate=len(out) / hours if hours else float("nan"))


# ---------------------------------------------------------------- 정비 주기
def optimal_age(beta, eta, cp, cf, tmax, steps=2000):
    T = np.linspace(tmax / steps, tmax, steps)
    Rt = np.exp(-((T / eta) ** beta))
    dt = T[1] - T[0]
    integral = np.cumsum((np.r_[1.0, Rt[:-1]] + Rt) / 2) * dt
    cost = (cp * Rt + cf * (1 - Rt)) / integral
    mttf = eta * math.gamma(1 + 1 / beta)
    crtf = cf / mttf
    i = int(np.argmin(cost))
    exists = beta > 1 and cost[i] < crtf and i < steps - 1
    return dict(T=T, cost=cost, crtf=crtf, Tstar=T[i], cstar=cost[i], saving=1 - cost[i] / crtf, Rstar=Rt[i], exists=exists, mttf=mttf)


# ---------------------------------------------------------------- 화면
st.caption("제조 PHM 팩 · 담당: 산공")
st.title("실제 3D 프린터의 이상을 찾아 종류까지 가르고, 남은 수명과 가장 싼 정비 시점까지")

X0 = study(6, 2)
rf = next(m for m in R["models"] if m["key"] == "rf")
base_m = next(m for m in R["models"] if m["key"] == "mean")
ols = next(m for m in R["models"] if m["key"] == "ols")
c1, c2, c3, c4 = st.columns(4)
c1.metric("베어링 분류 정확도 (부하 조건 분할)", f"{B['byLoad']['acc']:.1%}")
c1.caption(f"무작위 분할 {B['random']['acc']:.1%}")
c2.metric("3D 프린터 이상, 종류까지 맞힌 수", f"{X0['correct']} / {X0['n']}")
c2.caption("사례 확인이지 검증이 아니에요")
c3.metric("잔여수명 RMSE (랜덤포레스트)", f"{rf['rmse']:.1f} 사이클")
c3.caption(f"평균 수명으로 찍기 {base_m['rmse']:.1f}")
c4.metric("최적 정비 주기 (고장 비용 5배)", f"{W['Tstar']:.0f} 사이클")
c4.caption(f"고장 후 교체 대비 {W['saving']:.0%} 절감")
st.info("3D 프린터는 공개된 실제 데이터(AGH, CC BY 4.0)이고, 베어링·잔여수명·정비 주기는 시뮬레이션 값이에요.")

tab_phone, tab_bear, tab_rul, tab_maint, tab_engine = st.tabs(
    ["프린터 직접 측정", "베어링 진동 진단", "잔여수명 예측", "정비 주기", "3D 프린터 이상 탐지·분류"]
)

with tab_phone:
    st.subheader("정상 기준선과 헛경보")
    st.caption("10초 창 진동(1초 RMS의 95% 분위수) · 기준선 = 기준선 녹화의 중앙값 ± K × 퍼짐")
    recs = {
        "공개 데이터 · 정상 출력": [tuple(r[:2]) for r in PD["sec1"]["proper"]],
        "공개 데이터 · 다른 정상 출력 구간": [tuple(r[:2]) for r in PD["sec1"]["arm"]],
    }
    files = st.file_uploader(
        "우리 프린터 phyphox CSV 올리기 (열 순서: 시간, x, y, z)", type=["csv", "txt"], accept_multiple_files=True
    )
    for f in files or []:
        try:
            raw = f.getvalue().decode("utf-8-sig")
            sep = ";" if raw.split("\n")[0].count(";") > raw.split("\n")[0].count(",") else ","
            rows = per_second_rms(pd.read_csv(io.StringIO(raw), sep=sep))
            if len(rows) < 60:
                st.warning(f"{f.name}: 1분 넘게 녹화해 주세요.")
            else:
                recs[f.name] = rows
        except Exception as e:
            st.error(f"{f.name}을 읽지 못했어요: {e}")
    names = list(recs)
    col1, col2, col3, col4 = st.columns(4)
    base_name = col1.selectbox("기준선 녹화", names, index=1)
    test_name = col2.selectbox("확인용 녹화", names, index=0)
    k_val = col3.slider("K", 3, 12, 6, key="pr_k")
    p_val = col4.slider("지속(10초 창 수)", 1, 6, 2, key="pr_p")
    if base_name == test_name:
        half = len(recs[base_name]) // 2
        base_w, test_w = windows10(recs[base_name][:half]), windows10(recs[base_name][half:])
    else:
        base_w, test_w = windows10(recs[base_name]), windows10(recs[test_name])
    res = band_check(base_w, test_w, k_val, p_val)
    fig = go.Figure(go.Scatter(x=[t / 60 for t, _ in test_w], y=[v for _, v in test_w], name="확인용 녹화", line=dict(width=1)))
    fig.add_hline(y=res["hi"], line_dash="dash", line_color="crimson", annotation_text=f"기준선 위 {res['hi']:.3f}")
    if res["lo"] > 0:
        fig.add_hline(y=res["lo"], line_dash="dash", line_color="crimson", annotation_text=f"기준선 아래 {res['lo']:.3f}")
    for s, e in res["alarms"]:
        fig.add_vrect(x0=test_w[s][0] / 60, x1=test_w[e][0] / 60, fillcolor="crimson", opacity=0.2, line_width=0)
    fig.update_layout(height=320, xaxis_title="시간 (분)", yaxis_title="진동 (m/s²)", margin=dict(t=20))
    st.plotly_chart(fig, width="stretch")
    m1, m2, m3 = st.columns(3)
    m1.metric("기준선 중앙값", f"{res['m']:.3f} m/s²")
    m2.metric("기준선 범위", f"{max(0, res['lo']):.3f}~{res['hi']:.3f}")
    m3.metric("헛경보", f"{len(res['alarms'])}건", f"시간당 {res['rate']:.1f}건", delta_color="off")
    st.caption("확인용 녹화도 정상 출력이라 경보는 모두 헛경보예요. 기준선은 프린터마다 따로 만들어요.")

with tab_bear:
    st.subheader("손 재현: 결함 주파수")
    st.caption("드라이브 쪽 베어링 6205 · 볼 9개, 볼 지름 0.3126인치, 피치 지름 1.537인치")
    load = st.select_slider("부하(마력)", [0, 1, 2, 3], 0)
    rpm = {0: 1797, 1: 1772, 2: 1750, 3: 1730}[load]
    fr = rpm / 60
    ratio = 0.3126 / 1.537
    bpfo, bpfi = 4.5 * fr * (1 - ratio), 4.5 * fr * (1 + ratio)
    bsf2 = (1.537 / 0.3126) * fr * (1 - ratio ** 2)
    a, b, c, d = st.columns(4)
    a.metric("회전 주파수 fr", f"{fr:.2f} Hz", f"{rpm} rpm", delta_color="off")
    b.metric("BPFO (외륜)", f"{bpfo:.1f} Hz")
    c.metric("BPFI (내륜)", f"{bpfi:.1f} Hz")
    d.metric("2×BSF (볼)", f"{bsf2:.1f} Hz")

    st.subheader("분류 결과: 나누는 방법이 정확도를 바꿔요")
    a, b, c = st.columns(3)
    a.metric("무작위 분할", f"{B['random']['acc']:.1%}", f"시험 {B['random']['nTest']}창", delta_color="off")
    b.metric("부하 조건 분할", f"{B['byLoad']['acc']:.1%}", f"시험 {B['byLoad']['nTest']}창", delta_color="off")
    c.metric("누수로 부풀려진 몫", f"{(B['random']['acc'] - B['byLoad']['acc']) * 100:.1f}%p")
    left, right = st.columns(2)
    for col, key, title in ((left, "random", "무작위 분할"), (right, "byLoad", "부하 조건 분할")):
        col.caption(f"{title} · 행 = 실제, 열 = 예측")
        col.dataframe(pd.DataFrame(B[key]["conf"], index=B["classes"], columns=B["classes"]), width="stretch")
    imp = pd.DataFrame({"특징": B["feats"], "중요도": B["byLoad"]["importance"]}).sort_values("중요도")
    st.plotly_chart(go.Figure(go.Bar(x=imp["중요도"], y=imp["특징"], orientation="h")).update_layout(height=280, margin=dict(t=20)), width="stretch")
    st.caption("무작위로 섞으면 같은 녹음의 조각이 학습과 시험 양쪽에 들어가 정확도가 부풀려져요.")

with tab_rul:
    st.subheader("모델 비교")
    st.dataframe(
        pd.DataFrame([{"모델": m["name"], "RMSE(사이클)": round(m["rmse"], 2), "설명": m["note"]} for m in R["models"]]),
        width="stretch", hide_index=True,
    )
    st.subheader("엔진별 예측 궤적")
    labels = [f"엔진 {t['unit']} · {t['cut']}사이클까지 관측 · 실제 남은 수명 {t['life'] - t['cut']}" for t in R["trajectories"]]
    pick = st.selectbox("시험 엔진", range(len(labels)), format_func=lambda i: labels[i])
    pts = np.array(R["trajectories"][pick]["pts"], dtype=float)
    fig = go.Figure()
    fig.add_scatter(x=pts[:, 0], y=pts[:, 1], name=f"실제 (상한 {R['CAP']})")
    fig.add_scatter(x=pts[:, 0], y=pts[:, 2], name="랜덤포레스트")
    fig.add_scatter(x=pts[:, 0], y=pts[:, 3], name="선형 회귀", line=dict(dash="dash"))
    fig.update_layout(height=340, xaxis_title="사이클", yaxis_title="잔여수명 (사이클)", margin=dict(t=20))
    st.plotly_chart(fig, width="stretch")
    st.caption(f"학습 엔진 {R['nTrain']}대, {R['trainRows']:,}행 · 엔진 단위로 나눠 누수를 막아요.")

with tab_maint:
    st.subheader("나이 기준 교체: 언제 바꾸는 게 가장 싼가")
    st.caption(f"와이블 β {W['beta']:.2f}, η {W['eta']:.1f}사이클")
    cf = st.slider("고장 비용 Cf ÷ 예방정비 Cp", 1.5, 20.0, 5.0, 0.5)
    opt = optimal_age(W["beta"], W["eta"], 1, cf, W["eta"] * 2.2)
    fig = go.Figure(go.Scatter(x=opt["T"], y=opt["cost"] * 1000, name="기대 비용 C(T)"))
    fig.add_hline(y=opt["crtf"] * 1000, line_dash="dash", annotation_text="고장 날 때까지 쓰기")
    if opt["exists"]:
        fig.add_vline(x=opt["Tstar"], line_dash="dot", annotation_text=f"T* {opt['Tstar']:.0f}")
    fig.update_yaxes(range=[0, opt["crtf"] * 1000 * 2.2])
    fig.update_layout(height=320, xaxis_title="교체 나이 T (사이클)", yaxis_title="1,000사이클당 비용 (Cp = 1)", margin=dict(t=20))
    st.plotly_chart(fig, width="stretch")
    if opt["exists"]:
        a, b, c = st.columns(3)
        a.metric("최적 교체 나이 T*", f"{opt['Tstar']:.0f} 사이클")
        b.metric("고장 후 교체 대비 절감", f"{opt['saving']:.1%}")
        c.metric("T*까지 고장 없이 버틸 확률", f"{opt['Rstar']:.1%}")
    else:
        st.warning("이 비용비에서는 미리 바꿔도 기대 비용이 줄지 않아요.")
    st.latex(r"C(T)=\frac{C_p R(T)+C_f(1-R(T))}{\int_0^T R(t)\,dt},\quad R(t)=e^{-(t/\eta)^\beta}")
    chk = optimal_age(2, 1000, 1, 5, 3000)
    st.caption(f"검산: β=2, η=1000, Cp=1, Cf=5 → T* = {chk['Tstar']:.1f}시간, 절감 {chk['saving']:.1%} (기대값: 약 511시간, 약 28%)")

with tab_engine:
    st.subheader("판정 결과: 무엇이, 어느 센서에서 먼저")
    st.caption("같은 높이(층)의 정상 출력 값과 비교 · |z| ≥ K가 정해진 시간 이상 이어지면 경보 → 서명표로 분류")
    col1, col2 = st.columns(2)
    K = col1.slider("K (같은 높이 정상 퍼짐의 몇 배)", 4, 12, 6)
    P = col2.slider("지속(10초 창 수)", 1, 5, 2)
    X = study(K, P)
    n_oos = sum(len(X["runs"][k]["false_alarms"]) for k in OOS)
    h_oos = sum(X["runs"][k]["pre_hours"] for k in OOS)
    a, b, c = st.columns(3)
    a.metric("종류까지 맞힌 수", f"{X['correct']} / {X['n']}")
    b.metric("헛경보 · 다른 출력의 정상 구간", f"{n_oos}건", f"{h_oos:.1f}시간 · 시간당 {n_oos / h_oos if h_oos else 0:.1f}건", delta_color="off")
    c.metric("판정하지 않은 창", sum(r["unjudged"] for r in X["runs"].values()))
    rows = []
    for key, name, fault in RUNS:
        if not fault:
            continue
        r = X["runs"][key]
        c_ = r["fdc"]
        rows.append({
            "이상(정답)": name,
            "정한 시작": mmss(r["onset"]),
            "경보": mmss(r["detect"]) if c_ else "못 찾음",
            "먼저 반응한 센서": (FEATS[c_["order"][0]]["name"] + (" ↑" if c_["first"][c_["order"][0]]["dir"] > 0 else " ↓")) if c_ else "-",
            "분류": TYPES[c_["type"]] if c_ else "-",
            "결과": ("맞음" if c_["type"] == fault else "틀림") if c_ else "놓침",
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption("서명표를 이 5건을 보고 만들었으므로, 맞힌 수는 검증이 아니라 사례 확인이에요.")

    st.subheader("출력별로 보기")
    col1, col2 = st.columns(2)
    run_key = col1.selectbox("출력", list(RUN_NAME), format_func=RUN_NAME.get)
    feat_key = col2.selectbox("센서 값", list(FEATS), format_func=lambda k: FEATS[k]["name"], index=3)
    win, r_ = PD["runs"][run_key]["win"], X["runs"][run_key]
    fe = FEATS[feat_key]
    xs, lo, hi = [], [], []
    for i, w in enumerate(win):
        if r_["Z"][i]:
            rr = ref(X["BL"], feat_key, w[1])
            xs.append(w[0] / 60)
            lo.append(rr["m"] - K * rr["s"])
            hi.append(rr["m"] + K * rr["s"])
    fig = go.Figure()
    fig.add_scatter(x=xs, y=lo, line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=xs, y=hi, fill="tonexty", line=dict(width=0), name="같은 높이 정상 범위", fillcolor="rgba(120,120,120,0.25)")
    fig.add_scatter(x=[w[0] / 60 for w in win], y=[w[fe["i"]] for w in win], name=fe["name"], line=dict(width=1))
    for g in r_["A"][feat_key]:
        fig.add_vrect(x0=win[g["s"]][0] / 60, x1=(win[g["e"]][0] + 10) / 60, fillcolor="crimson", opacity=0.2, line_width=0)
    if r_["onset"] is not None:
        fig.add_vline(x=r_["onset"] / 60, line_dash="dash", annotation_text="이상 시작(정한 기준)")
    fig.update_layout(height=360, xaxis_title="출력 시작 뒤 (분)", yaxis_title=f"{fe['name']} {fe['unit']}", margin=dict(t=20))
    st.plotly_chart(fig, width="stretch")

    st.subheader("이상 시작 시점은 우리가 정했어요")
    st.dataframe(
        pd.DataFrame([{"이상": RUN_NAME[k], "시작": mmss(v.get("visible", v["t"])), "정한 기준": v["rule"]} for k, v in PD["labels"].items()]),
        width="stretch", hide_index=True,
    )
    st.caption("출처: Szydlo T. et al. (2021) Dataset for Anomalies Detection in 3D Printing, ICCS 2021 · CC BY 4.0")
