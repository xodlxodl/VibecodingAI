"""보험 계리 (담당: 수학과 A)
빈도 GLM × 심도 GLM → 순보험료 → 검증 A/E, 지급준비금(Chain Ladder).
예시 데이터는 freMTPL2 구조를 흉내 낸 시뮬레이션이에요. HTML과 난수 생성기가 달라 숫자는 조금 달라요.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
import statsmodels.formula.api as smf
import streamlit as st

st.set_page_config(page_title="보험 계리", layout="wide")

AGE = ["18–25", "26–35", "36–50", "51–65", "66 이상"]
VEH = ["0–4년", "5–9년", "10년 이상"]
BM = ["50 (최우량)", "51–70", "71–100", "101 이상"]
AREA = list("ABCDEF")
BASE = {"age": AGE[2], "veh": VEH[1], "bm": BM[0], "area": "C"}
LABEL = {"age": "운전자 나이대", "veh": "차량 나이", "bm": "보너스-말러스", "area": "지역 밀도 등급"}
RAA = [
    [5012, 8269, 10907, 11805, 13539, 16181, 18009, 18608, 18662, 18834],
    [106, 4285, 5396, 10666, 13782, 15599, 15496, 16169, 16704],
    [3410, 8992, 13873, 16141, 18735, 22214, 22863, 23466],
    [5655, 11555, 15766, 21266, 23425, 26083, 27067],
    [1092, 9565, 15836, 22169, 25955, 26180],
    [1513, 6445, 11702, 12935, 15852],
    [557, 4020, 10946, 12314],
    [1351, 6947, 13112],
    [3133, 5395],
    [2063],
]


def add_bands(df):
    df = df.copy()
    df["age"] = pd.cut(df["DrivAge"], [0, 25, 35, 50, 65, 200], labels=AGE).astype(str)
    df["veh"] = pd.cut(df["VehAge"], [-1, 4, 9, 500], labels=VEH).astype(str)
    df["bm"] = pd.cut(df["BonusMalus"], [0, 50, 70, 100, 1000], labels=BM).astype(str)
    df["area"] = df["Area"].astype(str)
    return df


@st.cache_data
def simulate(n=50000, seed=678013):
    r = np.random.default_rng(seed)
    expo = np.where(r.random(n) < 0.42, 1.0, np.round(r.uniform(0.02, 1, n), 2))
    age = np.minimum(90, np.round(18 + r.gamma(4.2, 6.6, n))).astype(int)
    veh = np.minimum(25, np.round(r.gamma(1.6, 4.4, n))).astype(int)
    young = age <= 25
    bm = np.minimum(230, np.round(50 + r.gamma(1.3, np.where(young, 26, 16)))).astype(int)
    bm = np.where(r.random(n) > np.where(young, 0.15, 0.6), bm, 50)
    area = r.choice(6, n, p=np.array([15, 11, 28, 22, 20, 4]) / 100)
    ab = np.digitize(age, [26, 36, 51, 66])
    vb = np.digitize(veh, [5, 10])
    bb = np.digitize(bm, [51, 71, 101])
    lam = 0.098 * np.array([2.05, 1.22, 1, 0.93, 1.06])[ab] * np.array([1.12, 1, 0.84])[vb]
    lam = lam * np.array([0.82, 1.28, 1.78, 2.55])[bb] * np.array([0.82, 0.89, 0.96, 1.05, 1.16, 1.3])[area]
    claims = np.minimum(4, r.poisson(lam * r.gamma(2.2, 1 / 2.2, n) * expo))
    freq = pd.DataFrame({"IDpol": np.arange(1, n + 1), "ClaimNb": claims, "Exposure": expo, "Area": np.array(AREA)[area], "VehAge": veh, "DrivAge": age, "BonusMalus": bm})

    mu = 1500 * np.array([1.35, 1.08, 1, 1.05, 1.18])[ab] * np.array([1.3, 1, 0.78])[vb] * np.array([0.85, 0.92, 1, 1.06, 1.15, 1.3])[area]
    ids, amounts = [], []
    for i in np.nonzero(claims)[0]:
        if r.random() < 0.06:  # 청구건수는 있는데 금액 기록이 없는 계약
            continue
        for _ in range(claims[i]):
            u = r.random()
            if u < 0.06:
                v = 1128.12  # 같은 금액 몰림
            elif u < 0.075:
                v = np.exp(np.log(18000) + 0.9 * r.normal())  # 고액 손해
            else:
                v = r.gamma(1.1, mu[i] / 1.1)
            ids.append(i + 1)
            amounts.append(round(max(1.0, v), 2))
    return freq, pd.DataFrame({"IDpol": ids, "ClaimAmount": amounts})


@st.cache_data
def fit_all(freq, sev):
    amt = sev.groupby("IDpol")["ClaimAmount"].agg(["sum", "count"]).rename(columns={"sum": "S", "count": "nAmt"})
    df = add_bands(freq).merge(amt, how="left", left_on="IDpol", right_index=True).fillna({"S": 0, "nAmt": 0})
    r = np.random.default_rng(2026)
    df["valid"] = r.random(len(df)) < 0.25  # IDpol 기준 25%
    train, valid = df[~df["valid"]], df[df["valid"]]
    terms = " + ".join(f'C({k}, Treatment("{BASE[k]}"))' for k in BASE)
    f_model = smf.glm(f"ClaimNb ~ {terms}", train, family=sm.families.Poisson(), offset=np.log(train["Exposure"])).fit()
    sv = train[train["nAmt"] > 0].copy()
    sv["avg"] = sv["S"] / sv["nAmt"]
    s_model = smf.glm(f"avg ~ {terms}", sv, family=sm.families.Gamma(sm.families.links.Log()), freq_weights=sv["nAmt"]).fit()
    phi = float((f_model.resid_pearson ** 2).sum() / f_model.df_resid)
    return dict(df=df, train=train, valid=valid, f=f_model, s=s_model, phi=phi, nSev=len(sv))


def rel_table(model, phi=1.0):
    ci = model.conf_int()
    rows = []
    for k in BASE:
        for name in model.params.index:
            if name.startswith(f"C({k},"):
                level = name.split("[T.")[1].rstrip("]")
                b, se = model.params[name], model.bse[name] * np.sqrt(phi)
                rows.append(dict(요인=LABEL[k], 수준=level, 배수=np.exp(b), 하한=np.exp(b - 1.96 * se), 상한=np.exp(b + 1.96 * se)))
    return pd.DataFrame(rows).round(3)


def rel_chart(tbl):
    y = tbl["요인"] + " · " + tbl["수준"]
    fig = go.Figure(go.Scatter(x=tbl["배수"], y=y, mode="markers", error_x=dict(type="data", symmetric=False, array=tbl["상한"] - tbl["배수"], arrayminus=tbl["배수"] - tbl["하한"])))
    fig.add_vline(x=1, line_dash="dash")
    fig.update_xaxes(type="log", title="기준 수준 대비 배수 (로그 눈금)")
    fig.update_layout(height=460, margin=dict(t=20))
    return fig


def chain_ladder(tri):
    n = len(tri)
    ldf = []
    for j in range(n - 1):
        up = sum(r[j + 1] for r in tri if len(r) > j + 1)
        down = sum(r[j] for r in tri if len(r) > j + 1)
        ldf.append(up / down)
    rows = []
    for i, r in enumerate(tri):
        latest = r[-1]
        ult = latest * float(np.prod(ldf[len(r) - 1:]))
        rows.append({"사고연도": 1981 + i, "최신 누적": latest, "최종 추정": round(ult), "미지급(IBNR)": round(ult - latest)})
    return ldf, pd.DataFrame(rows)


# ---------------------------------------------------------------- 화면
st.caption("보험 계리 팩 · 담당: 수학과 A · 팀장")
st.title("요율 산출(빈도·심도 GLM)과 지급준비금(IBNR) 산출")

with st.expander("데이터 올리기 (선택): 빈도 CSV와 심도 CSV를 IDpol로 연결"):
    st.caption("빈도: IDpol, ClaimNb, Exposure, Area, VehAge, DrivAge, BonusMalus · 심도: IDpol, ClaimAmount")
    up_f = st.file_uploader("빈도 CSV", type="csv")
    up_s = st.file_uploader("심도 CSV", type="csv")

freq, sev = simulate()
source = "예시 데이터(시뮬레이션)"
if up_f and up_s:
    try:
        f2, s2 = pd.read_csv(up_f), pd.read_csv(up_s)
        need_f = {"IDpol", "ClaimNb", "Exposure", "Area", "VehAge", "DrivAge", "BonusMalus"}
        need_s = {"IDpol", "ClaimAmount"}
        if need_f - set(f2.columns) or need_s - set(s2.columns):
            st.error(f"빠진 열: {sorted((need_f - set(f2.columns)) | (need_s - set(s2.columns)))}")
        else:
            freq, sev = f2[f2["Exposure"] > 0], s2[s2["ClaimAmount"] > 0]
            source = "올린 데이터"
    except Exception as e:
        st.error(f"CSV를 읽지 못했어요: {e}")
st.info(f"지금 보는 자료: {source} · 계약 {len(freq):,}건, 청구 금액 {len(sev):,}건")

M = fit_all(freq, sev)
df, train, valid = M["df"], M["train"], M["valid"]

t_exp, t_freq, t_sev, t_pp, t_val, t_res = st.tabs(["경험통계", "빈도 GLM", "심도 GLM", "순보험료", "검증 · A/E", "지급준비금(IBNR)"])

with t_exp:
    a, b, c, d = st.columns(4)
    a.metric("계약 수", f"{len(df):,}")
    b.metric("관측 빈도", f"{df['ClaimNb'].sum() / df['Exposure'].sum():.4f} 건/년")
    c.metric("관측 심도", f"€{df['S'].sum() / max(df['nAmt'].sum(), 1):,.0f}")
    d.metric("고액 청구(€50,000 이상)", f"{(sev['ClaimAmount'] >= 50000).sum():,}건")
    orphan = (~sev["IDpol"].isin(freq["IDpol"])).sum()
    no_amt = ((df["ClaimNb"] > 0) & (df["nAmt"] == 0)).sum()
    st.caption(f"자료 점검: 빈도 표에 없는 청구 {orphan:,}건 · 청구건수는 있지만 금액이 없는 계약 {no_amt:,}건")
    key = st.selectbox("집단", list(LABEL), format_func=LABEL.get)
    g = df.groupby(key).agg(노출=("Exposure", "sum"), 청구건수=("ClaimNb", "sum"), 금액=("S", "sum"), 금액건수=("nAmt", "sum"))
    g["빈도"] = g["청구건수"] / g["노출"]
    g["심도"] = g["금액"] / g["금액건수"].replace(0, np.nan)
    st.dataframe(g.round(4), width="stretch")
    st.bar_chart(g["빈도"])

with t_freq:
    st.subheader("포아송 빈도 GLM 상대도")
    st.latex(r"\log \lambda_i = \log t_i + \beta_0 + \sum_k \beta_k x_{ik}")
    st.metric("과산포 φ (피어슨)", f"{M['phi']:.3f}")
    tbl = rel_table(M["f"], max(M["phi"], 1))
    st.plotly_chart(rel_chart(tbl), width="stretch")
    st.dataframe(tbl, width="stretch", hide_index=True)
    st.caption("구간은 √φ배 넓힌 준포아송 95% 구간이에요. 구간이 1을 지나면 차이가 확실하지 않아요.")

with t_sev:
    st.subheader("감마 심도 GLM 상대도")
    st.caption(f"금액이 있는 학습 계약 {M['nSev']:,}건 · 계약별 평균 금액을 건수로 가중")
    tbl = rel_table(M["s"], float(M["s"].scale))
    st.plotly_chart(rel_chart(tbl), width="stretch")
    st.dataframe(tbl, width="stretch", hide_index=True)

with t_pp:
    st.subheader("계약 하나의 순보험료")
    cols = st.columns(4)
    pick = {k: cols[i].selectbox(LABEL[k], {"age": AGE, "veh": VEH, "bm": BM, "area": AREA}[k], index={"age": AGE, "veh": VEH, "bm": BM, "area": AREA}[k].index(BASE[k])) for i, k in enumerate(BASE)}
    x = pd.DataFrame([pick])
    lam = float(M["f"].predict(x, offset=np.zeros(1)).iloc[0])
    mu = float(M["s"].predict(x).iloc[0])
    a, b, c = st.columns(3)
    a.metric("예상 빈도 λ", f"{lam:.4f} 건/년")
    b.metric("예상 심도 μ", f"€{mu:,.0f}")
    c.metric("순보험료 λ × μ", f"€{lam * mu:,.1f}/계약년")
    st.caption("빈도와 심도가 서로 독립이라는 가정 아래 곱해요. 사업비·이익은 들어 있지 않아요.")

with t_val:
    st.subheader("검증자료 A/E")
    v = valid.copy()
    v["lam"] = M["f"].predict(v, offset=np.log(v["Exposure"]))
    v["pure"] = v["lam"] * M["s"].predict(v)
    a, b, c = st.columns(3)
    a.metric("검증 계약", f"{len(v):,}건 (25%)")
    b.metric("A/E 건수", f"{v['ClaimNb'].sum() / v['lam'].sum():.3f}")
    c.metric("A/E 금액", f"{v['S'].sum() / v['pure'].sum():.3f}")
    key2 = st.selectbox("A/E 집단", list(LABEL), format_func=LABEL.get, key="ae_key")
    g = v.groupby(key2).agg(건수=("ClaimNb", "sum"), 예상건수=("lam", "sum"), 금액=("S", "sum"), 예상금액=("pure", "sum"))
    g["A/E 건수"], g["A/E 금액"] = g["건수"] / g["예상건수"], g["금액"] / g["예상금액"]
    st.dataframe(g.round(3), width="stretch")
    st.plotly_chart(go.Figure(go.Bar(x=g.index, y=g["A/E 건수"])).add_hline(y=1, line_dash="dash").update_layout(height=300, margin=dict(t=20)), width="stretch")
    st.download_button("계약별 예측 CSV 내려받기", v[["IDpol", "ClaimNb", "Exposure", "S", "lam", "pure"]].to_csv(index=False).encode("utf-8-sig"), "validation_predictions.csv", "text/csv")

with t_res:
    st.subheader("Chain Ladder 지급준비금 · RAA 삼각형")
    ldf, res = chain_ladder(RAA)
    a, b = st.columns(2)
    a.metric("총 미지급 추정", f"{res['미지급(IBNR)'].sum():,}")
    b.metric("최종 손해 합계", f"{res['최종 추정'].sum():,}")
    st.dataframe(pd.DataFrame([np.round(ldf, 4)], index=["가중평균 LDF"], columns=[f"{i + 1}→{i + 2}" for i in range(len(ldf))]), width="stretch")
    st.dataframe(res, width="stretch", hide_index=True)
    st.caption("RAA는 공개 벤치마크로, 총 미지급 약 52,135가 알려진 값이에요. BF·Cape Cod·Mack·사후검증은 아직 옮기지 않았어요.")
