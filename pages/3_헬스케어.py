"""헬스케어 (담당: 식품영양학과)
위험군 분류 → 대사증후군 유형(군집) → 당뇨 고위험군 선별 → 개선 시나리오.
예시 데이터는 건강검진정보 구조를 흉내 낸 시뮬레이션이에요. HTML과 난수 생성기가 달라 숫자는 조금 달라요.
진단이 아니라 선별 도구이며, 기준은 학회 원문으로 확인한 뒤 출처와 연도를 적어야 해요.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="헬스케어", layout="wide")

BANDS = [("20대", 20, 29), ("30대", 30, 39), ("40대", 40, 49), ("50대", 50, 59), ("60대", 60, 69), ("70대 이상", 70, 200)]
LEVEL = {0: "정상", 1: "주의", 2: "위험"}
ITEMS = {"bp": "혈압", "fbg": "공복혈당", "tc": "총콜레스테롤", "bmi": "BMI", "waist": "허리둘레"}
RULES = pd.DataFrame(
    [
        ("혈압", "120/80 미만", "120~139 또는 80~89", "140 또는 90 이상", "대한고혈압학회"),
        ("공복혈당", "100 미만", "100~125", "126 이상", "대한당뇨병학회"),
        ("총콜레스테롤", "200 미만", "200~239", "240 이상", "한국지질·동맥경화학회"),
        ("BMI", "23 미만", "23~24.9", "25 이상", "대한비만학회"),
        ("허리둘레", "남 90·여 85 미만", "-", "남 90·여 85 이상", "대한비만학회"),
    ],
    columns=["항목", "정상", "주의", "위험", "출처"],
)
METS_NAMES = {"m_waist": "허리둘레", "m_tg": "중성지방", "m_hdl": "HDL 콜레스테롤", "m_bp": "혈압", "m_fbg": "공복혈당"}
CLUSTER_FEATS = {"waist": "허리둘레", "tg": "중성지방", "hdl": "HDL", "sbp": "수축기 혈압", "fbg": "공복혈당"}
ADVICE = {
    "waist": "총열량을 줄이고 정제 탄수화물과 야식을 줄여요.",
    "tg": "단순당과 음주를 줄이고 등푸른 생선을 늘려요.",
    "hdl": "트랜스지방을 피하고 불포화지방과 유산소 운동을 늘려요.",
    "sbp": "나트륨을 줄이고 채소·과일로 칼륨을 늘려요.",
    "fbg": "통곡물과 식이섬유를 늘리고 식사 시간을 규칙적으로 해요.",
}
DM_FEATS = ["age", "male", "bmi", "waist", "sbp", "dbp", "tg", "hdl", "ggt", "smoker", "drink"]


@st.cache_data
def simulate(n=10000, seed=2):
    r = np.random.default_rng(seed)
    g = lambda: r.standard_normal(n)
    male = (r.random(n) < 0.5).astype(int)
    codes = np.arange(5, 19)
    w = np.array([4, 5.5, 7, 8, 9, 10.5, 11, 11, 10, 8.5, 6.5, 4.5, 2.8, 1.7])
    code = r.choice(codes, n, p=w / w.sum())
    age = 20 + (code - 5) * 5 + 2 + r.random(n)
    bmi = np.clip(22.6 + 1.2 * male + 0.035 * (age - 45) - 0.0009 * (age - 45) ** 2 + 3.2 * g(), 15, 42)
    u = r.random(n)
    smoke = np.where(male == 1, np.select([u < 0.42, u < 0.72], [1, 2], 3), np.select([u < 0.9, u < 0.95], [1, 2], 3))
    drink = (r.random(n) < np.where(male == 1, 0.7, 0.46)).astype(int)
    waist = np.clip(np.where(male == 1, 2.55 * bmi + 23.5, 2.4 * bmi + 21) + 0.06 * (age - 45) + 3.8 * g(), 55, 140)
    tg = np.clip(np.exp(np.log(102) + 0.045 * (bmi - 23) + 0.28 * male + 0.14 * drink + 0.003 * (age - 45) + 0.45 * g()), 25, 1200)
    hdl = np.clip(58 - 7 * male - 0.85 * (bmi - 23) + 2.5 * drink - 1.5 * (smoke == 3) + 11 * g(), 18, 130)
    tc = np.clip(192 + 0.35 * (age - 45) + 1.3 * (bmi - 23) - 4 * male + 34 * g(), 90, 400)
    ggt = np.clip(np.exp(np.log(21) + 0.5 * male + 0.45 * drink + 0.035 * (bmi - 23) + 0.004 * (age - 45) + 0.55 * g()), 5, 900)
    sbp = np.round(np.clip(113 + 0.46 * (age - 40) + 1.25 * (bmi - 23) + 5 * male + 11.5 * g(), 80, 220))
    dbp = np.round(np.clip(72 + 0.12 * (age - 40) + 0.75 * (bmi - 23) + 4 * male + 0.1 * (sbp - 120) + 7 * g(), 45, 130))
    lin = -5.75 + 0.04 * age + 0.11 * (bmi - 23) + 0.018 * (waist - 82) + 0.3 * male + 0.0022 * (tg - 120) - 0.01 * (hdl - 52) + 0.0028 * (ggt - 30) + 0.25 * (smoke == 3) + 0.005 * (sbp - 120) + 0.55 * g()
    dm = r.random(n) < 1 / (1 + np.exp(-lin))
    fbg = np.round(np.where(dm, 126 + r.gamma(1.6, 18, n), np.clip(89 + 0.16 * (age - 40) + 0.75 * (bmi - 23) + 2 * male + 8.5 * g(), 65, 125)))
    missing = r.random(n) < np.where(age < 40, 0.55, 0.25)
    df = pd.DataFrame({
        "sex": np.where(male == 1, 1, 2), "age": np.round(age), "bmi": np.round(bmi, 1), "waist": np.round(waist, 1),
        "sbp": sbp, "dbp": dbp, "fbg": fbg, "tc": np.round(tc), "tg": np.round(tg), "hdl": np.round(hdl), "ggt": np.round(ggt),
        "smoke": smoke, "drink": drink,
    })
    df.loc[missing, ["tc", "tg", "hdl"]] = np.nan
    return df


def add_levels(df):
    df = df.copy()
    male = df["sex"] == 1
    df["lv_bp"] = np.select([(df["sbp"] >= 140) | (df["dbp"] >= 90), (df["sbp"] >= 120) | (df["dbp"] >= 80)], [2, 1], 0)
    df["lv_fbg"] = np.select([df["fbg"] >= 126, df["fbg"] >= 100], [2, 1], 0)
    df["lv_tc"] = np.select([df["tc"] >= 240, df["tc"] >= 200], [2, 1], 0).astype(float)
    df.loc[df["tc"].isna(), "lv_tc"] = np.nan
    df["lv_bmi"] = np.select([df["bmi"] >= 25, df["bmi"] >= 23], [2, 1], 0)
    df["lv_waist"] = np.where(df["waist"] >= np.where(male, 90, 85), 2, 0)
    df["worst"] = df[[f"lv_{k}" for k in ITEMS]].max(axis=1)
    df["band"] = pd.cut(df["age"], [b[1] - 1 for b in BANDS] + [200], labels=[b[0] for b in BANDS])
    return df


def add_mets(df):
    df = df.copy()
    male = df["sex"] == 1
    df["m_waist"] = df["waist"] >= np.where(male, 90, 85)
    df["m_tg"] = df["tg"] >= 150
    df["m_hdl"] = df["hdl"] < np.where(male, 40, 50)
    df["m_bp"] = (df["sbp"] >= 130) | (df["dbp"] >= 85)
    df["m_fbg"] = df["fbg"] >= 100
    df["judged"] = df[["waist", "tg", "hdl", "sbp", "dbp", "fbg"]].notna().all(axis=1)
    df["m_count"] = df[list(METS_NAMES)].sum(axis=1)
    df["mets"] = df["judged"] & (df["m_count"] >= 3)
    return df


@st.cache_data
def diabetes_model(df):
    d = df[df[["tg", "hdl"]].notna().all(axis=1)].copy()
    d["male"] = (d["sex"] == 1).astype(int)
    d["smoker"] = (d["smoke"] == 3).astype(int)
    y = (d["fbg"] >= 126).astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(d[DM_FEATS], y, test_size=0.3, random_state=7, stratify=y)
    scaler = StandardScaler().fit(X_tr)
    model = LogisticRegression(max_iter=1000).fit(scaler.transform(X_tr), y_tr)
    p = model.predict_proba(scaler.transform(X_te))[:, 1]
    return d, scaler, model, roc_auc_score(y_te, p), brier_score_loss(y_te, p), len(y_tr), len(y_te)


def mask(k, n):
    """5명 미만 칸은 가려요."""
    return "가림" if 0 < n < 5 or 0 < k < 5 else f"{k / n:.1%}" if n else "-"


# ---------------------------------------------------------------- 화면
st.caption("헬스케어 팩 · 담당: 식품영양학과")
st.title("검진 데이터로 위험군을 나누고, 대사증후군 유형과 당뇨 고위험군에 맞춘 영양 중재까지")
st.warning("이 화면은 진단이 아니라 선별 시연이에요. 예시 숫자는 시뮬레이션 값이에요.")

with st.expander("데이터 올리기 (선택)"):
    st.caption("필요한 열: sex(1남·2여), age, bmi, waist, sbp, dbp, fbg, tc, tg, hdl, ggt, smoke(3=현재 흡연), drink(0·1)")
    st.caption("국민건강보험공단 원본은 열 이름이 달라서, 위 이름으로 바꾼 뒤 올려요.")
    up = st.file_uploader("검진 CSV", type="csv")

data = simulate()
if up:
    try:
        tmp = pd.read_csv(up)
        need = {"sex", "age", "bmi", "waist", "sbp", "dbp", "fbg", "tc", "tg", "hdl", "ggt", "smoke", "drink"}
        if need - set(tmp.columns):
            st.error(f"빠진 열: {sorted(need - set(tmp.columns))}")
        else:
            data = tmp[list(need)].dropna(subset=["age", "sex"])
            st.success(f"{len(data):,}명을 읽었어요.")
    except Exception as e:
        st.error(f"CSV를 읽지 못했어요: {e}")

df = add_mets(add_levels(data))
t_risk, t_mets, t_dm = st.tabs(["① 위험군 분류", "② 대사증후군 유형", "③ 당뇨 고위험군 선별"])

with t_risk:
    a, b, c = st.columns(3)
    a.metric("검진자", f"{len(df):,}명")
    b.metric("위험 등급 비율", f"{(df['worst'] == 2).mean():.1%}")
    c.metric("지질 검사 없음", f"{df['tc'].isna().mean():.1%}")
    st.subheader("임상 기준표")
    st.dataframe(RULES, width="stretch", hide_index=True)
    st.caption("사람마다 가장 나쁜 항목의 등급을 그 사람의 등급으로 삼아요. 학회 원문 확인 후 연도를 적어야 해요.")
    st.subheader("연령대별 위험군 비율")
    tab = pd.crosstab(df["band"], df["worst"].map(LEVEL))
    tab = tab.reindex(columns=[v for v in LEVEL.values() if v in tab.columns])
    share = tab.div(tab.sum(axis=1), axis=0)
    colors = {"정상": "#4C9F70", "주의": "#E5A93C", "위험": "#C8553D"}
    fig = go.Figure([go.Bar(y=share.index, x=share[c_], name=c_, orientation="h", marker_color=colors[c_]) for c_ in share.columns])
    fig.update_layout(barmode="stack", height=320, xaxis_tickformat=".0%", margin=dict(t=20))
    st.plotly_chart(fig, width="stretch")
    if (tab < 5).any().any():
        st.caption("5명 미만 칸이 있어요. 공개할 때는 해당 칸을 가려야 해요.")
    st.subheader("항목별 위험 비율")
    st.bar_chart(pd.Series({n: (df[f"lv_{k}"] == 2).sum() / df[f"lv_{k}"].notna().sum() for k, n in ITEMS.items()}))

with t_mets:
    used = df[df["judged"]]
    pos = used[used["mets"]]
    a, b, c = st.columns(3)
    a.metric("대사증후군 비율", f"{len(pos) / max(len(used), 1):.1%}")
    b.metric("판정한 사람", f"{len(used):,}명")
    c.metric("값이 빠져 제외", f"{len(df) - len(used):,}명")
    st.caption("5개 기준 중 3개 이상 · NCEP-ATP III에 한국인 허리둘레 기준을 적용한 출발점이에요. 약 복용 기준은 적용하지 않았어요.")
    st.dataframe(
        pd.DataFrame([{"기준": n, "전체에서 해당": f"{used[k].mean():.1%}", "대사증후군 중 해당": f"{pos[k].mean():.1%}"} for k, n in METS_NAMES.items()]),
        width="stretch", hide_index=True,
    )
    rate_rows = []
    for b_name, _, _ in BANDS:
        row = {"연령대": b_name}
        for sx, label in ((1, "남성"), (2, "여성")):
            g = used[(used["band"] == b_name) & (used["sex"] == sx)]
            row[label] = mask(int(g["mets"].sum()), len(g))
        rate_rows.append(row)
    st.subheader("연령대·성별 대사증후군 비율")
    st.dataframe(pd.DataFrame(rate_rows), width="stretch", hide_index=True)

    st.subheader("대사증후군인 사람만 유형 분류")
    k = st.slider("유형 수 (k)", 2, 6, 3)
    if len(pos) >= k * 5:
        feats = list(CLUSTER_FEATS)
        Z = StandardScaler().fit_transform(pos[feats])
        km = KMeans(n_clusters=k, n_init=10, random_state=2026).fit(Z)
        rows = []
        for c_ in range(k):
            center = km.cluster_centers_[c_]
            marks = [f for f, v in zip(feats, center) if (v <= -0.5 if f == "hdl" else v >= 0.5)]
            name = " · ".join(CLUSTER_FEATS[f] + (" 낮음" if f == "hdl" else " 높음") for f in marks) or "뚜렷한 특징 없음"
            n_c = int((km.labels_ == c_).sum())
            row = {"유형": name, "인원": "가림" if n_c < 5 else n_c}
            row.update({CLUSTER_FEATS[f]: round(v, 2) for f, v in zip(feats, center)})
            row["영양 중재"] = " ".join(ADVICE[f] for f in marks) or "전반적 식습관을 점검해요."
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.caption("숫자는 표준화한 중심값이에요. 유형 이름과 개수는 데이터마다 달라져요. 중재 문구는 식품영양학과 검토가 필요한 초안이에요.")
    else:
        st.info("대사증후군 판정자가 너무 적어 유형을 나누지 않았어요.")

with t_dm:
    d, scaler, model, auc, brier, n_tr, n_te = diabetes_model(df)
    a, b, c = st.columns(3)
    a.metric("AUC (시험 데이터)", f"{auc:.3f}")
    b.metric("Brier", f"{brier:.3f}")
    c.metric("학습 / 시험", f"{n_tr:,} / {n_te:,}")
    st.caption("정답 = 공복혈당 126 이상(1회 측정이라 의심 수준) · 공복혈당은 설명변수에서 뺐어요(누수 방지).")
    top = st.slider("고위험군 기준: 위험 점수 상위 %", 5, 30, 10)
    under = d[d["fbg"] < 126].copy()
    under["score"] = model.predict_proba(scaler.transform(under[DM_FEATS]))[:, 1]
    high = under[under["score"] >= under["score"].quantile(1 - top / 100)]
    st.metric("고위험군", f"{len(high):,}명", f"평균 위험 점수 {high['score'].mean():.1%}", delta_color="off")
    st.subheader("개선 시나리오")
    col1, col2, col3 = st.columns(3)
    d_w = col1.slider("허리둘레 감소(cm)", 0, 10, 5)
    d_b = col2.slider("BMI 감소", 0.0, 3.0, 1.0, 0.5)
    d_t = col3.slider("중성지방 감소(%)", 0, 30, 15)
    after = high.copy()
    after["waist"] -= d_w
    after["bmi"] -= d_b
    after["tg"] *= 1 - d_t / 100
    new = model.predict_proba(scaler.transform(after[DM_FEATS]))[:, 1].mean()
    a, b = st.columns(2)
    a.metric("변경 전 평균 점수", f"{high['score'].mean():.1%}")
    b.metric("변경 후 평균 점수", f"{new:.1%}", f"{(new - high['score'].mean()) * 100:.1f}%p", delta_color="inverse")
    st.caption("같은 모델에 수치를 바꿔 넣은 추정이에요. 실제로 위험이 그만큼 줄어든다는 인과 효과가 아니에요.")
