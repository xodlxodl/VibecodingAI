"""보험 계리 (담당: 수학과 A)
빈도 GLM × 심도 GLM → 순보험료 → 검증 A/E, 지급준비금(Chain Ladder).
예시 데이터는 freMTPL2 구조를 흉내 낸 시뮬레이션이에요. HTML과 같은 난수열로 만들어 같은 자료를 재현해요.
"""
import math
import re

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


REGIONS = [
    ("R24", 23.7), ("R82", 12.5), ("R93", 11.7), ("R11", 10.2), ("R53", 6.2), ("R52", 5.7), ("R91", 5.3), ("R72", 4.6),
    ("R31", 4.0), ("R54", 2.8), ("R73", 2.5), ("R41", 1.9), ("R25", 1.6), ("R26", 1.6), ("R23", 1.3), ("R22", 1.2),
    ("R83", 0.8), ("R74", 0.7), ("R94", 0.7), ("R21", 0.45), ("R42", 0.33), ("R43", 0.2),
]
MASK32 = 0xFFFFFFFF


def js_rng(seed):
    """HTML과 같은 난수열(mulberry32)을 만들어 같은 예시 데이터를 재현해요."""
    state = [seed & MASK32]

    def next_value():
        state[0] = (state[0] + 0x6D2B79F5) & MASK32
        a = state[0]
        t = ((a ^ (a >> 15)) * (1 | a)) & MASK32
        t = ((t + (((t ^ (t >> 7)) * (61 | t)) & MASK32)) & MASK32) ^ t
        return (t ^ (t >> 14)) / 4294967296

    return next_value


def js_round(x, digits=0):
    p = 10 ** digits
    value = math.floor(x * p + 0.5) / p
    return int(value) if digits == 0 else value


def gauss(r):
    u = v = 0.0
    while u == 0:
        u = r()
    while v == 0:
        v = r()
    return math.sqrt(-2 * math.log(u)) * math.cos(2 * math.pi * v)


def gamma_sample(k, theta, r):
    d = k - 1 / 3
    c = 1 / math.sqrt(9 * d)
    while True:
        while True:
            x = gauss(r)
            v = 1 + c * x
            if v > 0:
                break
        v = v * v * v
        u = r()
        if u < 1 - 0.0331 * x ** 4:
            return d * v * theta
        if math.log(u) < 0.5 * x * x + d * (1 - v + math.log(v)):
            return d * v * theta


def poisson(lam, r):
    if lam <= 0:
        return 0
    limit, k, p = math.exp(-lam), 0, 1.0
    while True:
        k += 1
        p *= r()
        if p <= limit:
            return k - 1


def pick_weighted(items, weights, r):
    u = r() * sum(weights)
    for item, w in zip(items, weights):
        u -= w
        if u <= 0:
            return item
    return items[-1]


def age_band(a):
    return 0 if a <= 25 else 1 if a <= 35 else 2 if a <= 50 else 3 if a <= 65 else 4


def veh_band(v):
    return 0 if v <= 4 else 1 if v <= 9 else 2


def bm_band(b):
    return 0 if b <= 50 else 1 if b <= 70 else 2 if b <= 100 else 3


def is_valid_id(pid):
    """IDpol로 정하는 25% 검증자료 (같은 계약은 늘 같은 쪽)"""
    h = int(round(abs(pid))) & MASK32
    h ^= h >> 16
    h = (h * 0x45D9F3B) & MASK32
    h ^= h >> 16
    h = (h * 0x45D9F3B) & MASK32
    h ^= h >> 16
    return h % 100 < 25


def add_bands(df):
    df = df.copy()
    df["age"] = pd.cut(df["DrivAge"], [0, 25, 35, 50, 65, 200], labels=AGE).astype(str)
    df["veh"] = pd.cut(df["VehAge"], [-1, 4, 9, 500], labels=VEH).astype(str)
    df["bm"] = pd.cut(df["BonusMalus"], [0, 50, 70, 100, 1000], labels=BM).astype(str)
    df["area"] = df["Area"].astype(str)
    if "Region" in df.columns:
        df["region"] = df["Region"].astype(str)
    return df


@st.cache_data
def simulate(n=50000, seed=678013):
    r = js_rng(seed)
    rr = js_rng(seed + 1)
    reg_eff = {k: math.exp(0.07 * gauss(rr)) for k, _ in REGIONS}
    reg_eff["R31"] *= 1.22
    reg_eff["R54"] *= 1.2
    reg_eff["R43"] *= 1.35
    rel_age, rel_veh = [2.05, 1.22, 1.0, 0.93, 1.06], [1.12, 1.0, 0.84]
    rel_bm, rel_area = [0.82, 1.28, 1.78, 2.55], [0.82, 0.89, 0.96, 1.05, 1.16, 1.3]
    reg_names, reg_w = [x[0] for x in REGIONS], [x[1] for x in REGIONS]
    rows = []
    for i in range(n):
        expo = 1.0 if r() < 0.42 else js_round(0.02 + 0.98 * r(), 2)
        age = min(90, js_round(18 + gamma_sample(4.2, 6.6, r)))
        veh = min(25, js_round(gamma_sample(1.6, 4.4, r)))
        bm = 50
        young = age <= 25
        if r() > (0.15 if young else 0.6):
            bm = min(230, js_round(50 + gamma_sample(1.3, 26 if young else 16, r)))
        area = pick_weighted(list(range(6)), [15, 11, 28, 22, 20, 4], r)
        reg = pick_weighted(reg_names, reg_w, r)
        lam = 0.098 * rel_age[age_band(age)] * rel_veh[veh_band(veh)] * rel_bm[bm_band(bm)] * rel_area[area] * reg_eff[reg]
        frail = gamma_sample(2.2, 1 / 2.2, r)
        k = poisson(lam * frail * expo, r)
        rows.append((i + 1, min(4, k), expo, AREA[area], int(veh), int(age), int(bm), reg))
    freq = pd.DataFrame(rows, columns=["IDpol", "ClaimNb", "Exposure", "Area", "VehAge", "DrivAge", "BonusMalus", "Region"])

    # 심도: HTML의 simulateSeverity와 같은 순서로 난수를 써요
    s = js_rng(4417)
    sev_age, sev_veh = [1.35, 1.08, 1.0, 1.05, 1.18], [1.3, 1.0, 0.78]
    sev_area = [0.85, 0.92, 1.0, 1.06, 1.15, 1.3]
    ids, amounts = [], []
    for pid, claims, _, area_name, veh, age, _, _ in rows:
        if not claims:
            continue
        if s() < 0.06:  # 청구건수는 있는데 금액 기록이 없는 계약
            continue
        kk = claims - 1 if claims >= 2 and s() < 0.15 else claims  # 금액 기록이 청구건수보다 적은 계약
        mu = 1500 * sev_age[age_band(age)] * sev_veh[veh_band(veh)] * sev_area[AREA.index(area_name)]
        for _ in range(kk):
            u = s()
            if u < 0.06:
                v = 1128.12  # 같은 금액 몰림
            elif u < 0.075:
                v = math.exp(math.log(18000) + 0.9 * gauss(s))  # 고액 손해
            else:
                v = gamma_sample(1.1, mu / 1.1, s)
            ids.append(pid)
            amounts.append(js_round(max(1.0, v), 2))
    for j in range(12):  # 빈도 표에 없는 ID
        ids.append(n + 101 + j * 7)
        amounts.append(js_round(max(1.0, gamma_sample(1.1, 1500 / 1.1, s)), 2))
    return freq, pd.DataFrame({"IDpol": ids, "ClaimAmount": amounts})


@st.cache_data
def fit_all(freq, sev):
    amt = sev.groupby("IDpol")["ClaimAmount"].agg(["sum", "count"]).rename(columns={"sum": "S", "count": "nAmt"})
    df = add_bands(freq).merge(amt, how="left", left_on="IDpol", right_index=True).fillna({"S": 0, "nAmt": 0})
    df["valid"] = df["IDpol"].map(is_valid_id)
    train, valid = df[~df["valid"]], df[df["valid"]]
    terms = " + ".join(f'C({k}, Treatment("{BASE[k]}"))' for k in BASE)
    # 빈도는 금액 기록이 있는 청구 수(nAmt)로 세요. 그래야 빈도 × 심도가 실제 금액과 같은 범위를 봐요.
    f_model = smf.glm(f"nAmt ~ {terms}", train, family=sm.families.Poisson(), offset=np.log(train["Exposure"])).fit()
    sv = train[train["nAmt"] > 0].copy()
    sv["avg"] = sv["S"] / sv["nAmt"]
    s_model = smf.glm(f"avg ~ {terms}", sv, family=sm.families.Gamma(sm.families.links.Log()), freq_weights=sv["nAmt"]).fit()
    phi = float((f_model.resid_pearson ** 2).sum() / f_model.df_resid)
    # 감마 φ의 자유도는 계약 수 기준이에요. freq_weights의 기본값은 건수 합 기준이라 직접 계산해요.
    m_sv = s_model.predict(sv)
    phi_s = float((sv["nAmt"] * ((sv["avg"] - m_sv) / m_sv) ** 2).sum() / (len(sv) - len(s_model.params)))
    return dict(df=df, train=train, valid=valid, f=f_model, s=s_model, phi=phi, phi_s=phi_s, nSev=len(sv))


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


# ---------------------------------------------------------------- 신뢰도 (뷸만-스트라우브)
def buhlmann_straub(groups):
    """groups: [(가중치 배열, 값 배열), ...]"""
    r = len(groups)
    m = [float(np.sum(w)) for w, _ in groups]
    xb = [float(np.sum(w * x)) / mi for (w, x), mi in zip(groups, m)]
    num = sum(float(np.sum(w * (x - xi) ** 2)) for (w, x), xi in zip(groups, xb))
    den = sum(len(w) - 1 for w, _ in groups)
    epv = num / den
    total = sum(m)
    xbar = sum(mi * xi for mi, xi in zip(m, xb)) / total
    between = sum(mi * (xi - xbar) ** 2 for mi, xi in zip(m, xb))
    vhm = (between - epv * (r - 1)) / (total - sum(mi * mi for mi in m) / total)
    k = epv / vhm if vhm > 0 else math.inf
    z = [mi / (mi + k) if vhm > 0 else 0.0 for mi in m]
    return dict(EPV=epv, VHM=vhm, k=k, Z=z, m=m, xb=xb, xbar=xbar)


def cred_example():
    data = {
        "A": [(1.0, 0), (0.5, 1), (1.0, 0), (1.0, 0)],
        "B": [(1.0, 1), (1.0, 0), (0.5, 0), (1.0, 1), (1.0, 0), (1.0, 0)],
        "C": [(1.0, 2), (0.5, 1), (1.0, 1)],
    }
    groups = [(np.array([p[0] for p in v]), np.array([p[1] / p[0] for p in v])) for v in data.values()]
    bs = buhlmann_straub(groups)
    rows = []
    for i, (key, val) in enumerate(data.items()):
        ae = sum(p[1] for p in val) / (bs["xbar"] * bs["m"][i])
        rows.append({"집단": key, "노출 m": bs["m"][i], "가중 평균 빈도": round(bs["xb"][i], 3), "Z": round(bs["Z"][i], 2), "A/E": round(ae, 2), "가중 A/E": round(bs["Z"][i] * ae + 1 - bs["Z"][i], 2)})
    return bs, rows


def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def poisson_upper(a, lam):
    """P(X >= a)"""
    if a <= 0:
        return 1.0
    if lam > 400:
        return 1 - norm_cdf((a - 0.5 - lam) / math.sqrt(lam))
    return max(0.0, 1 - sum(math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1)) for k in range(int(a))))


def poisson_lower(a, lam):
    """P(X <= a)"""
    if lam > 400:
        return norm_cdf((a + 0.5 - lam) / math.sqrt(lam))
    return min(1.0, sum(math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1)) for k in range(int(a) + 1)))


def val_groups(v, key):
    order = {"age": AGE, "veh": VEH, "bm": BM, "area": AREA}.get(key)
    rows, groups = [], []
    for name, part in v.groupby(key, observed=True):
        e = part["lam"].to_numpy(dtype=float)
        rows.append(dict(key=name, policies=len(part), expo=float(part["Exposure"].sum()), claims=int(part["nAmt"].sum()), E=float(e.sum()), amount=float(part["S"].sum()), ES=float(part["pure"].sum()), w=e, x=part["nAmt"].to_numpy(dtype=float) / e))
    rows.sort(key=(lambda o: order.index(o["key"])) if order else (lambda o: -o["expo"]))
    bs = buhlmann_straub([(o["w"], o["x"]) for o in rows])
    for j, o in enumerate(rows):
        o["aeN"], o["aeS"] = o["claims"] / o["E"], o["amount"] / o["ES"] if o["ES"] else float("nan")
        o["Z"] = bs["Z"][j]
        o["credAE"] = o["Z"] * o["aeN"] + 1 - o["Z"]
        o["pUp"], o["pDown"] = poisson_upper(o["claims"], o["E"]), poisson_lower(o["claims"], o["E"])
        if o["credAE"] >= 1.05 and o["pUp"] < 0.05:
            o["flag"] = "예상보다 높음"
        elif o["credAE"] <= 0.95 and o["pDown"] < 0.05:
            o["flag"] = "예상보다 낮음"
        else:
            o["flag"] = "자료 부족" if o["Z"] < 0.3 else "예상 범위"
    return rows, bs


# ---------------------------------------------------------------- 지급준비금: 업로드 해석
BASES = {"reported": "보고발생 (IncurLoss − BulkLoss)", "paid": "지급 (CumPaidLoss)", "incurred": "회사 IBNR 포함 발생 (IncurLoss)"}
METHODS = {"weighted": "가중평균 (전체)", "simple": "단순평균 (전체)", "recentWeighted": "최근 N년 가중평균", "recentSimple": "최근 N년 단순평균", "exHiLo": "최대·최소 제외", "geometric": "기하평균"}
LINES = {"B": "개인용 자동차 배상책임", "C": "상업용 자동차 배상책임", "D": "근로자재해보상", "F2": "의료과실 (배상청구기준)", "H1": "기타 배상책임 (사고발생기준)", "R1": "제조물책임 (사고발생기준)"}


def to_num(x):
    s = re.sub(r'[",\s]', "", "" if x is None else str(x))
    try:
        return float(s) if s and s.lower() != "nan" else float("nan")
    except ValueError:
        return float("nan")


def is_num(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def detect_lines(columns):
    out = []
    for c in columns:
        m = re.match(r"^IncurLoss_(.+)$", str(c).strip())
        if m and m.group(1) not in out:
            out.append(m.group(1))
    return out


@st.cache_data
def ingest(table, sfx):
    table = table.rename(columns=lambda c: str(c).strip())
    need = ["GRCODE", "AccidentYear", "DevelopmentLag", "DevelopmentYear", f"IncurLoss_{sfx}", f"CumPaidLoss_{sfx}", f"BulkLoss_{sfx}", f"EarnedPremNet_{sfx}"]
    missing = [c for c in need if c not in table.columns]
    if missing:
        return dict(ok=False, missing=missing, dup=0, dupKeys=[], recs=[])
    recs, seen, dup_keys = [], set(), []
    bad = lag_mismatch = dup = 0
    for r in table.to_dict("records"):
        o = dict(
            g=to_num(r["GRCODE"]), name=str(r.get("GRNAME", "") or "").strip(), ay=to_num(r["AccidentYear"]), dy=to_num(r["DevelopmentYear"]), lag=to_num(r["DevelopmentLag"]),
            inc=to_num(r[f"IncurLoss_{sfx}"]), paid=to_num(r[f"CumPaidLoss_{sfx}"]), bulk=to_num(r[f"BulkLoss_{sfx}"]), prem=to_num(r[f"EarnedPremNet_{sfx}"]),
            posted=to_num(r.get(f"PostedReserve97_{sfx}")),
        )
        if not all(is_num(o[k]) for k in ("g", "ay", "dy", "lag")):
            bad += 1
            continue
        if o["lag"] != o["dy"] - o["ay"] + 1:
            lag_mismatch += 1
        k = (o["g"], o["ay"], o["lag"])
        if k in seen:
            dup += 1
            if len(dup_keys) < 5:
                dup_keys.append("|".join(str(int(x)) for x in k))
        seen.add(k)
        recs.append(o)
    return dict(
        ok=dup == 0 and bool(recs), missing=[], recs=recs, bad=bad, lagMismatch=lag_mismatch, dup=dup, dupKeys=dup_keys, nRows=len(recs),
        nCompanies=len({o["g"] for o in recs}), ayRange=(min((o["ay"] for o in recs), default=0), max((o["ay"] for o in recs), default=0)), dyRange=(min((o["dy"] for o in recs), default=0), max((o["dy"] for o in recs), default=0)),
    )


def value_of(o, basis):
    return o["paid"] if basis == "paid" else o["inc"] - o["bulk"] if basis == "reported" else o["inc"]


def companies(data, basis, eval_year):
    m = {}
    for o in data["recs"]:
        c = m.setdefault(o["g"], dict(g=o["g"], name=o["name"], prem={}, nonpos=0, cells=0))
        if o["ay"] not in c["prem"] or o["lag"] == 1:
            c["prem"][o["ay"]] = o["prem"]
        if o["ay"] <= eval_year and o["dy"] <= eval_year:
            c["cells"] += 1
            if not value_of(o, basis) > 0:
                c["nonpos"] += 1
    out = []
    for c in m.values():
        pv = [v for ay, v in c["prem"].items() if ay <= eval_year]
        need = len(pv) * (len(pv) + 1) / 2
        prem_missing = sum(1 for v in pv if not v > 0)
        out.append(dict(g=c["g"], name=c["name"], premSum=sum(v for v in pv if v > 0), premMissing=prem_missing, nonpos=c["nonpos"], missingUpper=max(0, need - c["cells"]), clean=prem_missing == 0 and c["nonpos"] == 0 and c["cells"] >= need))
    out.sort(key=lambda c: -c["premSum"])
    default = next((c for c in out if c["clean"]), out[0] if out else None)
    return out, default["g"] if default else None


# ---------------------------------------------------------------- 진전삼각형 · LDF · 꼬리
def build_triangle(data, grcode, basis, eval_year):
    every = grcode == "ALL"
    recs = data["recs"] if every else [o for o in data["recs"] if o["g"] == grcode]
    ays = sorted({o["ay"] for o in recs if o["ay"] <= eval_year})
    n = len(ays)
    width = max([n] + [int(o["lag"]) for o in recs])
    mk = lambda w: [[None] * w for _ in ays]
    T = dict(ays=ays, n=n, L=width, basis=basis, upper=mk(n), paid=mk(n), rep=mk(n), full=mk(width), prem=[None] * n)

    def add(M, i, j, val):
        M[i][j] = M[i][j] + val if every and M[i][j] is not None else val

    prem_by, posted_by = {}, {}
    for o in recs:
        if o["ay"] not in ays:
            continue
        i, j = ays.index(o["ay"]), int(o["lag"]) - 1
        if j < 0 or j >= width:
            continue
        val, rp = value_of(o, basis), o["inc"] - o["bulk"]
        add(T["full"], i, j, val)
        if o["dy"] <= eval_year and j < n:
            add(T["upper"], i, j, val)
            add(T["paid"], i, j, o["paid"])
            add(T["rep"], i, j, rp)
        pk = (o["g"], o["ay"])
        if pk not in prem_by or o["lag"] == 1:
            prem_by[pk] = (i, o["prem"])
        if is_num(o["posted"]):
            posted_by[o["g"]] = o["posted"]
    for i, val in prem_by.values():
        T["prem"][i] = (T["prem"][i] or 0) + (val if is_num(val) else 0)
    T["posted"] = sum(posted_by.values()) if posted_by else None
    T["missing"], T["nonpos"], T["paidGtRep"] = [], 0, 0
    for i, ay in enumerate(ays):
        for j in range(min(int(eval_year - ay) + 1, n)):
            if T["upper"][i][j] is None:
                T["missing"].append(f"{int(ay)}년 진전 {j + 1}")
            else:
                if not T["upper"][i][j] > 0:
                    T["nonpos"] += 1
                if T["paid"][i][j] > T["rep"][i][j]:
                    T["paidGtRep"] += 1
    T["d"] = [max([j + 1 for j, val in enumerate(row) if val is not None], default=0) for row in T["upper"]]
    return T


def ata(T):
    F = [[None] * max(0, T["n"] - 1) for _ in T["upper"]]
    for i, row in enumerate(T["upper"]):
        for j in range(T["n"] - 1):
            a, b = row[j], row[j + 1]
            if a is not None and b is not None and a > 0:
                F[i][j] = b / a
    return F


def ldf_value(T, F, j, method, recent_n=3, excluded=()):
    idx = [i for i in range(len(F)) if F[i][j] is not None and (i, j) not in excluded]
    if method in ("recentWeighted", "recentSimple"):
        idx = idx[-recent_n:]
    f = [F[i][j] for i in idx]
    if not f:
        return None
    simple = sum(f) / len(f)
    if method in ("weighted", "recentWeighted"):
        den = sum(T["upper"][i][j] for i in idx)
        return sum(T["upper"][i][j + 1] for i in idx) / den if den > 0 else simple
    if method == "exHiLo":
        if len(f) < 3:
            return simple
        s = sorted(f)[1:-1]
        return sum(s) / len(s)
    if method == "geometric":
        return math.exp(sum(math.log(x) for x in f) / len(f)) if all(x > 0 for x in f) else simple
    return simple


def fit_tail(sel, from_j, kind):
    n = len(sel) + 1
    xf = (lambda j: math.log(j)) if kind == "invpow" else (lambda j: j)
    pts = [(xf(j), math.log(sel[j - 1] - 1)) for j in range(max(1, from_j), n) if sel[j - 1] > 1]
    if len(pts) < 2:
        return dict(ok=False, msg="f > 1 인 구간이 2개 이상 필요해요", tail=1.0)
    mx, my = sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)
    sxx, sxy = sum((p[0] - mx) ** 2 for p in pts), sum((p[0] - mx) * (p[1] - my) for p in pts)
    if not sxx > 0:
        return dict(ok=False, msg="적합할 수 없어요", tail=1.0)
    b = sxy / sxx
    a = my - b * mx
    if not b < 0:
        return dict(ok=False, msg="기울기 b ≥ 0이라 외삽하면 발산해요(적합 실패)", tail=1.0)
    tail, steps = 1.0, 0
    for k in range(n, n + 100):
        g = math.exp(a + b * xf(k))
        if g < 0.0001:
            break
        tail *= 1 + g
        steps += 1
    ss, st_ = sum((p[1] - (a + b * p[0])) ** 2 for p in pts), sum((p[1] - my) ** 2 for p in pts)
    return dict(ok=True, a=a, b=b, tail=tail, steps=steps, capped=steps >= 100, r2=1 - ss / st_ if st_ > 0 else float("nan"))


def cdfs(sel, tail):
    n = len(sel) + 1
    c = [0.0] * n
    c[n - 1] = tail
    for j in range(n - 2, -1, -1):
        c[j] = sel[j] * c[j + 1]
    return c


# ---------------------------------------------------------------- Chain Ladder · BF · Cape Cod
def reserve(T, cdf, elr=None, elr_mode="manual", decay=1.0):
    rows = []
    for i, ay in enumerate(T["ays"]):
        d = T["d"][i]
        c = cdf[d - 1]
        prem = T["prem"][i]
        rows.append(dict(i=i, ay=ay, d=d, C=T["upper"][i][d - 1], P=prem if prem and prem > 0 else None, c=c, pct=1 / c, paid=T["paid"][i][d - 1], rep=T["rep"][i][d - 1]))
    valid = [r for r in rows if r["P"]]
    used_prem = sum(r["P"] * r["pct"] for r in valid)
    elr_cc = sum(r["C"] for r in valid) / used_prem if valid else None

    def elr_k(r):
        if not r["P"]:
            return None
        if decay >= 1:
            return elr_cc
        a = sum(decay ** abs(x["i"] - r["i"]) * x["C"] for x in valid)
        b = sum(decay ** abs(x["i"] - r["i"]) * x["P"] * x["pct"] for x in valid)
        return a / b if b > 0 else None

    def out(u, r):
        return dict(ult=u, ibnr=u - r["rep"], unpaid=u - r["paid"], lr=u / r["P"] if r["P"] else None)

    for r in rows:
        r["CL"] = out(r["C"] * r["c"], r)
    mature = [r for r in rows if r["pct"] >= 0.9 and r["P"]]
    elr_mature = sum(r["CL"]["lr"] for r in mature) / len(mature) if mature else None
    for r in rows:
        ek = elr_k(r)
        e = ek if elr_mode == "cc" else elr_mature if elr_mode == "mature" else elr
        r["elrBF"], r["elrCC"] = e, ek
        r["BF"] = out(r["C"] + e * r["P"] * (1 - r["pct"]), r) if r["P"] and is_num(e) else None
        r["CC"] = out(r["C"] + ek * r["P"] * (1 - r["pct"]), r) if r["P"] and is_num(ek) else None

    def tot(k):
        rs = [r for r in rows if r[k]]
        return dict(ult=sum(r[k]["ult"] for r in rs), ibnr=sum(r[k]["ibnr"] for r in rs), unpaid=sum(r[k]["unpaid"] for r in rs), nAY=len(rs), skipped=len(rows) - len(rs)) if rs else None

    return dict(rows=rows, elrCC=elr_cc, elrMature=elr_mature, totals=dict(CL=tot("CL"), BF=tot("BF"), CC=tot("CC"), rep=sum(r["rep"] for r in rows), paid=sum(r["paid"] for r in rows), case=sum(r["rep"] - r["paid"] for r in rows)))


def project(T, sel):
    out = []
    for i, row in enumerate(T["upper"]):
        p = list(row)
        for j in range(T["d"][i], T["n"]):
            p[j] = p[j - 1] * sel[j - 1]
        out.append(p)
    return out


# ---------------------------------------------------------------- 사후검증
def backtest(T, res, exclude_tail=True):
    target = T["n"] if exclude_tail else T["L"]
    rows = []
    for r in res["rows"]:
        actual = T["full"][r["i"]][target - 1]
        if not is_num(actual):
            rows.append(dict(ay=r["ay"], actual=None))
            continue
        a_i, a_u = actual - r["rep"], actual - r["paid"]
        small = lambda x: abs(x) < 0.01 * abs(actual)  # 실제값이 최종의 1% 미만이면 오차율이 의미 없음

        def err(k):
            if not r[k]:
                return None
            return dict(ult=r[k]["ult"], err=(r[k]["ult"] - actual) / actual, ibnrErr=None if small(a_i) else (r[k]["ibnr"] - a_i) / abs(a_i), unpaidErr=None if small(a_u) else (r[k]["unpaid"] - a_u) / abs(a_u))

        rows.append(dict(ay=r["ay"], d=r["d"], actual=actual, actualIBNR=a_i, actualUnpaid=a_u, CL=err("CL"), BF=err("BF"), CC=err("CC")))
    ok = [r for r in rows if r["actual"] is not None]

    def tot(k):
        rs = [r for r in ok if r[k]]
        if not rs:
            return None
        ult, act = sum(r[k]["ult"] for r in rs), sum(r["actual"] for r in rs)
        ai, au = sum(r["actualIBNR"] for r in rs), sum(r["actualUnpaid"] for r in rs)
        ibnr, unpaid = ult - (act - ai), ult - (act - au)
        small = lambda x: abs(x) < 0.01 * abs(act)
        return dict(ult=ult, actual=act, err=(ult - act) / act, ibnr=ibnr, actualIBNR=ai, ibnrErr=None if small(ai) else (ibnr - ai) / abs(ai), unpaid=unpaid, actualUnpaid=au, unpaidErr=None if small(au) else (unpaid - au) / abs(au), nAY=len(rs))

    return dict(target=target, rows=rows, nAY=len(ok), missing=len(rows) - len(ok), CL=tot("CL"), BF=tot("BF"), CC=tot("CC"))


def multi_backtest(data, grcode, basis, method, recent_n, elr, elr_mode, decay, years):
    out = []
    for y in years:
        T = build_triangle(data, grcode, basis, y)
        if T["missing"] or T["n"] < 3:
            continue
        F = ata(T)
        sel = [ldf_value(T, F, j, method, recent_n) or 1.0 for j in range(T["n"] - 1)]
        bt = backtest(T, reserve(T, cdfs(sel, 1.0), elr, elr_mode, decay), True)
        out.append(dict(year=y, n=T["n"], CL=bt["CL"], BF=bt["BF"], CC=bt["CC"]))
    return out


# ---------------------------------------------------------------- Mack 표준오차
def mack(T):
    n, C = T["n"], T["upper"]
    f, s2, S = [], [], []
    for j in range(n - 1):
        idx = [i for i in range(len(C)) if C[i][j] is not None and C[i][j] > 0 and C[i][j + 1] is not None]
        a, b = sum(C[i][j] for i in idx), sum(C[i][j + 1] for i in idx)
        f.append(b / a if a > 0 else 1.0)
        S.append(a)
        s2.append(sum(C[i][j] * (C[i][j + 1] / C[i][j] - f[j]) ** 2 for i in idx) / (len(idx) - 1) if len(idx) > 1 else None)
    for j in range(n - 1):
        if s2[j] is None:
            s2[j] = min(s2[j - 1] ** 2 / s2[j - 2], s2[j - 2], s2[j - 1]) if j >= 2 and s2[j - 1] and s2[j - 2] else 0.0
    P = project(T, f)
    U = [r[n - 1] for r in P]
    rows = []
    for i, ay in enumerate(T["ays"]):
        d = T["d"][i]
        t = sum(s2[k] / f[k] ** 2 * (1 / P[i][k] + 1 / S[k]) for k in range(d - 1, n - 1))
        mse = U[i] ** 2 * t
        r = U[i] - C[i][d - 1]
        rows.append(dict(ay=ay, ult=U[i], ibnr=r, se=math.sqrt(mse), cv=math.sqrt(mse) / abs(r) if r else None, mse=mse))
    total_mse = sum(r["mse"] for r in rows)
    for i in range(n):
        younger = sum(U[i + 1:])
        t = sum(2 * s2[k] / f[k] ** 2 / S[k] for k in range(T["d"][i] - 1, n - 1))
        total_mse += U[i] * younger * t
    total_r = sum(r["ibnr"] for r in rows)
    return dict(f=f, sigma2=s2, rows=rows, total=dict(ult=sum(U), ibnr=total_r, se=math.sqrt(total_mse), cv=math.sqrt(total_mse) / abs(total_r) if total_r else None))


# ---------------------------------------------------------------- 검산 자료: 5×5 예제, RAA
EX5 = dict(
    ays=[1993, 1994, 1995, 1996, 1997],
    rep=[[3000, 4200, 4800, 5050, 5150], [3300, 4500, 5200, 5450, None], [3500, 4900, 5550, None, None], [3800, 5500, None, None, None], [4100, None, None, None, None]],
    prem=[7000, 7400, 7800, 8300, 8800], paid=[4950, 4900, 4300, 3300, 1500], tail=1.01, elr=0.70,
    ata=[[1.4000, 1.1429, 1.0521, 1.0198], [1.3636, 1.1556, 1.0481], [1.4000, 1.1327], [1.4474]],
    ldf=dict(weighted=[1.4044, 1.1434, 1.0500, 1.0198], simple=[1.4028, 1.1437, 1.0501, 1.0198], recentWeighted=[1.4057, 1.1434, 1.0500, 1.0198], recentSimple=[1.4037, 1.1437, 1.0501, 1.0198], exHiLo=[1.4000, 1.1429, 1.0501, 1.0198], geometric=[1.4024, 1.1437, 1.0501, 1.0198]),
    cdf=[1.7367, 1.2366, 1.0815, 1.0300, 1.0100], elrCC=0.7778,
    CL=[[5201.5, 51.5], [5613.5, 163.5], [6002.3, 452.3], [6801.1, 1301.1], [7120.3, 3020.3]],
    BF=[[5198.5, 48.5], [5600.9, 150.9], [5961.5, 411.5], [6611.5, 1111.5], [6712.9, 2612.9]],
    CC=[[5203.9, 53.9], [5617.6, 167.6], [6007.2, 457.2], [6735.0, 1235.0], [7003.3, 2903.3]],
    tot=dict(CL=[30738.7, 4988.7, 11788.7], BF=[30085.3, 4335.3, 11135.3], CC=[30567.1, 4817.1, 11617.1]), caseTot=6800,
)


def check_ex5():
    n = 5
    T = dict(ays=EX5["ays"], n=n, L=n, upper=EX5["rep"], rep=EX5["rep"], prem=EX5["prem"])
    T["paid"] = [[EX5["paid"][i] if val is not None and j == n - 1 - i else None for j, val in enumerate(r)] for i, r in enumerate(EX5["rep"])]
    T["d"] = [sum(1 for x in r if x is not None) for r in EX5["rep"]]
    checks = []

    def chk(label, got, want, dp):
        checks.append(dict(항목=label, 계산값=round(got, dp), 기대값=want, 결과="일치" if abs(round(got, dp) - want) < 10 ** -dp * 0.51 else "다름"))

    F = ata(T)
    for i, r in enumerate(EX5["ata"]):
        for j, val in enumerate(r):
            chk(f"진전계수 {EX5['ays'][i]} {j + 1}-{j + 2}", F[i][j], val, 4)
    for k, vals in EX5["ldf"].items():
        for j, val in enumerate(vals):
            chk(f"LDF {k} {j + 1}-{j + 2}", ldf_value(T, F, j, k), val, 4)
    cdf = cdfs([ldf_value(T, F, j, "weighted") for j in range(n - 1)], EX5["tail"])
    for j, val in enumerate(EX5["cdf"]):
        chk(f"CDF {j + 1}", cdf[j], val, 4)
    R = reserve(T, cdf, EX5["elr"], "manual")
    chk("Cape Cod ELR", R["elrCC"], EX5["elrCC"], 4)
    for k in ("CL", "BF", "CC"):
        for i, val in enumerate(EX5[k]):
            chk(f"{k} 최종 {EX5['ays'][i]}", R["rows"][i][k]["ult"], val[0], 1)
            chk(f"{k} IBNR {EX5['ays'][i]}", R["rows"][i][k]["ibnr"], val[1], 1)
        t = R["totals"][k]
        chk(f"{k} 최종 합계", t["ult"], EX5["tot"][k][0], 1)
        chk(f"{k} IBNR 합계", t["ibnr"], EX5["tot"][k][1], 1)
        chk(f"{k} 지급준비금 합계", t["unpaid"], EX5["tot"][k][2], 1)
    chk("개별추산액 합계", R["totals"]["case"], EX5["caseTot"], 1)
    return checks


def raa_benchmark():
    up = [r + [None] * (10 - len(r)) for r in RAA]
    T = dict(ays=[1981 + i for i in range(10)], n=10, L=10, upper=up, rep=up, paid=up, prem=[None] * 10, d=[len(r) for r in RAA])
    F = ata(T)
    R = reserve(T, cdfs([ldf_value(T, F, j, "weighted") for j in range(9)], 1.0))
    return R["totals"]["CL"]["ibnr"], mack(T)["total"]["se"]


# ---------------------------------------------------------------- 예시 자료: CAS 파일과 같은 열 구조의 가상 회사 6곳 (실제 회사·실제 값 아님)
SIM_CO = [
    dict(g=90011, name="가상 손해보험 A (대형)", base=120000, elr=0.74, sd=1, single=0),
    dict(g=90029, name="가상 손해보험 B (중형)", base=42000, elr=0.70, sd=1.4, single=1),
    dict(g=90037, name="가상 손해보험 C (보험료 결측)", base=30000, elr=0.76, sd=1.6, single=1, premZero=1988),
    dict(g=90045, name="가상 손해보험 D (0원 셀)", base=9000, elr=0.72, sd=2.6, single=1, zeroCell=(1990, 1)),
    dict(g=90053, name="가상 손해보험 E (개별추산 강화)", base=65000, elr=0.73, sd=1.1, single=0, strengthen=1995),
    dict(g=90061, name="가상 손해보험 F (소형)", base=3500, elr=0.69, sd=4, single=1),
]
RP = [0.655, 0.848, 0.922, 0.96, 0.979, 0.989, 0.9945, 0.9972, 0.9986, 0.9993]
PP = [0.37, 0.665, 0.815, 0.9, 0.946, 0.97, 0.983, 0.9905, 0.995, 0.9975]


@st.cache_data
def simulate_cas(seed=19971231):
    r = js_rng(seed)
    rows = []
    for co in SIM_CO:
        s, cell, unpaid97 = 0.035 * co["sd"], [], 0.0
        for k in range(10):
            ay = 1988 + k
            direct = js_round(co["base"] * 1.045 ** k * math.exp(0.03 * gauss(r)))
            ceded = js_round(direct * (0.07 + 0.02 * r()))
            net = direct - ceded
            elr = co["elr"] + 0.035 * math.sin(k * 0.9) + 0.03 * gauss(r)
            ult = net * elr * math.exp(s * gauss(r))
            rep = ult * RP[0] * math.exp(1.6 * s * gauss(r))
            paid = ult * PP[0] * math.exp(1.4 * s * gauss(r))
            for j in range(10):
                if j > 0:
                    rep += ult * (RP[j] - RP[j - 1]) * math.exp((2.5 * s + 0.08) * gauss(r))
                    paid += ult * (PP[j] - PP[j - 1]) * math.exp((2 * s + 0.05) * gauss(r))
                dy = ay + j
                rp = rep
                if co.get("strengthen") and dy >= co["strengthen"]:
                    rp = rep * (1 + 0.11 * min(1, (dy - co["strengthen"] + 1) / 2) * (1 - RP[j]) / (1 - RP[0]))
                rv = js_round(rp)
                pv = js_round(min(paid, rp * (0.985 + 0.013 * j / 9)) if co["sd"] < 3 else paid)
                if co.get("zeroCell") == (ay, j + 1):
                    rv = 0
                bulk = max(0, js_round(rv * (1 / RP[j] - 1) * math.exp(0.15 * gauss(r))))
                if dy == 1997:
                    unpaid97 += ult - pv
                cell.append([co["g"], co["name"], ay, dy, j + 1, rv + bulk, pv, bulk, direct, ceded, 0 if co.get("premZero") == ay else net, co["single"]])
        posted = js_round(unpaid97 * 1.08 * math.exp(0.05 * gauss(r)))
        rows += [c + [posted] for c in cell]
    columns = ["GRCODE", "GRNAME", "AccidentYear", "DevelopmentYear", "DevelopmentLag", "IncurLoss_B", "CumPaidLoss_B", "BulkLoss_B", "EarnedPremDIR_B", "EarnedPremCeded_B", "EarnedPremNet_B", "Single", "PostedReserve97_B"]
    return pd.DataFrame(rows, columns=columns).astype(str)


def pct(x, digits=1):
    return "-" if x is None else f"{x:+.{digits}%}"


def money(x):
    return "-" if x is None else f"{x:,.0f}"


def reserve_tab():
    st.subheader("지급준비금 · IBNR")
    st.caption("CAS Loss Reserving Database와 같은 열 구조의 파일을 쓰고, 회사와 평가연도를 골라 진전삼각형을 만들어요.")
    up = st.file_uploader("삼각형 CSV 올리기 (없으면 가상 회사 6곳 예시)", type="csv", key="res_file")
    st.caption("필수 열: GRCODE, AccidentYear, DevelopmentYear, DevelopmentLag, IncurLoss_종목, CumPaidLoss_종목, BulkLoss_종목, EarnedPremNet_종목 (종목 예: B). 한 행 = 회사·사고연도·진전연수 한 칸이에요.")
    table = simulate_cas()
    source = "CAS 구조 예시 · 가상 회사 6곳 (실제 회사·실제 값 아님)"
    if up is not None:
        try:
            table = pd.read_csv(up, dtype=str)
            source = up.name
        except Exception as e:
            st.error(f"CSV를 읽지 못했어요: {e}")
            return
    lines = detect_lines(table.columns)
    if not lines:
        st.error("IncurLoss_로 시작하는 열이 없어요. CAS Loss Reserving Database 파일인지 확인해 주세요.")
        return
    c1, c2, c3 = st.columns(3)
    sfx = c1.selectbox("보험 종목", lines, format_func=lambda x: f"{x} · {LINES.get(x, '종목 코드')}", key="res_sfx")
    data = ingest(table, sfx)
    if not data["ok"]:
        if data["missing"]:
            st.error("필수 열이 없어 업로드를 막았어요: " + ", ".join(data["missing"]))
        elif data["dup"]:
            st.error(f"같은 (GRCODE, 사고연도, 진전연수) 행이 {data['dup']}개 중복돼요. 예: " + " · ".join(data["dupKeys"]))
        else:
            st.error("읽을 수 있는 행이 없어요.")
        return
    st.info(f"{source} · {data['nRows']:,}행 · 회사 {data['nCompanies']}곳 · 사고연도 {int(data['ayRange'][0])}–{int(data['ayRange'][1])} · 숫자가 아닌 행 {data['bad']}개 · 진전연수 불일치 {data['lagMismatch']}개")

    basis = c2.selectbox("손해액 기준", list(BASES), format_func=BASES.get, key="res_basis")
    years = list(range(int(data["ayRange"][0]) + 2, int(data["dyRange"][1]) + 1)) or [int(data["dyRange"][1])]
    eval_year = c3.selectbox("평가연도", years, index=years.index(1997) if 1997 in years else len(years) - 1, key="res_eval")
    co_list, co_default = companies(data, basis, eval_year)
    co_names = {c["g"]: f"{c['name'] or int(c['g'])} ({int(c['g'])})" + ("" if c["clean"] else " · 자료 주의") for c in co_list}
    options = [c["g"] for c in co_list] + ["ALL"]
    grcode = st.selectbox("회사", options, index=options.index(co_default), format_func=lambda g: "전체 합산" if g == "ALL" else co_names[g], key="res_co")

    T = build_triangle(data, grcode, basis, eval_year)
    if T["missing"] or T["n"] < 3:
        st.error("상삼각형에 빈 칸이 있거나 사고연도가 3개 미만이라 계산할 수 없어요: " + ", ".join(T["missing"][:8]))
        return
    if T["nonpos"]:
        st.warning(f"0 이하인 칸이 {T['nonpos']}개 있어요. 그 칸에서 시작하는 진전계수는 계산에서 빠져요.")
    if T["paidGtRep"]:
        st.warning(f"지급액이 보고발생액보다 큰 칸이 {T['paidGtRep']}개 있어요.")
    if any(not (p and p > 0) for p in T["prem"]):
        st.warning("보험료가 0이거나 없는 사고연도가 있어요. 그 해는 BF·Cape Cod에서 빠져요.")
    n = T["n"]
    ay_labels = [str(int(a)) for a in T["ays"]]
    lag_labels = [str(j + 1) for j in range(n)]

    st.markdown("#### 진전삼각형")
    st.dataframe(pd.DataFrame(T["upper"], index=ay_labels, columns=lag_labels), width="stretch")

    st.markdown("#### 진전계수와 LDF 선택")
    F = ata(T)
    st.dataframe(pd.DataFrame(F, index=ay_labels, columns=[f"{j + 1}-{j + 2}" for j in range(n - 1)]).round(4), width="stretch")
    c1, c2, c3 = st.columns(3)
    method = c1.selectbox("LDF 방법", list(METHODS), format_func=METHODS.get, key="res_method")
    recent_n = c2.slider("최근 N년", 2, n - 1, min(3, n - 1), key="res_n") if n > 3 else 2
    cells = {f"{ay_labels[i]} · {j + 1}-{j + 2}": (i, j) for i in range(n) for j in range(n - 1) if F[i][j] is not None}
    picked = c3.multiselect("제외할 진전계수", list(cells), key="res_ex")
    excluded = {cells[x] for x in picked}
    matrix = {k: [ldf_value(T, F, j, k, recent_n, excluded) for j in range(n - 1)] for k in METHODS}
    st.dataframe(pd.DataFrame(matrix, index=[f"{j + 1}-{j + 2}" for j in range(n - 1)]).rename(columns=METHODS).T.round(4), width="stretch")
    sel = [x if is_num(x) else 1.0 for x in matrix[method]]

    st.markdown("#### 꼬리계수")
    c1, c2, c3 = st.columns(3)
    tail_mode = c1.selectbox("꼬리", ["none", "manual", "exp", "invpow"], format_func={"none": "없음 (1.0)", "manual": "직접 입력", "exp": "지수 감쇠 적합", "invpow": "역멱 적합"}.get, key="res_tail")
    tail = 1.0
    if tail_mode == "manual":
        tail = c2.number_input("꼬리계수", 1.0, 2.0, 1.0, 0.001, format="%.3f", key="res_tail_v")
    elif tail_mode in ("exp", "invpow"):
        from_j = c2.slider("적합 시작 구간", 1, n - 2, min(3, n - 2), key="res_tail_from") if n > 3 else 1
        fit = fit_tail(sel, from_j, tail_mode)
        tail = fit["tail"]
        if fit["ok"]:
            c3.metric("적합 꼬리계수", f"{tail:.4f}", f"R² {fit['r2']:.3f}" + (" · 100구간 상한" if fit["capped"] else ""), delta_color="off")
        else:
            st.warning(fit["msg"] + " 꼬리 1.0으로 계산해요.")
    cdf = cdfs(sel, tail)

    st.markdown("#### 기대손해율 (BF · Cape Cod)")
    c1, c2, c3 = st.columns(3)
    elr_mode = c1.selectbox("BF 기대손해율", ["manual", "mature", "cc"], format_func={"manual": "직접 입력", "mature": "성숙 사고연도 평균 (진전율 90% 이상)", "cc": "Cape Cod 값 사용"}.get, key="res_elr_mode")
    elr = c2.number_input("기대손해율", 0.0, 3.0, 0.70, 0.01, key="res_elr", disabled=elr_mode != "manual")
    decay = c3.slider("Cape Cod 감쇠 D (1 = 전체 단일)", 0.0, 1.0, 1.0, 0.05, key="res_decay")
    R = reserve(T, cdf, elr, elr_mode, decay)
    R0 = reserve(T, cdfs(sel, 1.0), elr, elr_mode, decay)
    tt = R["totals"]

    st.markdown("#### 결과 비교")
    a, b, c = st.columns(3)
    for col, key, name in ((a, "CL", "Chain Ladder"), (b, "BF", "Bornhuetter-Ferguson"), (c, "CC", "Cape Cod")):
        col.metric(f"{name} IBNR", money(tt[key]["ibnr"]) if tt[key] else "-")
        if tt[key]:
            col.caption(f"최종 {money(tt[key]['ult'])} · 미지급 {money(tt[key]['unpaid'])} · 빠진 사고연도 {tt[key]['skipped']}")
    st.caption(f"Cape Cod ELR {R['elrCC']:.4f}" if is_num(R["elrCC"]) else "보험료가 없어 Cape Cod ELR을 계산하지 못했어요.")
    if is_num(T["posted"]):
        st.caption(f"회사 공시 준비금(PostedReserve97): {money(T['posted'])}")
    rows = []
    for r in R["rows"]:
        row = {"사고연도": int(r["ay"]), "최신 누적": r["C"], "보험료": r["P"], "CDF": round(r["c"], 4), "진전율": f"{r['pct']:.1%}"}
        for key in ("CL", "BF", "CC"):
            row[f"{key} 최종"] = round(r[key]["ult"]) if r[key] else None
            row[f"{key} IBNR"] = round(r[key]["ibnr"]) if r[key] else None
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    fig = go.Figure([go.Bar(x=ay_labels, y=[r[k]["ibnr"] if r[k] else None for r in R["rows"]], name=k) for k in ("CL", "BF", "CC")])
    fig.update_layout(barmode="group", height=300, yaxis_title="IBNR", xaxis_title="사고연도", margin=dict(t=20))
    st.plotly_chart(fig)

    st.markdown("#### 사후검증")
    no_tail = st.checkbox("꼬리를 빼고 평가연도 삼각형 끝 진전연수와 비교", True, key="res_bt_tail")
    bt = backtest(T, R0 if no_tail else R, no_tail)
    if bt["nAY"] == 0:
        st.info("파일에 평가연도 뒤의 실제 값(하삼각)이 없어 사후검증을 할 수 없어요.")
    else:
        st.caption(f"진전연수 {bt['target']}의 실제 값과 비교 · 비교한 사고연도 {bt['nAY']}개, 값이 없는 사고연도 {bt['missing']}개")
        summary = []
        for key in ("CL", "BF", "CC"):
            x = bt[key]
            if x:
                summary.append({"방법": key, "예측 최종": money(x["ult"]), "실제 최종": money(x["actual"]), "최종 오차": pct(x["err"]), "예측 IBNR": money(x["ibnr"]), "실제 IBNR": money(x["actualIBNR"]), "IBNR 오차": pct(x["ibnrErr"]), "미지급 오차": pct(x["unpaidErr"])})
        st.dataframe(pd.DataFrame(summary), width="stretch", hide_index=True)
        detail = []
        for r in bt["rows"]:
            if r["actual"] is not None:
                row = {"사고연도": int(r["ay"]), "실제 최종": money(r["actual"]), "실제 IBNR": money(r["actualIBNR"])}
                row.update({f"{k} 오차": pct(r[k]["err"]) if r[k] else "-" for k in ("CL", "BF", "CC")})
                detail.append(row)
        st.dataframe(pd.DataFrame(detail), width="stretch", hide_index=True)
        st.caption("실제 IBNR이나 미지급액이 최종의 1% 미만이면 오차율이 의미 없어 '-'로 표시해요.")

    st.markdown("#### 평가연도를 바꿔 가며 반복 사후검증")
    all_years = list(range(int(data["ayRange"][0]) + 3, int(data["dyRange"][1]) + 1))
    mb = multi_backtest(data, grcode, basis, method, recent_n, elr, elr_mode, decay, all_years)
    if mb:
        st.dataframe(pd.DataFrame([{"평가연도": x["year"], "사고연도 수": x["n"], **{f"{k} IBNR 오차": pct(x[k]["ibnrErr"]) if x[k] else "-" for k in ("CL", "BF", "CC")}, **{f"{k} 최종 오차": pct(x[k]["err"]) if x[k] else "-" for k in ("CL", "BF", "CC")}} for x in mb]), width="stretch", hide_index=True)
        st.caption("꼬리 1.0, 제외 없음으로 해마다 다시 계산해요. 판단을 나중에 바꾼 효과는 들어 있지 않아요.")

    st.markdown("#### Mack 표준오차")
    st.caption("Mack(1993) · 가중평균 전체 LDF · 제외 없음 · 꼬리 1.0 기준이라, 위에서 고른 LDF와 꼬리는 반영되지 않아요.")
    M = mack(T)
    a, b, c = st.columns(3)
    a.metric("IBNR", money(M["total"]["ibnr"]))
    b.metric("표준오차", money(M["total"]["se"]))
    c.metric("변동계수(CV)", f"{M['total']['cv']:.1%}" if M["total"]["cv"] is not None else "-")
    st.dataframe(pd.DataFrame([{"사고연도": int(r["ay"]), "최종": money(r["ult"]), "IBNR": money(r["ibnr"]), "표준오차": money(r["se"]), "CV": f"{r['cv']:.1%}" if r["cv"] is not None else "-"} for r in M["rows"]]), width="stretch", hide_index=True)

    st.markdown("#### 검산과 테스트")
    checks = check_ex5()
    passed = sum(1 for x in checks if x["결과"] == "일치")
    raa_ibnr, raa_se = raa_benchmark()
    a, b, c = st.columns(3)
    a.metric("5×5 예제", f"{passed} / {len(checks)} 일치")
    b.metric("RAA IBNR (기대 52,135)", f"{raa_ibnr:,.0f}")
    c.metric("RAA Mack 표준오차 (기대 26,909)", f"{raa_se:,.0f}")
    with st.expander("5×5 예제 항목별 결과"):
        st.dataframe(pd.DataFrame(checks), width="stretch", hide_index=True)


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

t_exp, t_freq, t_sev, t_pp, t_val, t_excel, t_res = st.tabs(["경험통계", "빈도 GLM", "심도 GLM", "순보험료", "검증 · A/E · 신뢰도", "엑셀 재현판", "지급준비금(IBNR)"])

with t_exp:
    a, b, c, d = st.columns(4)
    a.metric("계약 수", f"{len(df):,}")
    b.metric("관측 빈도 (금액 기록 기준)", f"{df['nAmt'].sum() / df['Exposure'].sum():.4f} 건/년")
    c.metric("관측 심도", f"€{df['S'].sum() / max(df['nAmt'].sum(), 1):,.0f}")
    d.metric("고액 청구(€50,000 이상)", f"{(sev['ClaimAmount'] >= 50000).sum():,}건")
    orphan = (~sev["IDpol"].isin(freq["IDpol"])).sum()
    no_amt = ((df["ClaimNb"] > 0) & (df["nAmt"] == 0)).sum()
    st.caption(f"자료 점검: 빈도 표에 없는 청구 {orphan:,}건 · 청구건수는 있지만 금액이 없는 계약 {no_amt:,}건")
    key = st.selectbox("집단", list(LABEL), format_func=LABEL.get)
    g = df.groupby(key).agg(노출=("Exposure", "sum"), 금액건수=("nAmt", "sum"), 금액=("S", "sum"))
    g["빈도"] = g["금액건수"] / g["노출"]
    g["심도"] = g["금액"] / g["금액건수"].replace(0, np.nan)
    st.dataframe(g.round(4), width="stretch")
    st.bar_chart(g["빈도"])

with t_freq:
    st.subheader("포아송 빈도 GLM 상대도")
    st.latex(r"\log \lambda_i = \log t_i + \beta_0 + \sum_k \beta_k x_{ik}")
    st.metric("과산포 φ (피어슨)", f"{M['phi']:.3f}")
    tbl = rel_table(M["f"], max(M["phi"], 1))
    st.plotly_chart(rel_chart(tbl))
    st.dataframe(tbl, width="stretch", hide_index=True)
    st.caption("구간은 √φ배 넓힌 준포아송 95% 구간이에요. 구간이 1을 지나면 차이가 확실하지 않아요.")

with t_sev:
    st.subheader("감마 심도 GLM 상대도")
    st.caption(f"금액이 있는 학습 계약 {M['nSev']:,}건 · 계약별 평균 금액을 건수로 가중")
    tbl = rel_table(M["s"], M["phi_s"] / float(M["s"].scale))
    st.plotly_chart(rel_chart(tbl))
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
    pp_total = (M["f"].predict(train, offset=np.log(train["Exposure"])) * M["s"].predict(train)).sum()
    st.metric("학습자료 평균 순보험료", f"€{pp_total / train['Exposure'].sum():,.1f}/계약년")
    st.caption("빈도와 심도가 서로 독립이라는 가정 아래 곱해요. 사업비·이익은 들어 있지 않아요.")

with t_val:
    st.subheader("검증자료 A/E")
    v = valid.copy()
    v["lam"] = M["f"].predict(v, offset=np.log(v["Exposure"]))
    v["pure"] = v["lam"] * M["s"].predict(v)
    a, b, c = st.columns(3)
    a.metric("검증 계약", f"{len(v):,}건 (25%)")
    b.metric("A/E 건수", f"{v['nAmt'].sum() / v['lam'].sum():.3f}")
    c.metric("A/E 금액", f"{v['S'].sum() / v['pure'].sum():.3f}")
    group_label = dict(LABEL)
    if "region" in v.columns:
        group_label["region"] = "지역(Region)"
    key2 = st.selectbox("A/E 집단", list(group_label), format_func=group_label.get, key="ae_key")
    vg_rows, vg_bs = val_groups(v, key2)
    a, b, c = st.columns(3)
    a.metric("집단 안 변동 EPV", f"{vg_bs['EPV']:.3f}")
    b.metric("집단 간 변동 VHM", f"{vg_bs['VHM']:.2e}" if vg_bs["VHM"] > 0 else "0 이하")
    c.metric("k = EPV ÷ VHM", f"{vg_bs['k']:,.0f}" if math.isfinite(vg_bs["k"]) else "∞")
    if vg_bs["VHM"] <= 0:
        st.caption("VHM이 0 이하면 모형이 집단 차이를 이미 설명한다는 뜻이라 Z = 0이에요.")
    st.latex(r"Z=\frac{\text{예상 건수}}{\text{예상 건수}+k},\quad \text{가중 A/E}=Z\times A/E+(1-Z)")
    st.dataframe(
        pd.DataFrame([{group_label[key2]: o["key"], "계약": o["policies"], "노출": round(o["expo"], 1), "실제 건수": o["claims"], "예상 건수": round(o["E"], 1), "A/E 건수": round(o["aeN"], 3), "A/E 금액": round(o["aeS"], 3), "Z": round(o["Z"], 3), "가중 A/E 건수": round(o["credAE"], 3), "우연일 확률": round(o["pUp"] if o["credAE"] >= 1 else o["pDown"], 4), "판정": o["flag"]} for o in vg_rows]),
        width="stretch", hide_index=True,
    )
    st.plotly_chart(go.Figure(go.Bar(x=[o["key"] for o in vg_rows], y=[o["aeN"] for o in vg_rows], name="A/E 건수")).add_scatter(x=[o["key"] for o in vg_rows], y=[o["credAE"] for o in vg_rows], mode="markers", name="가중 A/E 건수").add_hline(y=1, line_dash="dash").update_layout(height=320, margin=dict(t=20)))
    st.caption("판정: 가중 A/E 건수 ≥ 1.05이고 포아송 단측 p < 0.05면 높음, ≤ 0.95이고 반대쪽 p < 0.05면 낮음, Z < 0.3이면 자료 부족이에요. A/E가 1보다 크다는 것만으로 구조적 악화나 보험료 인상 필요를 확정하지 않아요.")
    st.subheader("신뢰도 검산 예제: 집단 3개")
    ex_bs, ex_rows = cred_example()
    st.dataframe(pd.DataFrame(ex_rows), width="stretch", hide_index=True)
    ex_ok = abs(ex_bs["EPV"] - 0.359) < 0.0005 + 1e-9 and abs(ex_bs["VHM"] - 0.335) < 0.0005 + 1e-9 and abs(ex_bs["k"] - 1.07) < 0.005 + 1e-9
    st.caption(("검산값과 일치" if ex_ok else "검산값과 다름") + f": EPV {ex_bs['EPV']:.3f} · VHM {ex_bs['VHM']:.3f} · k {ex_bs['k']:.2f} (기대값 0.359 · 0.335 · 1.07)")

with t_excel:
    st.subheader("엑셀 재현판")
    st.caption("아래 CSV를 엑셀에 붙이고 같은 표를 수식으로 다시 만들어 앱 값과 대조해요. 2행 = 검증자료 첫 집단이에요.")
    xl_key = "region" if "region" in v.columns else "age"
    xl_col = "Region" if xl_key == "region" else "Group"
    xl = df.copy()
    xl["lambda"] = M["f"].predict(xl, offset=np.zeros(len(xl)))
    xl["mu"] = M["s"].predict(xl)
    xl["PredCount"] = xl["Exposure"] * xl["lambda"]
    xl["PredLoss"] = xl["PredCount"] * xl["mu"]
    xl["Split"] = np.where(xl["valid"], "검증", "학습")
    xl[xl_col] = xl[xl_key]
    xl = xl.rename(columns={"nAmt": "N", "S": "ClaimTotal"})[["IDpol", xl_col, "Split", "Exposure", "N", "lambda", "mu", "PredCount", "PredLoss", "ClaimTotal"]]
    st.download_button("계약별 예측 CSV 내려받기", xl.to_csv(index=False).encode("utf-8-sig"), "policy_predictions.csv", "text/csv")
    xl_rows, xl_bs = val_groups(v, xl_key)
    r0 = xl_rows[0]
    st.dataframe(
        pd.DataFrame(
            [
                ["A 집단", "(집단 이름)", str(r0["key"])],
                ["B 노출", f'=SUMIFS(Exposure, {xl_col}, A2, Split, "검증")', f"{r0['expo']:,.2f}"],
                ["C 실제 건수", f'=SUMIFS(N, {xl_col}, A2, Split, "검증")', str(r0["claims"])],
                ["D 예상 건수", f'=SUMIFS(PredCount, {xl_col}, A2, Split, "검증")', f"{r0['E']:,.2f}"],
                ["E A/E 건수", "=C2 / D2", f"{r0['aeN']:.4f}"],
                ["F 실제 금액", f'=SUMIFS(ClaimTotal, {xl_col}, A2, Split, "검증")', f"{r0['amount']:,.2f}"],
                ["G 예상 금액", f'=SUMIFS(PredLoss, {xl_col}, A2, Split, "검증")', f"{r0['ES']:,.2f}"],
                ["H A/E 금액", "=F2 / G2", f"{r0['aeS']:.4f}"],
                ["I Z", "=D2 / (D2 + $k)", f"{r0['Z']:.4f}"],
                ["J 가중 A/E", "=I2 * E2 + (1 - I2) * 1", f"{r0['credAE']:.4f}"],
            ],
            columns=["열", "엑셀 수식", "앱 값"],
        ),
        width="stretch", hide_index=True,
    )
    st.caption(f"PredCount = Exposure × lambda, PredLoss = PredCount × mu. k = {xl_bs['k']:,.1f}" if math.isfinite(xl_bs["k"]) else "VHM이 0 이하라 k = ∞, Z = 0이에요.")
    st.markdown("**처음 10개 집단을 CSV 값으로 다시 계산해 앱 값과 대조**")
    xl_val = xl[xl["Split"] == "검증"]
    compare = []
    for o in xl_rows[:10]:
        part = xl_val[xl_val[xl_col] == o["key"]]
        same = abs(part["N"].sum() / part["PredCount"].sum() - o["aeN"]) < 1e-9 and abs(part["ClaimTotal"].sum() / part["PredLoss"].sum() - o["aeS"]) < 1e-9
        compare.append({"집단": o["key"], "CSV A/E 건수": round(part["N"].sum() / part["PredCount"].sum(), 4), "앱 A/E 건수": round(o["aeN"], 4), "CSV A/E 금액": round(part["ClaimTotal"].sum() / part["PredLoss"].sum(), 4), "앱 A/E 금액": round(o["aeS"], 4), "결과": "일치" if same else "다름"})
    st.dataframe(pd.DataFrame(compare), width="stretch", hide_index=True)

with t_res:
    reserve_tab()
