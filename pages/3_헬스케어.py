"""헬스케어 (담당: 식품영양학과)
집단 영양관리과정(NCP): ① 영양판정 → ② 영양진단 → ③ 영양중재 → ④ 평가계획.
지역사회영양학의 집단 영양판정 개념으로, 개인용 NCP를 집단 데이터에 맞게 바꿨어요.

- 예시 데이터는 국민건강영양조사(국건영) 변수 이름을 흉내 낸 시뮬레이션이에요. 숫자는 실제 통계가 아니에요.
- 진짜 국건영 CSV를 올리면 같은 화면이 그 데이터로 다시 계산돼요. (notebooks/국건영_CSV_만들기.ipynb 참고)
- 식사 변수가 없는 건강검진 CSV를 올려도 돌아가요. 이때는 섭취·행동 영역 진단은 빠지고 임상 영역만 남아요.
- 변수 이름·코드값·기준값은 국건영 원시자료 이용지침서와 학회·정부 원문으로 꼭 확인해야 해요.
"""
import io
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="헬스케어", layout="wide")

# ================================================================ 기준값 (원문 확인 후 출처·연도 확정)
SODIUM_CDRR = 2300          # 나트륨 만성질환위험감소섭취량 (mg/일, 성인) · 2025 한국인 영양소 섭취기준
CHO_RANGE = (50, 65)        # 탄수화물 에너지적정비율 (%) · 2025 KDRIs (2020: 55~65)
FAT_RANGE = (15, 30)        # 지방 에너지적정비율 (%)
SUGAR_MAX = 20              # 총당류 에너지 비율 상한 (%) · 2025 KDRIs (첨가당은 10% 이내 제한)
FIBER_AI = {"남": 30, "여": 20, "남65": 25}  # 식이섬유 충분섭취량 (g/일) · 2020 KDRIs 기준, 2025 개정값 확인 필요
ENERGY_OK = (500, 5000)     # 하루 섭취 열량이 이 범위를 벗어나면 식사 자료를 믿기 어려워 뺌 (과소·과대 보고)

COLORS = {"정상": "#4C9F70", "위험군": "#E5A93C", "대사증후군": "#C8553D"}
GROUPS = list(COLORS)

# ================================================================ 열 맞추기
# 페이지 안에서 쓰는 이름 → (화면 이름, 꼭 필요한지)
FIELDS = {
    "sex": ("성별 (1=남, 2=여)", True), "age": ("나이", True),
    "height": ("키 (BMI 계산용)", False), "weight": ("몸무게 (BMI 계산용)", False), "bmi": ("BMI", False),
    "waist": ("허리둘레", True), "sbp": ("수축기 혈압", True), "dbp": ("이완기 혈압", True),
    "glu": ("공복혈당", True), "tg": ("중성지방", True), "hdl": ("HDL 콜레스테롤", True),
    "hba1c": ("당화혈색소", False), "tc": ("총콜레스테롤", False), "fst": ("공복 시간 (8시간 미만이면 혈당 제외)", False),
    "dm_dx": ("당뇨 의사진단", False), "dm_med": ("혈당강하제 복용", False), "dm_ins": ("인슐린 주사", False),
    "ht_dx": ("고혈압 의사진단", False), "ht_med": ("혈압조절제 복용", False), "dl_dx": ("이상지질혈증 의사진단", False),
    "smoke": ("흡연 상태", False), "drink": ("음주 (빈도 또는 Y/N)", False),
    "breakfast": ("아침식사 빈도", False), "eatout": ("외식 빈도", False),
    "energy": ("에너지 섭취량 (kcal)", False), "cho": ("탄수화물 (g)", False), "fat": ("지방 (g)", False),
    "sugar": ("당류 (g)", False), "fiber": ("식이섬유 (g)", False), "sodium": ("나트륨 (mg)", False),
    "wt": ("가중치 (없으면 표본 기준)", False),
}
# 별명 사전: 대소문자·띄어쓰기·밑줄·괄호를 지운 뒤 비교해요. 국건영 이름은 이용지침서로 다시 확인할 것.
ALIASES = {
    "sex": ["sex", "gender", "성별", "성별코드"],
    "age": ["age", "나이", "연령", "만나이"],
    "height": ["heht", "height", "ht", "키", "신장", "신장5cm단위"],
    "weight": ["hewt", "weight", "wt", "몸무게", "체중", "체중5kg단위"],
    "bmi": ["hebmi", "bmi", "체질량지수"],
    "waist": ["hewc", "waist", "waistline", "허리둘레", "허리"],
    "sbp": ["hesbp", "sbp", "systolic", "수축기혈압", "최고혈압"],
    "dbp": ["hedbp", "dbp", "diastolic", "이완기혈압", "최저혈압"],
    "glu": ["heglu", "glu", "fbg", "blds", "glucose", "공복혈당", "식전혈당", "식전혈당공복혈당"],
    "tg": ["hetg", "tg", "triglyceride", "중성지방", "트리글리세라이드"],
    "hdl": ["hehdlst2", "hehdl", "hdl", "hdlchole", "hdl콜레스테롤"],
    "hba1c": ["hehba1c", "hba1c", "당화혈색소"],
    "tc": ["hechol", "tc", "totchole", "총콜레스테롤"],
    "fst": ["hefst", "fst", "공복시간"],
    "dm_dx": ["de1dg", "당뇨진단", "당뇨병의사진단"],
    "dm_med": ["de131", "혈당강하제", "혈당강하제복용"],
    "dm_ins": ["de132", "인슐린", "인슐린주사"],
    "ht_dx": ["di1dg", "고혈압진단", "고혈압의사진단"],
    "ht_med": ["di12", "혈압조절제", "혈압조절제복용"],
    "dl_dx": ["di2dg", "이상지질혈증진단"],
    "smoke": ["bs31", "smkstattypecd", "smoke", "흡연", "흡연상태"],
    "drink": ["bd111", "drkyn", "drink", "음주", "음주여부", "음주빈도"],
    "breakfast": ["lbrfq", "아침식사빈도", "아침식사"],
    "eatout": ["loutfq", "외식빈도", "외식"],
    "energy": ["nen", "energy", "에너지", "열량"],
    "cho": ["ncho", "탄수화물"],
    "fat": ["nfat", "지방"],
    "sugar": ["nsugar", "당류", "총당류"],
    "fiber": ["ntdf", "식이섬유"],
    "sodium": ["nna", "sodium", "나트륨"],
    "wt": ["wttot", "wtntr", "wtitvex", "weight_s", "가중치"],
}
NONE = "(없음)"
VALID = {"age": (1, 120), "bmi": (10, 70), "height": (100, 220), "weight": (20, 250), "waist": (40, 200),
         "sbp": (60, 260), "dbp": (30, 180), "glu": (40, 600), "tg": (10, 5000), "hdl": (5, 200),
         "hba1c": (3, 20), "tc": (50, 600), "fst": (0, 72)}


def norm(name):
    return re.sub(r"[\s_\-\(\)\[\]\.·/]", "", str(name)).lower()


@st.cache_data
def read_any_csv(raw: bytes):
    """한글 엑셀 CSV(cp949)와 UTF-8 모두 읽어요."""
    for enc in ("utf-8-sig", "cp949"):
        try:
            t = pd.read_csv(io.BytesIO(raw), encoding=enc, low_memory=False)
            if t.shape[1] == 1:
                t = pd.read_csv(io.BytesIO(raw), encoding=enc, sep=None, engine="python")
            return t
        except UnicodeDecodeError:
            continue
    raise ValueError("글자 인코딩을 알 수 없어요. UTF-8이나 CP949로 저장해 주세요.")


def guess_columns(columns):
    by_norm = {norm(c): c for c in columns}
    return {k: next((by_norm[a] for a in ALIASES[k] if a in by_norm), None) for k in FIELDS}


def missing_fields(mapping):
    miss = [k for k, (_, need) in FIELDS.items() if need and not mapping.get(k)]
    if not mapping.get("bmi") and not (mapping.get("height") and mapping.get("weight")):
        miss.append("bmi")
    return miss


# ---------------------------------------------------------------- 코드값 바꾸기
# 국건영 설문 코드: 8 = 비해당(해당 없음), 9 = 모름·무응답. 이용지침서로 꼭 확인할 것.
def yes_no(s):
    """1=예 → 1, 0=아니오·8=비해당 → 0, 9=모름 → 빈칸. Y/N 글자도 처리."""
    if not pd.api.types.is_numeric_dtype(s):
        t = s.astype(str).str.strip().str.upper()
        return t.map({"Y": 1, "N": 0, "YES": 1, "NO": 0, "예": 1, "아니오": 0, "1": 1, "0": 0}).astype(float)
    return s.map({1: 1, 0: 0, 8: 0}).astype(float)


def smoke_now(s, colname):
    """현재 흡연 여부. 공단 검진(SMK_STAT_TYPE_CD)은 3=현재 흡연, 국건영(BS3_1)은 1=매일·2=가끔 피움."""
    if norm(colname) == "smkstattypecd":
        return s.map({1: 0, 2: 0, 3: 1}).astype(float)
    return s.map({1: 1, 2: 1, 3: 0, 8: 0}).astype(float)


def drink_flag(s):
    """국건영 BD1_11(음주 빈도 1~6)이면 주 2회 이상(5·6)을 '잦은 음주'로, Y/N이면 음주 여부로."""
    if not pd.api.types.is_numeric_dtype(s):
        return yes_no(s), "음주함"
    if s.dropna().isin([0, 1]).all():
        return s.astype(float), "음주함"
    return s.map({1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 1, 8: 0}).astype(float), "주 2회 이상 음주"


def build_data(raw, mapping):
    """고른 열을 페이지 형식으로 바꿔요. 바꾼 내용은 notes에 적어요."""
    num = lambda k: pd.to_numeric(raw[mapping[k]], errors="coerce")
    out, notes, meta = pd.DataFrame(index=raw.index), [], {}

    s = raw[mapping["sex"]]
    if pd.api.types.is_numeric_dtype(s):
        out["sex"] = s
    else:
        t = s.astype(str).str.strip().str.lower()
        out["sex"] = t.map({"male": 1, "m": 1, "남": 1, "남성": 1, "1": 1, "female": 2, "f": 2, "여": 2, "여성": 2, "2": 2})
    out["age"] = num("age")

    for k in ("height", "weight", "bmi", "waist", "sbp", "dbp", "glu", "tg", "hdl", "hba1c", "tc", "fst",
              "energy", "cho", "fat", "sugar", "fiber", "sodium", "wt"):
        out[k] = num(k) if mapping.get(k) else np.nan
    if not mapping.get("bmi"):
        out["bmi"] = (out["weight"] / (out["height"] / 100) ** 2).round(1)
        notes.append("BMI 열이 없어 몸무게 ÷ 키(m)²로 계산했어요.")

    for k in ("dm_dx", "dm_med", "dm_ins", "ht_dx", "ht_med", "dl_dx"):
        out[k] = yes_no(raw[mapping[k]]) if mapping.get(k) else np.nan
    out["smoke"] = smoke_now(num("smoke"), mapping["smoke"]) if mapping.get("smoke") else np.nan
    meta["drink_label"] = "음주"
    if mapping.get("drink"):
        col = raw[mapping["drink"]]
        col = pd.to_numeric(col, errors="coerce") if pd.api.types.is_numeric_dtype(col) else col
        out["drink"], meta["drink_label"] = drink_flag(col)
    else:
        out["drink"] = np.nan
    # 아침식사 빈도(L_BR_FQ): 1=주5~7회, 2=주3~4회, 3=주1~2회, 4=거의 안 함 → 주 2회 이하를 결식으로
    out["skip_bf"] = num("breakfast").map({1: 0, 2: 0, 3: 1, 4: 1}).astype(float) if mapping.get("breakfast") else np.nan
    # 외식 빈도(L_OUT_FQ): 1=하루 2회 이상, 2=하루 1회 → 하루 1회 이상을 잦은 외식으로
    out["eatout_d"] = num("eatout").map({1: 1, 2: 1, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}).astype(float) if mapping.get("eatout") else np.nan

    bad = 0
    for k, (lo, hi) in VALID.items():
        m = out[k].notna() & ~out[k].between(lo, hi)
        bad += int(m.sum())
        out.loc[m, k] = np.nan
    if bad:
        notes.append(f"범위를 벗어난 값 {bad:,}개(예: 허리둘레 999)를 빈칸으로 바꿨어요.")

    if mapping.get("fst"):
        short = out["fst"].notna() & (out["fst"] < 8)
        out.loc[short, "glu"] = np.nan
        if short.any():
            notes.append(f"공복 8시간 미만인 {int(short.sum()):,}명은 공복혈당을 빈칸으로 바꿨어요.")

    if mapping.get("energy"):
        odd = out["energy"].notna() & ~out["energy"].between(*ENERGY_OK)
        out.loc[odd, ["energy", "cho", "fat", "sugar", "fiber", "sodium"]] = np.nan
        if odd.any():
            notes.append(f"하루 섭취 열량이 {ENERGY_OK[0]:,}kcal 미만이거나 {ENERGY_OK[1]:,}kcal 초과인 {int(odd.sum()):,}명은 식사 자료를 뺐어요 (과소·과대 보고 의심).")

    before = len(out)
    out = out[out["age"] >= 19]
    if len(out) < before:
        notes.append(f"19세 미만 {before - len(out):,}명은 성인 기준을 쓰기 위해 뺐어요.")
    before = len(out)
    out = out.dropna(subset=["sex", "age", "bmi", "waist", "sbp", "dbp", "glu", "tg", "hdl"])
    if len(out) < before:
        notes.append(f"대사증후군 판정에 필요한 값(허리·혈압·공복혈당·중성지방·HDL·BMI)이 빈 {before - len(out):,}명은 뺐어요.")

    meta["weighted"] = bool(mapping.get("wt")) and out["wt"].notna().any()
    if meta["weighted"]:
        out = out[out["wt"].notna() & (out["wt"] > 0)]
    else:
        out["wt"] = 1.0
    return out.reset_index(drop=True), notes, meta


# ---------------------------------------------------------------- 예시 데이터 (국건영 이름을 흉내 낸 시뮬레이션)
@st.cache_data
def simulate(n=6000, seed=11):
    r = np.random.default_rng(seed)
    g = lambda: r.standard_normal(n)
    male = (r.random(n) < 0.49).astype(int)
    age = np.round(r.uniform(19, 80, n))
    a = (age - 50) / 15
    eatout = r.choice([1, 2, 3, 4, 5, 6], n, p=[0.1, 0.22, 0.18, 0.2, 0.18, 0.12])
    eatout = np.where(age > 65, np.maximum(eatout, 4), eatout)
    bf = r.choice([1, 2, 3, 4], n, p=[0.6, 0.1, 0.1, 0.2])
    bf = np.where(age < 35, r.choice([1, 2, 3, 4], n, p=[0.3, 0.12, 0.18, 0.4]), bf)
    out_d = (eatout <= 2).astype(int)
    smoke = np.where(male == 1, r.choice([1, 2, 3, 8], n, p=[0.3, 0.04, 0.4, 0.26]), r.choice([1, 2, 3, 8], n, p=[0.05, 0.02, 0.06, 0.87]))
    drink = np.where(male == 1, r.choice([1, 2, 3, 4, 5, 6], n, p=[0.15, 0.1, 0.1, 0.25, 0.25, 0.15]), r.choice([1, 2, 3, 4, 5, 6], n, p=[0.3, 0.2, 0.15, 0.2, 0.1, 0.05]))
    freq = (drink >= 5).astype(int)

    energy = np.clip(np.where(male == 1, 2250, 1650) - 120 * a + 180 * out_d + np.where(male == 1, 650, 480) * g(), 400, 5500)
    cho_p = np.clip(61 + 4 * a - 2 * out_d + 8 * g(), 30, 85)
    fat_p = np.clip(22 - 3 * a + 3 * out_d + 6.5 * g(), 5, 50)
    sug_p = np.clip(13 - 1 * a + 1.5 * (bf >= 3) + 6 * g(), 2, 45)
    na_dens = np.clip(1.6 + 0.25 * out_d + 0.1 * a + 0.35 * g(), 0.5, 4)        # mg/kcal
    fib_dens = np.clip(13 + 2 * a - 1.5 * out_d + 3.5 * g(), 3, 35)             # g/1000kcal

    bmi = np.clip(23.4 + 1.0 * male + 0.4 * a - 0.35 * a ** 2 + 0.06 * (sug_p - 13) + 0.05 * (fat_p - 22) + 0.4 * out_d + 3.1 * g(), 15, 45)
    waist = np.clip(np.where(male == 1, 2.5 * bmi + 25, 2.35 * bmi + 22) + 1.5 * a + 4 * g(), 55, 140)
    sbp = np.clip(116 + 6 * a + 1.2 * (bmi - 23.5) + 5 * male + 3.5 * (na_dens - 1.6) + 2 * freq + 12 * g(), 85, 220)
    dbp = np.clip(75 + 1.5 * a + 0.7 * (bmi - 23.5) + 4 * male + 0.15 * (sbp - 120) + 8 * g(), 45, 130)
    glu = np.clip(89 + 4 * a + 1.3 * (bmi - 23.5) + 0.15 * (cho_p - 61) + 2 * male + 9 * g() + r.gamma(1.0, 4, n), 65, 400)
    dm_true = glu >= 126
    glu = np.where(r.random(n) < 0.06, glu + r.gamma(2, 20, n), glu)
    hba1c = np.clip(3.2 + 0.024 * glu + 0.3 * g(), 4, 13)
    tg = np.clip(np.exp(np.log(110) + 0.05 * (bmi - 23.5) + 0.012 * (cho_p - 61) + 0.01 * (sug_p - 13) + 0.25 * male + 0.15 * freq + 0.45 * g()), 25, 2000)
    hdl = np.clip(53 - 6 * male - 0.9 * (bmi - 23.5) - 0.15 * (cho_p - 61) + 3 * freq - 2 * (smoke <= 2) + 11 * g(), 15, 120)
    tc = np.clip(190 + 5 * a + 1.2 * (bmi - 23.5) + 0.3 * (fat_p - 22) + 34 * g(), 90, 400)
    height = np.where(male == 1, 172, 159) - 2.5 * a + 6 * g()
    weight = bmi * (height / 100) ** 2

    dm_dx = ((glu >= 126) | dm_true) & (r.random(n) < 0.65)
    ht_dx = ((sbp >= 140) | (dbp >= 90)) & (r.random(n) < 0.6)
    dl_dx = ((tc >= 240) | (tg >= 200)) & (r.random(n) < 0.4)
    dm_med = dm_dx & (r.random(n) < 0.8)
    ht_med = ht_dx & (r.random(n) < 0.85)
    # 진단 후 약을 먹으면 수치가 내려가고, 식사도 조금 바꿔요 (역인과 확인용)
    glu = np.where(dm_med, np.clip(glu - r.uniform(10, 40, n), 85, None), glu)
    sbp = np.where(ht_med, np.clip(sbp - r.uniform(5, 20, n), 100, None), sbp)
    na_dens = np.where(ht_dx, na_dens * 0.92, na_dens)
    sug_p = np.where(dm_dx, sug_p * 0.85, sug_p)

    def code(x, miss=0.02):
        x = x.astype(float)
        x[r.random(n) < miss] = 9
        return x

    df = pd.DataFrame({
        "sex": np.where(male == 1, 1, 2), "age": age,
        "HE_ht": height.round(1), "HE_wt": weight.round(1), "HE_BMI": bmi.round(1), "HE_wc": waist.round(1),
        "HE_sbp": sbp.round(), "HE_dbp": dbp.round(), "HE_glu": glu.round(), "HE_HbA1c": hba1c.round(1),
        "HE_chol": tc.round(), "HE_TG": tg.round(), "HE_HDL_st2": hdl.round(), "HE_fst": np.where(r.random(n) < 0.05, 5, 12),
        "DE1_dg": code(np.where(dm_dx, 1, 0)), "DE1_31": np.where(dm_dx, np.where(dm_med, 1, 0), 8),
        "DE1_32": np.where(dm_dx, (r.random(n) < 0.1).astype(int), 8),
        "DI1_dg": code(np.where(ht_dx, 1, 0)), "DI1_2": np.where(ht_dx, np.where(ht_med, 1, 0), 8),
        "DI2_dg": code(np.where(dl_dx, 1, 0)),
        "BS3_1": code(smoke), "BD1_11": code(drink), "L_BR_FQ": code(bf), "L_OUT_FQ": code(eatout),
        "N_EN": energy.round(), "N_CHO": (energy * cho_p / 100 / 4).round(1), "N_FAT": (energy * fat_p / 100 / 9).round(1),
        "N_SUGAR": (energy * sug_p / 100 / 4).round(1), "N_TDF": (energy * fib_dens / 1000).round(1),
        "N_NA": (energy * na_dens).round(), "wt_tot": np.round(np.exp(np.log(7000) + 0.5 * g()) * np.where(age > 65, 0.8, 1.0)),
    })
    lab_miss = r.random(n) < 0.04
    df.loc[lab_miss, ["HE_TG", "HE_HDL_st2", "HE_chol", "HE_HbA1c"]] = np.nan
    diet_miss = r.random(n) < 0.12
    df.loc[diet_miss, ["N_EN", "N_CHO", "N_FAT", "N_SUGAR", "N_TDF", "N_NA"]] = np.nan
    return df


# ================================================================ 가중 통계
def wmean(x, w):
    m = x.notna() & w.notna()
    return float(np.average(x[m], weights=w[m])) if m.any() else np.nan


def wvar(x, w):
    m = x.notna() & w.notna()
    if m.sum() < 2:
        return np.nan
    mu = np.average(x[m], weights=w[m])
    return float(np.average((x[m] - mu) ** 2, weights=w[m]))


def smd(x, w, a, b):
    """두 그룹의 표준화 평균 차이(효과크기). 단위가 달라도 같은 잣대로 비교해요. 0.2 작음 · 0.5 중간 · 0.8 큼."""
    va, vb = wvar(x[a], w[a]), wvar(x[b], w[b])
    sd = np.sqrt((va + vb) / 2) if pd.notna(va) and pd.notna(vb) else np.nan
    return (wmean(x[a], w[a]) - wmean(x[b], w[b])) / sd if sd and sd > 0 else np.nan


def md(t):
    """마크다운에서 ~ 두 개 사이가 취소선이 되지 않게 막아요."""
    return str(t).replace("~", "\\~")


def n_known(x):
    return int(x.notna().sum())


def fmt(v, kind):
    if pd.isna(v):
        return "-"
    return f"{v:.1%}" if kind == "pct" else f"{v:,.1f}"


# ================================================================ 판정
def add_status(df):
    df = df.copy()
    male = df["sex"] == 1
    med_bp = df["ht_med"] == 1
    med_glu = (df["dm_med"] == 1) | (df["dm_ins"] == 1)
    df["m_waist"] = df["waist"] >= np.where(male, 90, 85)
    df["m_tg"] = df["tg"] >= 150
    df["m_hdl"] = df["hdl"] < np.where(male, 40, 50)
    df["m_bp"] = (df["sbp"] >= 130) | (df["dbp"] >= 85) | med_bp
    df["m_glu"] = (df["glu"] >= 100) | med_glu
    df["m_count"] = df[["m_waist", "m_tg", "m_hdl", "m_bp", "m_glu"]].sum(axis=1)
    df["group"] = np.select([df["m_count"] >= 3, df["m_count"] >= 1], ["대사증후군", "위험군"], "정상")
    df["target"] = df["group"] != "정상"

    # 당뇨: 국민건강통계 정의 (공복혈당 126↑ · 의사진단 · 혈당강하제·인슐린 · 당화혈색소 6.5↑ 중 하나)
    lab_dm = (df["glu"] >= 126) | (df["hba1c"] >= 6.5)
    df["dm"] = lab_dm | (df["dm_dx"] == 1) | med_glu
    df["pre_dm"] = ~df["dm"] & ((df["glu"] >= 100) | (df["hba1c"] >= 5.7))
    df["glu_stage"] = np.select([df["dm"], df["pre_dm"]], ["당뇨", "당뇨 전단계"], "정상")
    # 진료 연계 대상: 수치는 높은데 진단을 받지 않은 사람
    df["undx_dm"] = lab_dm & (df["dm_dx"] != 1)
    df["undx_ht"] = ((df["sbp"] >= 140) | (df["dbp"] >= 90)) & (df["ht_dx"] != 1)
    df["undx_dl"] = ((df["tc"] >= 240) | (df["tg"] >= 200)) & (df["dl_dx"] != 1)

    # 식사: 에너지 보정 (많이 먹는 사람이 모든 영양소를 많이 먹는 효과를 빼요)
    df["cho_p"] = df["cho"] * 4 / df["energy"] * 100
    df["fat_p"] = df["fat"] * 9 / df["energy"] * 100
    df["sug_p"] = df["sugar"] * 4 / df["energy"] * 100
    df["na_d"] = df["sodium"] / df["energy"] * 1000
    df["fib_d"] = df["fiber"] / df["energy"] * 1000
    df["obese"] = (df["bmi"] >= 25).astype(float)
    fib_ai = np.where(male, np.where(df["age"] >= 65, FIBER_AI["남65"], FIBER_AI["남"]), FIBER_AI["여"])
    df["p_na"] = (df["sodium"] > SODIUM_CDRR).where(df["sodium"].notna())
    df["p_cho"] = (df["cho_p"] > CHO_RANGE[1]).where(df["cho_p"].notna())
    df["p_fat"] = (df["fat_p"] > FAT_RANGE[1]).where(df["fat_p"].notna())
    df["p_sug"] = (df["sug_p"] > SUGAR_MAX).where(df["sug_p"].notna())
    df["p_fib"] = (df["fiber"] < fib_ai).where(df["fiber"].notna())
    for k in ("p_na", "p_cho", "p_fat", "p_sug", "p_fib"):
        df[k] = df[k].astype(float)
    return df


# 영양판정 영역별 비교 지표: (열, 이름, 영역, 종류, 판정 기준에 쓰였는지)
INDICATORS = [
    ("bmi", "BMI (kg/m²)", "신체계측", "mean", False),
    ("obese", "비만 (BMI 25 이상)", "신체계측", "pct", False),
    ("waist", "허리둘레 (cm)", "신체계측", "mean", True),
    ("glu", "공복혈당 (mg/dL)", "생화학 자료", "mean", True),
    ("hba1c", "당화혈색소 (%)", "생화학 자료", "mean", False),
    ("tg", "중성지방 (mg/dL)", "생화학 자료", "mean", True),
    ("hdl", "HDL 콜레스테롤 (mg/dL)", "생화학 자료", "mean", True),
    ("tc", "총콜레스테롤 (mg/dL)", "생화학 자료", "mean", False),
    ("sbp", "수축기 혈압 (mmHg)", "신체검사", "mean", True),
    ("energy", "에너지 섭취 (kcal)", "식사력", "mean", False),
    ("cho_p", "탄수화물 에너지 비율 (%)", "식사력", "mean", False),
    ("fat_p", "지방 에너지 비율 (%)", "식사력", "mean", False),
    ("sug_p", "당류 에너지 비율 (%)", "식사력", "mean", False),
    ("na_d", "나트륨 (mg/1,000kcal)", "식사력", "mean", False),
    ("sodium", "나트륨 (mg/일)", "식사력", "mean", False),
    ("fib_d", "식이섬유 (g/1,000kcal)", "식사력", "mean", False),
    ("skip_bf", "아침 결식 (주 2회 이하)", "생활습관", "pct", False),
    ("eatout_d", "외식 하루 1회 이상", "생활습관", "pct", False),
    ("drink", "음주", "생활습관", "pct", False),
    ("smoke", "현재 흡연", "생활습관", "pct", False),
    ("dm_dx", "당뇨 의사진단", "과거력", "pct", False),
    ("ht_dx", "고혈압 의사진단", "과거력", "pct", False),
]

# ================================================================ 진단 후보와 중재 내용
# 원인(E) 후보로 볼 행동 변수
E_BEHAV = {"eatout_d": "외식 하루 1회 이상", "skip_bf": "아침 결식", "drink": "음주"}

DX = {
    "obese": {
        "P": "과체중·비만", "영역": "임상", "지표": "bmi", "지표명": "BMI", "단위": "kg/m²", "공통": True,
        "기준": "BMI 25 이상 (대한비만학회)", "E": "섭취 열량이 소비보다 많은 생활 습관 (단 음료·간식·야식, 신체활동 부족 등)",
        "E변수": ["eatout_d", "skip_bf"], "측정": "검진 신체계측",
        "제공": "하루 섭취 열량을 지금보다 약 500kcal 줄이는 것을 출발점으로, 3~6개월에 체중 5% 이상 감량",
        "교육": ["밥은 2/3공기로 줄이고, 줄인 만큼 채소 반찬을 늘려요.",
               "단 음료·과자·야식 중 가장 자주 먹는 것 하나부터 끊어요.",
               "한 끼를 20분 이상 천천히 먹고, 채소 → 단백질 반찬 → 밥 순서로 먹어요.",
               "튀김·볶음 대신 구이·찜·무침을 골라요."],
        "상담": ["1~2주: 체중을 주 1회 같은 시간에 재고 기록해요.", "3~4주: 야식·간식을 주 2회 이하로 줄여요.",
               "2~3개월: 한 달에 1~2kg 감량을 목표로 점검해요."],
        "협의": "BMI 30 이상이거나 당뇨·고혈압이 함께 있으면 비만 진료를 권해요.",
    },
    "p_na": {
        "P": "나트륨 섭취 과다", "영역": "섭취", "지표": "sodium", "지표명": "나트륨 섭취량", "단위": "mg/일",
        "기준": f"나트륨 {SODIUM_CDRR:,}mg/일 초과 (만성질환위험감소섭취량)", "E": "국·찌개·김치류와 외식 중심의 식사",
        "E변수": ["eatout_d"], "측정": "24시간 회상법", "목표선": SODIUM_CDRR,
        "제공": f"나트륨 하루 {SODIUM_CDRR:,}mg 이하",
        "교육": ["국·찌개는 건더기 위주로 먹고 국물은 남겨요.",
               "김치·장아찌는 한 끼에 작은 접시 하나만 먹어요.",
               "라면은 스프를 절반만 넣고 국물은 마시지 않아요.",
               "외식할 때는 '싱겁게' 요청하고, 소스·양념은 따로 받아 찍어 먹어요.",
               "가공식품은 영양표시에서 나트륨(%영양성분 기준치)을 확인하고 골라요."],
        "상담": ["1~2주: 하루 한 끼 이상 국물 남기기", "3~4주: 김치 한 접시 지키기", "2개월: 외식 때 싱겁게 요청 주 3회"],
        "협의_키": "undx_ht", "협의": "혈압 140/90 이상인데 고혈압 진단을 받지 않은 사람은 진료를 먼저 권해요.",
    },
    "p_cho": {
        "P": "탄수화물 섭취 과다", "영역": "섭취", "지표": "cho_p", "지표명": "탄수화물 에너지 비율", "단위": "%",
        "기준": f"탄수화물 에너지 비율 {CHO_RANGE[1]}% 초과 (에너지적정비율 {CHO_RANGE[0]}~{CHO_RANGE[1]}%)",
        "E": "밥·면·떡·빵 위주의 식사와 단백질 반찬 부족", "E변수": ["skip_bf"], "측정": "24시간 회상법", "목표선": CHO_RANGE[1],
        "제공": f"탄수화물 에너지 비율 {CHO_RANGE[0]}~{CHO_RANGE[1]}%",
        "교육": ["흰쌀밥을 잡곡 1/3을 섞은 밥으로 바꾸고, 양은 2/3공기로 해요.",
               "밥과 면, 떡과 빵처럼 탄수화물 식품을 한 끼에 겹쳐 먹지 않아요.",
               "끼니마다 단백질 반찬(두부·생선·달걀·살코기)을 한 가지 넣어요.",
               "간식은 떡·빵 대신 우유 한 잔이나 견과 한 줌으로 바꿔요."],
        "상담": ["1~2주: 저녁 밥 2/3공기", "3~4주: 잡곡밥 주 5회", "2개월: 매끼 단백질 반찬 1가지"],
        "협의_키": "undx_dm", "협의": "공복혈당 126 이상 또는 당화혈색소 6.5% 이상인데 당뇨 진단을 받지 않은 사람은 확인 검사를 먼저 권해요.",
    },
    "p_sug": {
        "P": "당류 섭취 과다", "영역": "섭취", "지표": "sug_p", "지표명": "당류 에너지 비율", "단위": "%",
        "기준": f"총당류 에너지 비율 {SUGAR_MAX}% 초과 (첨가당은 10% 이내 제한)", "E": "단 음료·믹스커피·과자 같은 간식 섭취",
        "E변수": ["skip_bf"], "측정": "24시간 회상법", "목표선": SUGAR_MAX,
        "제공": f"총당류 에너지의 {SUGAR_MAX}% 이내, 첨가당 10% 이내",
        "교육": ["탄산음료·과일음료·믹스커피 대신 물·보리차·탄산수를 마셔요.",
               "커피는 시럽·설탕을 빼고 마셔요.",
               "과일은 주스 대신 생과일로, 하루 1~2회분 먹어요.",
               "과자·케이크 같은 간식은 주 2회 이하로 정해요."],
        "상담": ["1~2주: 단 음료 하루 1잔 이하", "3~4주: 믹스커피를 블랙이나 라떼로", "2개월: 단 간식 주 2회 이하"],
        "협의_키": "undx_dm", "협의": "공복혈당 126 이상 또는 당화혈색소 6.5% 이상인데 당뇨 진단을 받지 않은 사람은 확인 검사를 먼저 권해요.",
    },
    "p_fat": {
        "P": "지방 섭취 과다", "영역": "섭취", "지표": "fat_p", "지표명": "지방 에너지 비율", "단위": "%",
        "기준": f"지방 에너지 비율 {FAT_RANGE[1]}% 초과 (에너지적정비율 {FAT_RANGE[0]}~{FAT_RANGE[1]}%)",
        "E": "튀김·볶음과 기름진 육류 위주의 외식", "E변수": ["eatout_d", "drink"], "측정": "24시간 회상법", "목표선": FAT_RANGE[1],
        "제공": f"지방 에너지 비율 {FAT_RANGE[0]}~{FAT_RANGE[1]}%",
        "교육": ["튀김·전 대신 구이·찜을 골라요.",
               "삼겹살 같은 기름진 부위 대신 살코기, 껍질 뺀 닭고기를 먹어요.",
               "볶을 때 기름은 1작은술 정도만 써요.",
               "마요네즈·크림소스 대신 간장·식초 드레싱을 써요."],
        "상담": ["1~2주: 튀김 음식 주 1회 이하", "3~4주: 고기 메뉴는 살코기 부위로", "2개월: 외식 메뉴를 구이·찜 위주로"],
        "협의_키": "undx_dl", "협의": "총콜레스테롤 240 이상 또는 중성지방 200 이상인데 이상지질혈증 진단을 받지 않은 사람은 진료를 권해요.",
    },
    "p_fib": {
        "P": "식이섬유 섭취 부족", "영역": "섭취", "지표": "fiber", "지표명": "식이섬유 섭취량", "단위": "g/일",
        "기준": f"식이섬유 충분섭취량 미만 (남 {FIBER_AI['남']}g, 여 {FIBER_AI['여']}g · 2020 기준, 2025 개정값 확인)",
        "E": "채소·잡곡·콩류가 적은 식사", "E변수": ["eatout_d", "skip_bf"], "측정": "24시간 회상법",
        "제공": f"식이섬유 하루 남 {FIBER_AI['남']}g, 여 {FIBER_AI['여']}g 수준",
        "교육": ["끼니마다 채소 반찬을 두 가지 먹어요.",
               "흰밥 대신 잡곡밥을 먹어요.",
               "콩·두부를 주 3회 이상 먹어요.",
               "해조류·버섯을 국이나 반찬으로 자주 넣어요."],
        "상담": ["1~2주: 점심·저녁 채소 반찬 2가지", "3~4주: 잡곡밥 주 5회", "2개월: 콩·두부 주 3회"],
        "협의_키": None, "협의": "변비·장 질환 증상이 있으면 의료진과 상의해요.",
    },
    "skip_bf": {
        "P": "불규칙한 식사 패턴 (아침 결식)", "영역": "행동·환경", "지표": "skip_bf", "지표명": "아침 결식 비율", "단위": "",
        "기준": "아침식사 주 2회 이하", "E": "아침 준비 시간 부족, 늦은 저녁·야식", "E변수": ["eatout_d"], "측정": "식생활 설문",
        "제공": "아침식사 주 5회 이상",
        "교육": ["준비가 쉬운 아침 조합을 정해 둬요: 삶은 달걀+우유+과일, 잡곡 주먹밥+두유.",
               "전날 밤에 아침거리를 미리 꺼내 둬요.",
               "아침을 거르면 점심·저녁에 과식하기 쉬워요. 늦은 야식부터 줄여 보세요."],
        "상담": ["1~2주: 아침 주 3회", "3~4주: 아침 주 4회", "2개월: 아침 주 5회 이상"],
        "협의_키": None, "협의": "교대근무처럼 생활 패턴 때문이라면 직장 보건관리자와 함께 방법을 찾아요.",
    },
    "drink": {
        "P": "알코올 섭취 과다 의심", "영역": "섭취", "지표": "drink", "지표명": "음주 비율", "단위": "",
        "기준": "주 2회 이상 음주 (빈도만 있어 양은 상담에서 확인)", "E": "잦은 회식·술자리", "E변수": ["eatout_d"], "측정": "건강 설문",
        "제공": "음주 횟수를 주 1회 이하로, 한 번에 마시는 양도 줄이기",
        "교육": ["빈속에 마시지 않도록 술자리 전에 식사를 해요.",
               "안주는 튀김·찌개 대신 두부·생선구이·채소로 골라요.",
               "술 한 잔에 물 한 잔을 같이 마셔요.",
               "2차는 가지 않는 것을 원칙으로 해요."],
        "상담": ["1~2주: 음주 날짜 기록", "3~4주: 음주 주 2회 이하", "2개월: 음주 주 1회 이하"],
        "협의_키": None, "협의": "음주 조절이 어렵다면 보건소 절주 상담을 연계해요.",
    },
}


def diagnose(df):
    """진단 후보마다 중재 대상(위험군+대사증후군)과 정상 그룹의 해당 비율을 비교해요."""
    w, tgt, nor = df["wt"], df["target"], ~df["target"]
    rows = []
    for key, d in DX.items():
        x = df[key]
        if n_known(x[tgt]) < 30 or n_known(x[nor]) < 30:
            continue
        p_t, p_n = wmean(x[tgt], w[tgt]), wmean(x[nor], w[nor])
        rows.append({"key": key, "P": d["P"], "영역": d["영역"], "중재 대상": p_t, "정상": p_n, "차이": p_t - p_n,
                     "n": n_known(x[tgt]), "공통": d.get("공통", False)})
    return pd.DataFrame(rows)


def pick_top(tab, k=3, floor=0.2):
    """공통 처방(비만)은 따로 두고, 나머지 중 대상 그룹 해당 비율 20% 이상 · 정상보다 높은 것을 차이 순으로 k개."""
    if tab.empty:
        return [], []
    common = tab[tab["공통"] & (tab["중재 대상"] >= floor) & (tab["차이"] > 0)]["key"].tolist()
    rest = tab[~tab["공통"] & (tab["중재 대상"] >= floor) & (tab["차이"] > 0)].sort_values("차이", ascending=False)
    return common, rest["key"].head(k).tolist()


def pes_text(df, key):
    """PES 문장과 근거 숫자를 만들어요."""
    d, w, tgt, nor = DX[key], df["wt"], df["target"], ~df["target"]
    x = df[key]
    p_t, p_n = wmean(x[tgt], w[tgt]), wmean(x[nor], w[nor])
    ind = df[d["지표"]]
    if d["지표"] == key or d["단위"] == "":
        s = f"중재 대상 그룹 {d['지표명']} {p_t:.1%} (정상 그룹 {p_n:.1%})"
    else:
        s = (f"중재 대상 그룹 {d['지표명']} 평균 {wmean(ind[tgt], w[tgt]):,.1f}{d['단위']}, "
             f"기준 해당 비율 {p_t:.1%} (정상 그룹 {p_n:.1%})")
    # 원인 근거: P 해당자와 비해당자의 행동 비교 (중재 대상 그룹 안에서)
    ev = []
    for b in d["E변수"]:
        if b == key or df[b].notna().sum() < 30:
            continue
        yes = tgt & (x == 1)
        no = tgt & (x == 0)
        if n_known(df.loc[yes, b]) < 10 or n_known(df.loc[no, b]) < 10:
            continue
        a_, b_ = wmean(df.loc[yes, b], w[yes]), wmean(df.loc[no, b], w[no])
        if a_ > b_:
            label = E_BEHAV[b] if b != "drink" else "음주"
            ev.append(f"해당자 중 {label} {a_:.1%} (비해당자 {b_:.1%})")
    return s, ev


# ================================================================ 화면
st.caption("헬스케어 팩 · 담당: 식품영양학과")
st.title("검진·식사 데이터로 집단의 영양 문제를 진단하고, 맞춤 영양 중재까지")
st.write("영양사가 쓰는 표준 절차인 영양관리과정(NCP)을 집단 데이터에 맞게 바꿨어요. "
         "**영양판정**으로 누가 위험한지 찾고, **영양진단**으로 무엇이 문제인지 정하고, "
         "**영양중재**로 어떻게 바꿀지 제시하고, **평가계획**으로 무엇으로 확인할지 정해요.")

raw, mapping_used = simulate(), None
uploaded = False
with st.expander("데이터 올리기 (선택)"):
    st.caption("국민건강영양조사 원시자료에서 필요한 열만 뽑은 CSV를 올려 주세요 (저장소의 notebooks/국건영_CSV_만들기.ipynb). "
               "식사 변수가 없는 건강검진 CSV도 돼요. 이때는 섭취·행동 영역 진단이 빠져요.")
    up = st.file_uploader("CSV", type="csv")
    if up:
        try:
            raw_up = read_any_csv(up.getvalue())
            auto = guess_columns(raw_up.columns)
            st.markdown("**열 맞추기** · 자동으로 찾은 결과예요. 틀렸거나 (없음)이면 직접 골라 주세요. 별표(*)는 꼭 필요한 항목이에요.")
            options = [NONE] + list(raw_up.columns)
            mapping, grid = {}, st.columns(4)
            for i, (key, (label, need)) in enumerate(FIELDS.items()):
                default = auto[key] if auto[key] in options else NONE
                pick = grid[i % 4].selectbox(label + (" *" if need else ""), options, index=options.index(default), key=f"map_{key}_{up.name}")
                mapping[key] = None if pick == NONE else pick
            miss = missing_fields(mapping)
            if miss:
                st.error("아직 못 찾은 필수 항목: " + ", ".join(FIELDS[k][0] for k in miss))
            else:
                raw, mapping_used, uploaded = raw_up, mapping, True
        except Exception as e:
            st.error(f"CSV를 읽지 못했어요: {e}")

if mapping_used is None:
    mapping_used = guess_columns(raw.columns)
data, notes, meta = build_data(raw, mapping_used)
if len(data) < 100:
    st.error(f"분석할 수 있는 성인이 {len(data)}명뿐이라 예시 데이터로 보여줘요.")
    raw, uploaded = simulate(), False
    data, notes, meta = build_data(raw, guess_columns(raw.columns))

if uploaded:
    st.info(f"올린 데이터의 성인 {len(data):,}명으로 분석하고 있어요.")
else:
    st.warning("예시 데이터(국건영 변수 이름을 흉내 낸 시뮬레이션)로 보여주고 있어요. 숫자는 실제 통계가 아니에요.")
st.caption(("가중치를 적용한 값이에요 (국민 전체를 대표하는 추정). 표준오차까지 정확히 내려면 층·집락 정보를 쓰는 복합표본 분석이 필요해요."
            if meta["weighted"] else "가중치 열이 없어 표본 기준 값이에요. 국건영이면 가중치 열을 골라 주세요."))
for n_ in notes:
    st.caption("· " + n_)

df = add_status(data)
if meta.get("drink_label") == "음주함":
    DX["drink"].update({"P": "음주 (양·빈도는 상담에서 확인)", "기준": "음주 여부 (빈도·양 정보 없음)"})
has_dx = df[["dm_dx", "ht_dx"]].notna().any().any()
if has_dx:
    excl = st.toggle("이미 진단받은 사람(당뇨·고혈압·이상지질혈증) 빼고 보기", value=False,
                     help="진단 후 식사를 바꾼 사람이 섞이면 식사 비교가 거꾸로 나올 수 있어요 (역인과). 결과가 이상하면 켜서 비교해 보세요.")
    if excl:
        dxd = (df["dm_dx"] == 1) | (df["ht_dx"] == 1) | (df["dl_dx"] == 1)
        df = df[~dxd].reset_index(drop=True)
        st.caption(f"진단받은 {int(dxd.sum()):,}명을 빼고 {len(df):,}명으로 보고 있어요.")
w = df["wt"]
has_diet = df["energy"].notna().sum() >= 30

dx_tab = diagnose(df)
common_keys, top_keys = pick_top(dx_tab)

t1, t2, t3, t4 = st.tabs(["① 영양판정", "② 영양진단", "③ 영양중재", "④ 평가계획"])

# ---------------------------------------------------------------- ① 영양판정
with t1:
    st.subheader("누가, 어떤 상태인가")
    shares = {g_: wmean((df["group"] == g_).astype(float), w) for g_ in GROUPS}
    counts = df["group"].value_counts()
    cols = st.columns(4)
    cols[0].metric("분석 인원 (성인)", f"{len(df):,}명")
    for c_, g_ in zip(cols[1:], GROUPS):
        c_.metric(f"{g_} ({counts.get(g_, 0):,}명)", f"{shares[g_]:.1%}")
    st.caption("대사증후군 기준 5개(허리둘레, 중성지방, HDL, 혈압, 공복혈당) 중 해당 개수로 나눴어요. "
               "0개 정상 · 1~2개 위험군(예방 중재) · 3개 이상 대사증후군(집중 중재). "
               "혈압약·혈당약을 먹고 있으면 해당 항목에 해당하는 것으로 셌어요.")

    fig = go.Figure(go.Bar(x=[shares[g_] for g_ in GROUPS], y=GROUPS, orientation="h",
                           marker_color=[COLORS[g_] for g_ in GROUPS],
                           text=[f"{shares[g_]:.1%}" for g_ in GROUPS], textposition="outside"))
    fig.update_layout(height=200, margin=dict(t=10, b=10, l=10, r=40), xaxis_tickformat=".0%",
                      xaxis_range=[0, max(shares.values()) * 1.2], yaxis_autorange="reversed")
    st.plotly_chart(fig, width="stretch")

    st.markdown("**영양판정 5개 영역으로 그룹 비교**")
    rows = []
    for col, name, area, kind, used in INDICATORS:
        x = df[col]
        if n_known(x) < 30:
            continue
        row = {"영역": area, "지표": name + (" ·판정 기준" if used else "")}
        for g_ in GROUPS:
            m = df["group"] == g_
            row[g_] = "가림" if 0 < n_known(x[m]) < 5 else fmt(wmean(x[m], w[m]), kind)
        row["효과크기"] = smd(x, w, df["group"] == "대사증후군", df["group"] == "정상")
        rows.append(row)
    comp = pd.DataFrame(rows)
    st.dataframe(comp, width="stretch", hide_index=True,
                 column_config={"효과크기": st.column_config.NumberColumn(format="%.2f",
                                help="대사증후군 그룹과 정상 그룹의 표준화 평균 차이. 0.2 작음 · 0.5 중간 · 0.8 큼")})
    st.caption("식사 지표는 에너지 보정 값(1,000kcal당, 에너지 비율)을 함께 봐요. 많이 먹는 사람이 모든 영양소를 많이 먹는 효과를 빼야 그룹을 공정하게 비교할 수 있어요. "
               "'식사력'이 비어 있으면 식사 자료가 없는 데이터예요.")

    st.markdown("**바꿀 수 있는 요인 중 차이가 큰 순서**")
    rank = comp[comp["영역"].isin(["신체계측", "식사력", "생활습관"]) & ~comp["지표"].str.contains("판정 기준")
                & ~comp["지표"].isin(["나트륨 (mg/일)", "에너지 섭취 (kcal)"])].dropna(subset=["효과크기"])
    rank = rank.reindex(rank["효과크기"].abs().sort_values().index)
    if len(rank):
        fig = go.Figure(go.Bar(x=rank["효과크기"], y=rank["지표"], orientation="h",
                               marker_color=np.where(rank["효과크기"] > 0, "#C8553D", "#4C9F70")))
        fig.add_vline(x=0, line_color="#888")
        fig.update_layout(height=40 + 30 * len(rank), margin=dict(t=10, b=10, l=10, r=10), xaxis_title="효과크기 (대사증후군 − 정상)")
        st.plotly_chart(fig, width="stretch")
        st.caption("판정 기준 5개(허리·혈압·혈당·중성지방·HDL)는 그룹을 나눈 기준이라 차이가 큰 게 당연해서 순위에서 뺐어요. "
                   "빨간 막대는 대사증후군 그룹에서 더 높고, 초록 막대는 더 낮아요. 같은 날 잰 자료라 원인이 아니라 함께 나타나는 관계예요.")

    st.markdown("**혈당 단계** · 국민건강통계 당뇨 정의")
    stage = {s_: wmean((df["glu_stage"] == s_).astype(float), w) for s_ in ["정상", "당뇨 전단계", "당뇨"]}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("혈당 정상", f"{stage['정상']:.1%}")
    c2.metric("당뇨 전단계", f"{stage['당뇨 전단계']:.1%}", help="공복혈당 100~125 또는 당화혈색소 5.7~6.4%")
    c3.metric("당뇨", f"{stage['당뇨']:.1%}", help="공복혈당 126 이상, 당화혈색소 6.5% 이상, 의사진단, 혈당강하제·인슐린 중 하나")
    if df["dm_dx"].notna().any():
        c4.metric("수치는 높은데 진단 없음", f"{int(df['undx_dm'].sum()):,}명", help="③ 영양중재의 다분야 협의(진료 연계) 대상이에요")
    if df["hba1c"].isna().all():
        st.caption("당화혈색소·진단·약 정보가 없으면 공복혈당만으로 판정해서, 공복혈당은 정상인 당뇨인을 놓쳐요.")
    st.caption("당뇨 전단계는 영양 중재로 진행을 늦출 수 있는 핵심 대상이에요.")

# ---------------------------------------------------------------- ② 영양진단
with t2:
    st.subheader("무엇이 문제이고, 왜 그런가")
    st.write("영양판정 결과에서 문제를 골라 **집단 PES**로 써요. 문제(P), 원인(E), 근거(S)라는 형식은 개인 영양진단과 같고, "
             "근거(S)가 한 사람의 수치 대신 집단 통계라는 점만 달라요.")
    if dx_tab.empty:
        st.info("진단 후보를 계산할 자료가 부족해요.")
    else:
        show = dx_tab.copy().sort_values(["공통", "차이"], ascending=[False, False])
        show["선택"] = show["key"].map(lambda k: "공통 처방" if k in common_keys else ("상위 3" if k in top_keys else ""))
        st.dataframe(show[["P", "영역", "중재 대상", "정상", "차이", "선택"]], width="stretch", hide_index=True,
                     column_config={"중재 대상": st.column_config.NumberColumn("중재 대상 그룹 해당", format="percent"),
                                    "정상": st.column_config.NumberColumn("정상 그룹 해당", format="percent"),
                                    "차이": st.column_config.NumberColumn("차이 (%p)", format="percent")})
        st.caption("중재 대상 그룹 = 위험군 + 대사증후군. 고르는 규칙: 중재 대상 그룹에서 해당 비율이 20% 이상이고 정상 그룹보다 높은 것 중 차이가 큰 3개. "
                   "과체중·비만은 대부분에게 공통인 기본 처방으로 따로 둬요.")
        if not has_diet:
            st.info("식사 자료가 없어 섭취·행동 영역 진단은 계산하지 않았어요. 국건영 영양조사 자료를 올리면 나타나요.")

        chosen = st.multiselect("영양진단으로 쓸 문제 (기본은 자동 선택)", options=show["key"].tolist(),
                                default=common_keys + top_keys, format_func=lambda k: DX[k]["P"], key=f"dx_pick_{uploaded}_{len(df)}")
        st.session_state["chosen"] = chosen
        for key in chosen:
            d = DX[key]
            s, ev = pes_text(df, key)
            with st.container(border=True):
                st.markdown(f"**{d['P']}** · {d['영역']} 영역" + (" · 공통 처방" if key in common_keys else ""))
                st.markdown(f"- **P (문제)** {d['P']}\n- **E (원인)** {d['E']}"
                            + (f"\n    - 데이터: {'; '.join(ev)}" if ev else "\n    - 데이터로 확인되지 않아 상담에서 확인할 원인 후보예요.")
                            + f"\n- **S (근거)** {s}")
                st.caption(md(f"기준: {d['기준']}"))
                st.markdown(md(f"> 이 집단의 영양 문제는 '**{d['P']}**'이다. 원인 후보는 '**{d['E']}**'이고, 근거는 '**{s}**'이다."))

# ---------------------------------------------------------------- ③ 영양중재
with t3:
    st.subheader("무엇을, 어떻게 바꿀까")
    chosen = st.session_state.get("chosen", common_keys + top_keys)
    st.write("영양진단 하나마다 카드 한 장이에요. 영양중재 4영역(식품·영양소 제공, 영양교육, 영양상담, 다분야 협의)을 틀로 썼어요. "
             "데이터는 **누구에게, 어떤 문제부터** 할지를 정하고, 중재 내용은 근거 기반 가이드라인에서 와요.")
    mets = df["group"] == "대사증후군"
    if mets.sum() >= 5:
        lean = wmean((df.loc[mets, "bmi"] < 25).astype(float), w[mets])
        st.info(f"대사증후군 그룹의 {lean:.1%}는 BMI 25 미만이에요. 이 사람들에게는 체중 감량보다 식사의 질(아래 문제별 중재)이 중심이에요.")
    if not chosen:
        st.info("② 영양진단 탭에서 문제를 하나 이상 골라 주세요.")
    for key in chosen:
        d = DX[key]
        n_target = int((df["target"] & (df[key] == 1)).sum())
        with st.container(border=True):
            st.markdown(f"### {d['P']}")
            st.caption(f"중재 대상 그룹에서 해당하는 사람 {n_target:,}명 (표본 기준)")
            a, b = st.columns(2)
            with a:
                st.markdown("**식품·영양소 제공** · 목표")
                st.write(d["제공"])
                st.markdown("**영양교육** · 구체적 실천 방안")
                st.markdown("\n".join(f"- {md(t)}" for t in d["교육"]))
            with b:
                st.markdown("**영양상담** · 작게 시작하는 단계별 목표")
                st.markdown("\n".join(f"- {md(t)}" for t in d["상담"]))
                st.markdown("**영양관리를 위한 다분야 협의** · 연계")
                ck = d.get("협의_키")
                if ck and df[ck].notna().any() and df[{"undx_ht": "ht_dx", "undx_dm": "dm_dx", "undx_dl": "dl_dx"}[ck]].notna().any():
                    st.write(f"{d['협의']} 해당 {int((df['target'] & df[ck]).sum()):,}명")
                else:
                    st.write(d["협의"])
    st.caption("중재 문구는 식품영양학과 검토 초안이에요. 섭취기준·진료지침 원문으로 목표값을 확인해야 해요.")

# ---------------------------------------------------------------- ④ 평가계획
with t4:
    st.subheader("효과를 무엇으로, 어떻게 확인할까")
    st.write("국건영은 매년 다른 사람을 조사해서 같은 사람의 변화를 따라갈 수 없어요. 그래서 개인 모니터링 대신 **평가계획**을 세워요. "
             "영양진단의 근거(S)가 그대로 평가 지표가 되고, 영양판정 5개 영역 중 과거력을 뺀 4개 영역에서 지표를 골라요.")
    chosen = st.session_state.get("chosen", common_keys + top_keys)
    cut = st.slider("목표: 해당 비율을 지금보다 몇 % 줄일까 (상대 감소)", 5, 50, 20, 5)
    tgt = df["target"]
    plan = []
    for key in chosen:
        d = DX[key]
        now = wmean(df.loc[tgt, key], w[tgt])
        plan.append({"영양진단": d["P"], "지표": f"{d['지표명']} 기준 해당 비율", "현재": now, "목표": now * (1 - cut / 100),
                     "측정 방법": d["측정"], "재평가": "1년 (국건영 조사 주기)"})
    if plan:
        st.dataframe(pd.DataFrame(plan), width="stretch", hide_index=True,
                     column_config={"현재": st.column_config.NumberColumn(format="percent"),
                                    "목표": st.column_config.NumberColumn(format="percent")})
        st.caption("목표 감소 폭은 팀 논의 후 정해요. 지금은 슬라이더로 바꿔 볼 수 있어요.")

    st.markdown("**기대 효과** · 기준을 넘는 사람이 기준선까지 맞추면")
    eff = []
    for key in chosen:
        d = DX[key]
        x = df[d["지표"]]
        if key == "obese":
            after_bmi = np.where(df["bmi"] >= 25, df["bmi"] * 0.95, df["bmi"])
            before, after = wmean(df.loc[tgt, "obese"], w[tgt]), wmean(pd.Series(after_bmi >= 25, index=df.index)[tgt].astype(float), w[tgt])
            eff.append({"영양진단": d["P"], "가정": "BMI 25 이상인 사람이 체중 5% 감량", "지표": "비만 비율",
                        "지금": f"{before:.1%}", "가정 후": f"{after:.1%}"})
        elif "목표선" in d and x.notna().sum() >= 30:
            capped = x.clip(upper=d["목표선"])
            before, after = wmean(x[tgt], w[tgt]), wmean(capped[tgt], w[tgt])
            eff.append({"영양진단": d["P"], "가정": f"기준 초과자가 {d['목표선']:,}{d['단위']}까지 줄임",
                        "지표": f"그룹 평균 {d['지표명']}", "지금": f"{before:,.1f}{d['단위']}", "가정 후": f"{after:,.1f}{d['단위']}"})
    if eff:
        st.dataframe(pd.DataFrame(eff), width="stretch", hide_index=True)
        st.caption("단순 계산이에요. 실제로 그만큼 바뀐다는 보장이 아니라, 목표를 세울 때 크기를 가늠하는 용도예요.")
    st.markdown("**한계**")
    st.markdown("- 같은 날 잰 단면 자료라, 식사와 대사 위험이 함께 나타나는 관계(연관성)만 볼 수 있어요.\n"
                "- 24시간 회상법은 하루치라 한 사람의 평소 식사를 대표하지 못해요. 집단 평균·비율로만 해석해요.\n"
                "- 이미 진단받은 사람은 식사를 바꿨을 수 있어요. 위의 '진단받은 사람 빼고 보기'로 비교해 보세요.\n"
                "- 연도별 비교는 '같은 사람이 좋아졌다'가 아니라 '집단 전체 수준이 변했다'는 뜻이에요.")
