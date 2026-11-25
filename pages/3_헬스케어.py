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
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st

st.set_page_config(page_title="헬스케어", layout="wide")

# ================================================================ 기준값
# 국건영 제9기(2022-2024)가 쓰는 2020 한국인 영양소 섭취기준(KDRIs)에 맞췄어요 (원시자료 이용지침서 SAS 프로그램 기준).
# 2025 KDRIs 개정값으로 바꿀 때는 이 칸만 고치면 돼요.
KDRI_NAME = "2020 한국인 영양소 섭취기준"
# 에너지 필요추정량 (kcal/일): (성별, 나이 하한) → 값. 나이 구간 19~29, 30~49, 50~64, 65~74, 75 이상
EER = {1: [(19, 2600), (30, 2500), (50, 2200), (65, 2000), (75, 1900)],
       2: [(19, 2000), (30, 1900), (50, 1700), (65, 1600), (75, 1500)]}
EN_HIGH, EN_LOW = 125, 75   # 필요추정량의 125% 초과 = 에너지 과잉, 75% 미만 = 에너지 부족 (국민건강통계 지표 정의)
SODIUM_CDRR = [(19, 2300), (65, 2100), (75, 1700)]  # 나트륨 만성질환위험감소섭취량 (mg/일)
CHO_RANGE = (55, 65)        # 탄수화물 에너지적정비율 (%)
FAT_RANGE = (15, 30)        # 지방 에너지적정비율 (%)
SUGAR_MAX = 20              # 총당류 에너지 비율 상한 (%) (첨가당은 10% 이내)
FIBER_AI = {1: [(19, 30), (65, 25)], 2: [(19, 20)]}  # 식이섬유 충분섭취량 (g/일)
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
    "prg": ("임신 여부 (임신부는 성인 기준에서 빼요)", False),
    "smoke": ("현재 흡연 상태", False), "smoke_life": ("평생 흡연량 (5갑 이상 여부)", False),
    "drink": ("음주 빈도 (또는 Y/N)", False), "drink_amt": ("한 번 음주량", False),
    "breakfast": ("아침식사 빈도", False), "eatout": ("외식 빈도", False),
    "pa": ("유산소 신체활동 실천 (0/1)", False), "income": ("가구 소득 4분위", False), "edu": ("교육수준 (1~4)", False),
    "town": ("동/읍면", False), "cfam": ("가구원 수", False), "nedu": ("영양교육·상담 경험", False),
    "label_co": ("영양표시 인지", False), "label_us": ("영양표시 이용", False),
    "energy": ("에너지 섭취량 (kcal)", False), "prot": ("단백질 (g)", False), "cho": ("탄수화물 (g)", False), "fat": ("지방 (g)", False),
    "sugar": ("당류 (g)", False), "fiber": ("식이섬유 (g)", False), "sodium": ("나트륨 (mg)", False),
    "wt": ("가중치 (없으면 표본 기준)", False),
    "psu": ("조사구 (집락, 신뢰구간 계산용)", False), "strata": ("층 (신뢰구간 계산용)", False),
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
    "dm_med": ["de132", "혈당강하제", "혈당강하제복용"],   # 국건영 DE1_32 = 당뇨병약 복용
    "dm_ins": ["de131", "인슐린", "인슐린주사"],          # 국건영 DE1_31 = 인슐린 주사
    "ht_dx": ["di1dg", "고혈압진단", "고혈압의사진단"],
    "ht_med": ["di12", "혈압조절제", "혈압조절제복용"],
    "dl_dx": ["di2dg", "이상지질혈증진단"],
    "prg": ["heprg", "임신여부"],
    "smoke": ["bs31", "smkstattypecd", "smoke", "흡연", "흡연상태"],
    "smoke_life": ["bs11", "평생흡연량"],
    "drink": ["bd111", "drkyn", "drink", "음주", "음주여부", "음주빈도"],
    "drink_amt": ["bd21", "한번음주량", "음주량"],
    "breakfast": ["lbrfq", "아침식사빈도", "아침식사"],
    "eatout": ["loutfq", "외식빈도", "외식"],
    "pa": ["paaerobic", "유산소신체활동"],
    "income": ["hoincm", "incm", "소득4분위", "소득"],
    "edu": ["edu", "교육수준"],
    "town": ["townt", "동읍면"],
    "cfam": ["cfam", "가구원수"],
    "nedu": ["lkedu", "영양교육경험"],
    "label_co": ["lklbco", "영양표시인지"],
    "label_us": ["lklbus", "영양표시이용"],
    "energy": ["nen", "energy", "에너지", "열량"],
    "prot": ["nprot", "단백질"],
    "cho": ["ncho", "탄수화물"],
    "fat": ["nfat", "지방"],
    "sugar": ["nsugar", "당류", "총당류"],
    "fiber": ["ntdf", "식이섬유"],
    "sodium": ["nna", "sodium", "나트륨"],
    "wt": ["wttot", "wtntr", "wtitvex", "weight_s", "가중치"],
    "psu": ["psu", "조사구"],
    "strata": ["kstrata", "strata", "층"],
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


def med_flag(s):
    """국건영 약 복용 빈도(DI1_2 등): 1=매일, 2=한 달 20일 이상, 3=15일 이상, 4=15일 미만 → 복용,
    5=복용 안 함·8=비해당 → 0, 9=모름 → 빈칸. 0/1 열이나 Y/N 열도 처리해요."""
    if not pd.api.types.is_numeric_dtype(s):
        return yes_no(s)
    return s.map({0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 0, 8: 0}).astype(float)


def smoke_now(s, colname, life=None):
    """현재 흡연. 국건영 공식 정의: 평생 5갑 이상(BS1_1=2) 피웠고 지금 매일·가끔 피움(BS3_1=1,2).
    공단 검진(SMK_STAT_TYPE_CD)은 3=현재 흡연."""
    if norm(colname) == "smkstattypecd":
        return s.map({1: 0, 2: 0, 3: 1}).astype(float)
    now = s.map({1: 1, 2: 1, 3: 0, 8: 0}).astype(float)
    if life is None:
        return now
    out = pd.Series(np.nan, index=s.index)
    out[life.isin([1, 3])] = 0                       # 5갑 미만이거나 피운 적 없음
    lifetime = life == 2
    out[lifetime] = now[lifetime]                      # 5갑 이상이면 지금 피우는지로
    return out


def drink_flag(s, sex=None, amt=None):
    """국건영 공식 고위험음주: 주 2회 이상(BD1_11=5,6) 마시고 한 번에 남 7잔(BD2_1=4,5)·여 5잔(BD2_1=3,4,5) 이상.
    한 번 음주량이 없으면 주 2회 이상 음주로, Y/N이면 음주 여부로 대신해요."""
    if not pd.api.types.is_numeric_dtype(s):
        return yes_no(s), "음주함"
    if s.dropna().isin([0, 1]).all():
        return s.astype(float), "음주함"
    freq = s.map({1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 1, 8: 0}).astype(float)
    if amt is None or sex is None:
        return freq, "주 2회 이상 음주"
    heavy = np.where(sex == 1, amt.isin([4, 5]), amt.isin([3, 4, 5])).astype(float)
    heavy = pd.Series(heavy, index=s.index).where(amt.isin([1, 2, 3, 4, 5]))
    out = freq.copy()
    often = freq == 1
    out[often] = heavy[often]                          # 자주 마시는 사람만 양으로 다시 판정
    return out, "고위험음주"


def by_age(table, age):
    """[(나이 하한, 값), ...] 표에서 나이에 맞는 값을 골라요."""
    out = pd.Series(np.nan, index=age.index)
    for lo, v in table:
        out[age >= lo] = v
    return out


def kdri(sex, age, table):
    """성별·나이별 기준값. table은 {성별: [(나이 하한, 값), ...]} 또는 [(나이 하한, 값), ...]."""
    if isinstance(table, list):
        return by_age(table, age)
    out = pd.Series(np.nan, index=age.index)
    for sx, rows in table.items():
        m = sex == sx
        out[m] = by_age(rows, age[m])
    return out


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
              "energy", "prot", "cho", "fat", "sugar", "fiber", "sodium", "wt"):
        out[k] = num(k) if mapping.get(k) else np.nan
    for k in ("psu", "strata"):   # 조사구·층은 'A123' 같은 글자 ID일 수 있어서 그대로 둬요
        out[k] = raw[mapping[k]].astype(str).where(raw[mapping[k]].notna()) if mapping.get(k) else np.nan
    if not mapping.get("bmi"):
        out["bmi"] = (out["weight"] / (out["height"] / 100) ** 2).round(1)
        notes.append("BMI 열이 없어 몸무게 ÷ 키(m)²로 계산했어요.")

    for k in ("dm_dx", "dm_med", "dm_ins", "ht_dx", "dl_dx"):
        out[k] = yes_no(raw[mapping[k]]) if mapping.get(k) else np.nan
    out["ht_med"] = med_flag(raw[mapping["ht_med"]]) if mapping.get("ht_med") else np.nan
    if mapping.get("smoke"):
        out["smoke"] = smoke_now(num("smoke"), mapping["smoke"], num("smoke_life") if mapping.get("smoke_life") else None)
    else:
        out["smoke"] = np.nan
    meta["smoke_label"] = "현재 흡연" + ("" if mapping.get("smoke_life") else " (평생 흡연량 없이 판정)")
    meta["drink_label"] = "음주"
    if mapping.get("drink"):
        col = raw[mapping["drink"]]
        col = pd.to_numeric(col, errors="coerce") if pd.api.types.is_numeric_dtype(col) else col
        out["drink"], meta["drink_label"] = drink_flag(col, out["sex"], num("drink_amt") if mapping.get("drink_amt") else None)
    else:
        out["drink"] = np.nan
    # 아침식사 빈도(L_BR_FQ): 1=주5~7회, 2=주3~4회, 3=주1~2회, 4=거의 안 함 → 주 2회 이하를 결식으로
    out["skip_bf"] = num("breakfast").map({1: 0, 2: 0, 3: 1, 4: 1}).astype(float) if mapping.get("breakfast") else np.nan
    # 외식 빈도(L_OUT_FQ): 1=하루 2회 이상, 2=하루 1회 → 하루 1회 이상을 잦은 외식으로
    out["eatout_d"] = num("eatout").map({1: 1, 2: 1, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}).astype(float) if mapping.get("eatout") else np.nan
    # 원인(E) 분석용 생활·사회경제 요인 (국건영 코드 기준, 9=모름은 빈칸)
    opt = lambda k, f: f(num(k)).astype(float) if mapping.get(k) else np.nan
    out["inactive"] = opt("pa", lambda x: x.map({0: 1, 1: 0}))
    out["low_inc"] = opt("income", lambda x: x.map({1: 1, 2: 0, 3: 0, 4: 0}))
    out["low_edu"] = opt("edu", lambda x: x.map({1: 1, 2: 1, 3: 0, 4: 0}))
    out["rural"] = opt("town", lambda x: x.map({1: 0, 2: 1}))
    out["alone"] = opt("cfam", lambda x: (x == 1).where(x.notna()))
    out["no_nedu"] = opt("nedu", lambda x: x.map({1: 0, 2: 1}))
    if mapping.get("label_us"):
        us = num("label_us")
        co = num("label_co") if mapping.get("label_co") else pd.Series(np.nan, index=raw.index)
        out["no_label"] = us.map({1: 0, 2: 1, 8: 1}).astype(float).where(~((us == 8) & (co == 9)))
    else:
        out["no_label"] = np.nan

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
        out.loc[odd, ["energy", "prot", "cho", "fat", "sugar", "fiber", "sodium"]] = np.nan
        if odd.any():
            notes.append(f"하루 섭취 열량이 {ENERGY_OK[0]:,}kcal 미만이거나 {ENERGY_OK[1]:,}kcal 초과인 {int(odd.sum()):,}명은 식사 자료를 뺐어요 (과소·과대 보고 의심).")

    before = len(out)
    out = out[out["age"] >= 19]
    if len(out) < before:
        notes.append(f"19세 미만 {before - len(out):,}명은 성인 기준을 쓰기 위해 뺐어요.")
    if mapping.get("prg"):
        preg = pd.to_numeric(raw.loc[out.index, mapping["prg"]], errors="coerce") == 1
        out = out[~preg]
        if preg.any():
            notes.append(f"임신부 {int(preg.sum()):,}명은 허리둘레·혈당·섭취기준이 달라 뺐어요.")
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
    inactive = (r.random(n) < 0.45 + 0.1 * a).astype(int)
    label_use = (r.random(n) < np.clip(0.45 - 0.12 * a - 0.1 * male, 0.05, 0.9)).astype(int)
    low_inc = (r.random(n) < np.clip(0.2 + 0.1 * a, 0.05, 0.6)).astype(int)

    energy = np.clip(np.where(male == 1, 2250, 1650) - 120 * a + 180 * out_d + np.where(male == 1, 650, 480) * g(), 400, 5500)
    cho_p = np.clip(61 + 4 * a - 2 * out_d + 8 * g(), 30, 85)
    fat_p = np.clip(22 - 3 * a + 3 * out_d + 6.5 * g(), 5, 50)
    sug_p = np.clip(13 - 1 * a + 1.5 * (bf >= 3) + 6 * g(), 2, 45)
    na_dens = np.clip(1.6 + 0.25 * out_d + 0.1 * a + 0.12 * (1 - label_use) + 0.08 * low_inc + 0.35 * g(), 0.5, 4)        # mg/kcal
    fib_dens = np.clip(13 + 2 * a - 1.5 * out_d + 3.5 * g(), 3, 35)             # g/1000kcal

    bmi = np.clip(23.4 + 1.0 * male + 0.4 * a - 0.35 * a ** 2 + 0.06 * (sug_p - 13) + 0.05 * (fat_p - 22) + 0.4 * out_d + 0.7 * inactive
                  + 0.25 * (energy - 2000) / 500 + 3.1 * g(), 15, 45)
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
        "HE_prg": np.where(male == 1, 8, np.where((age < 45) & (r.random(n) < 0.01), 1, 0)),
        "DE1_dg": code(np.where(dm_dx, 1, 0)), "DE1_31": np.where(dm_dx, (r.random(n) < 0.1).astype(int), 8),
        "DE1_32": np.where(dm_dx, np.where(dm_med, 1, 0), 8),
        "DI1_dg": code(np.where(ht_dx, 1, 0)), "DI1_2": np.where(ht_dx, np.where(ht_med, r.choice([1, 1, 1, 2, 4], n), 5), 8),
        "DI2_dg": code(np.where(dl_dx, 1, 0)),
        "BS1_1": np.where(smoke == 8, 3, np.where(r.random(n) < 0.95, 2, 1)),
        "BS3_1": code(smoke), "BD1_11": code(drink),
        "BD2_1": np.where(male == 1, r.choice([1, 2, 3, 4, 5], n, p=[0.2, 0.2, 0.2, 0.2, 0.2]), r.choice([1, 2, 3, 4, 5], n, p=[0.4, 0.25, 0.15, 0.12, 0.08])),
        "L_BR_FQ": code(bf), "L_OUT_FQ": code(eatout),
        "N_EN": energy.round(), "N_PROT": (energy * np.clip(100 - cho_p - fat_p, 8, 30) / 100 / 4).round(1),
        "N_CHO": (energy * cho_p / 100 / 4).round(1), "N_FAT": (energy * fat_p / 100 / 9).round(1),
        "N_SUGAR": (energy * sug_p / 100 / 4).round(1), "N_TDF": (energy * fib_dens / 1000).round(1),
        "N_NA": (energy * na_dens).round(), "wt_tot": np.round(np.exp(np.log(7000) + 0.5 * g()) * np.where(age > 65, 0.8, 1.0)),
        "kstrata": r.integers(1, 28, n),
        "pa_aerobic": np.where(r.random(n) < 0.08, np.nan, 1 - inactive), "ho_incm": np.where(low_inc == 1, 1, r.choice([2, 3, 4], n)),
        "edu": np.where(age > 65, r.choice([1, 2, 3, 4], n, p=[0.45, 0.2, 0.25, 0.1]), r.choice([1, 2, 3, 4], n, p=[0.03, 0.05, 0.37, 0.55])),
        "town_t": np.where(r.random(n) < 0.18, 2, 1), "cfam": r.choice([1, 2, 3, 4, 5], n, p=[0.18, 0.35, 0.22, 0.18, 0.07]),
        "LK_EDU": np.where(r.random(n) < 0.06, 1, 2), "LK_LB_CO": np.where(label_use == 1, 1, np.where(r.random(n) < 0.3, 2, 1)),
    })
    df["LK_LB_US"] = np.where(df["LK_LB_CO"] == 2, 8, np.where(label_use == 1, 1, 2))
    df["psu"] = df["kstrata"] * 10 + r.integers(0, 7, n)
    lab_miss = r.random(n) < 0.04
    df.loc[lab_miss, ["HE_TG", "HE_HDL_st2", "HE_chol", "HE_HbA1c"]] = np.nan
    diet_miss = r.random(n) < 0.12
    df.loc[diet_miss, ["N_EN", "N_PROT", "N_CHO", "N_FAT", "N_SUGAR", "N_TDF", "N_NA"]] = np.nan
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

    # 식사: 성별·나이별 섭취기준(KDRIs)과 비교해요
    # 에너지 비율 분모는 국건영 지침처럼 탄수화물·지방·단백질에서 온 열량의 합 (단백질이 없으면 총 에너지)
    macro = df["cho"] * 4 + df["fat"] * 9 + df["prot"] * 4
    denom = macro.where(df["prot"].notna(), df["energy"])
    df["cho_p"] = df["cho"] * 4 / denom * 100
    df["fat_p"] = df["fat"] * 9 / denom * 100
    df["sug_p"] = df["sugar"] * 4 / denom * 100
    df["eer"] = kdri(df["sex"], df["age"], EER)
    df["en_eer"] = df["energy"] / df["eer"] * 100
    df["na_lim"] = kdri(df["sex"], df["age"], SODIUM_CDRR)
    df["fib_ai"] = kdri(df["sex"], df["age"], FIBER_AI)
    df["obese"] = (df["bmi"] >= 25).astype(float)
    flag = lambda cond, base: cond.where(base.notna()).astype(float)
    df["p_en_hi"] = flag(df["en_eer"] > EN_HIGH, df["en_eer"])
    df["p_en_lo"] = flag(df["en_eer"] < EN_LOW, df["en_eer"])
    df["p_na"] = flag(df["sodium"] > df["na_lim"], df["sodium"])
    df["p_cho"] = flag(df["cho_p"] > CHO_RANGE[1], df["cho_p"])
    df["p_fat"] = flag(df["fat_p"] > FAT_RANGE[1], df["fat_p"])
    df["p_sug"] = flag(df["sug_p"] > SUGAR_MAX, df["sug_p"])
    df["p_fib"] = flag(df["fiber"] < df["fib_ai"], df["fiber"])
    return df


# A 건강 결과: 대사증후군 판정 기준 5개 (열, 이름, 기준)
CRITERIA = [
    ("m_waist", "복부비만", "허리둘레 남 90cm·여 85cm 이상"),
    ("m_bp", "혈압 높음", "130/85mmHg 이상 또는 혈압약"),
    ("m_glu", "공복혈당 높음", "100mg/dL 이상 또는 당뇨약·인슐린"),
    ("m_tg", "중성지방 높음", "150mg/dL 이상"),
    ("m_hdl", "HDL 낮음", "남 40·여 50mg/dL 미만"),
]

# B 식사·생활 판정: (열, 이름, 영역, 기준 설명)
JUDGE = [
    ("p_en_hi", "에너지 과잉 섭취", "식사", f"필요추정량의 {EN_HIGH}% 초과 (성별·나이별)"),
    ("p_en_lo", "에너지 부족 섭취", "식사", f"필요추정량의 {EN_LOW}% 미만 (성별·나이별)"),
    ("p_cho", "탄수화물 비율 높음", "식사", f"에너지의 {CHO_RANGE[1]}% 초과 (적정 {CHO_RANGE[0]}~{CHO_RANGE[1]}%)"),
    ("p_fat", "지방 비율 높음", "식사", f"에너지의 {FAT_RANGE[1]}% 초과 (적정 {FAT_RANGE[0]}~{FAT_RANGE[1]}%)"),
    ("p_sug", "당류 비율 높음", "식사", f"에너지의 {SUGAR_MAX}% 초과"),
    ("p_na", "나트륨 기준 초과", "식사", "만성질환위험감소섭취량 초과 (나이별 2,300·2,100·1,700mg)"),
    ("p_fib", "식이섬유 충분섭취량 미만", "식사", "남 30g(65세 이상 25g) · 여 20g 미만 · '부족' 판정은 아님"),
    ("skip_bf", "아침 결식", "생활", "아침식사 주 2회 이하"),
    ("eatout_d", "잦은 외식", "생활", "외식 하루 1회 이상"),
    ("drink", "고위험음주", "생활", "주 2회 이상, 한 번에 남 7잔·여 5잔 이상"),
    ("smoke", "현재 흡연", "생활", "평생 5갑 이상 피웠고 지금도 피움"),
]

# 상세 표 (그룹 평균 비교): (열, 이름, 영역, 종류, 판정 기준에 쓰였는지)
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
    ("en_eer", "에너지 섭취 (필요추정량 대비 %)", "식사력", "mean", False),
    ("cho_p", "탄수화물 에너지 비율 (%)", "식사력", "mean", False),
    ("fat_p", "지방 에너지 비율 (%)", "식사력", "mean", False),
    ("sug_p", "당류 에너지 비율 (%)", "식사력", "mean", False),
    ("sodium", "나트륨 (mg/일)", "식사력", "mean", False),
    ("fiber", "식이섬유 (g/일)", "식사력", "mean", False),
    ("skip_bf", "아침 결식 (주 2회 이하)", "생활습관", "pct", False),
    ("eatout_d", "외식 하루 1회 이상", "생활습관", "pct", False),
    ("drink", "고위험음주", "생활습관", "pct", False),
    ("smoke", "현재 흡연", "생활습관", "pct", False),
    ("dm_dx", "당뇨 의사진단", "과거력", "pct", False),
    ("ht_dx", "고혈압 의사진단", "과거력", "pct", False),
]

# C 관련성: 성별·나이 보정 로지스틱 회귀
ENERGY_ADJ = {"p_na", "p_fib"}   # 하루 총량 기준이라 많이 먹는 사람일수록 해당하기 쉬워서 에너지 섭취량도 보정해요


OR_COLS = ["key", "OR", "low", "high", "p", "n"]


@st.cache_data(show_spinner="성별·나이를 보정해서 계산하고 있어요")
def assoc_or(d, outcome, keys, energy_keys=()):
    """결과(outcome, 0/1 열)와 요인 하나씩의 관련성: 요인 + 나이(곡선) + 성별 (+에너지)로 가중 로지스틱 회귀.
    신뢰구간은 같은 조사구(psu) 사람끼리 비슷한 점을 반영한 군집 표준오차로 계산해요."""
    use_psu = d["psu"].notna().any()
    rows = []
    for key in keys:
        if key == outcome:
            continue
        need = [outcome, key, "age", "sex", "wt"] + (["energy"] if key in energy_keys else []) + (["psu"] if use_psu else [])
        dd = d.dropna(subset=need)
        y = dd[outcome].astype(float)
        cells = pd.crosstab(y, dd[key])
        if len(dd) < 100 or cells.shape != (2, 2) or cells.values.min() < 10:
            continue
        a = (dd["age"] - 50) / 10
        X = pd.DataFrame({"지표": dd[key].astype(float), "나이": a, "나이²": a ** 2, "여자": (dd["sex"] == 2).astype(float)})
        if key in energy_keys:
            X["에너지"] = dd["energy"] / 1000
        X = sm.add_constant(X)
        model = sm.GLM(y, X, family=sm.families.Binomial(), var_weights=dd["wt"] / dd["wt"].mean())
        with warnings.catch_warnings():   # 가중치+군집 표준오차 조합 경고: 직접 계산한 값과 같은 걸 확인했어요
            warnings.simplefilter("ignore")
            try:
                if use_psu:
                    res = model.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(dd["psu"])[0]})
                else:
                    res = model.fit(cov_type="HC1")
            except Exception:
                continue
        b, se = res.params["지표"], res.bse["지표"]
        rows.append({"key": key, "OR": np.exp(b), "low": np.exp(b - 1.96 * se), "high": np.exp(b + 1.96 * se),
                     "p": res.pvalues["지표"], "n": len(dd)})
    return pd.DataFrame(rows, columns=OR_COLS)


def adjusted_or(df, case, keys):
    """대상 그룹(case) vs 정상 그룹. case='중재 대상'이면 위험군+대사증후군 vs 정상."""
    if case == "중재 대상":
        sub = df.assign(_y=(df["group"] != "정상").astype(float))
    else:
        sub = df[df["group"].isin([case, "정상"])]
        sub = sub.assign(_y=(sub["group"] == case).astype(float))
    return assoc_or(sub, "_y", tuple(keys), tuple(ENERGY_ADJ))


# ================================================================ 진단 후보와 중재 내용
# 원인(E) 후보: 문제(P)마다 데이터로 관련성을 확인할 생활·사회경제 요인 → (이름, 이 원인이 나오면 할 중재, 중재 영역)
E_LIB = {
    "eatout_d": ("잦은 외식 (하루 1회 이상)", "외식·배달 메뉴 고르는 법을 교육하고, 구내식당·지역 음식점과 저염·적정량 메뉴를 함께 만들어요.", "교육 + 다분야 협의"),
    "skip_bf": ("아침 결식", "준비 쉬운 아침 조합을 알려주고, 직장·학교에서 간편 아침을 제공하는 방안을 찾아요.", "교육 + 제공"),
    "drink": ("고위험음주", "절주 목표를 함께 정하고 안주 고르는 법을 교육해요. 조절이 어려우면 보건소 절주 상담을 연계해요.", "상담 + 다분야 협의"),
    "smoke": ("현재 흡연", "보건소 금연클리닉을 연계해요. 금연 후 체중이 늘지 않도록 식사 관리를 함께 안내해요.", "다분야 협의"),
    "inactive": ("유산소 신체활동 부족", "보건소 운동 프로그램·걷기 모임과 연계해서 식사 관리와 같이 진행해요.", "다분야 협의"),
    "no_label": ("영양표시를 읽지 않음", "가공식품 영양표시(나트륨·당류 %영양성분 기준치) 읽기 실습을 교육에 넣어요.", "교육"),
    "p_en_hi": ("에너지 과잉 섭취", "끼니별 적정량(밥 2/3공기, 고기 손바닥 크기 등)을 눈으로 익히는 분량 교육을 해요.", "교육"),
    "low_inc": ("가구 소득 하위 25%", "적은 비용으로 차리는 식단을 안내하고 푸드뱅크·식품바우처 같은 복지 자원과 연계해요.", "제공 + 다분야 협의"),
    "low_edu": ("교육수준 중졸 이하", "글보다 그림·실물 모형·조리 시연 중심의 교육 자료를 써요.", "교육"),
    "rural": ("읍·면 지역 거주", "보건지소·마을회관을 찾아가는 방문 교육으로 해요.", "제공"),
    "alone": ("1인 가구", "편의점·간편식 고르는 법과 소포장 장보기, 한 그릇 요리를 교육해요.", "교육"),
}
SOCIO = {"low_inc", "low_edu", "rural", "alone"}     # 누구에게 먼저 할지 정할 때 쓰는 사회경제 요인
NON_NUTRITION = {"smoke"}                            # 우선순위에는 보이지만 영양진단(P)은 아닌 요인
TRACK = {
    "대사증후군": ("관리 트랙", "2·3차 예방 · 진행과 합병증을 막아요",
               "소그룹·개별 영양상담 주 1회 × 8주, 3개월마다 허리둘레·혈압·혈당 재측정. 약을 먹는 사람은 담당 의사와 목표를 맞춰요."),
    "위험군": ("예방 트랙", "1차 예방 · 대사증후군으로 넘어가지 않게 해요",
            "집단 영양교육 월 1회와 식생활 환경 개선(구내식당·지역 음식점·영양표시). 1년마다 건강검진으로 확인해요."),
}

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
        "기준": "나트륨 만성질환위험감소섭취량 초과 (19~64세 2,300 · 65~74세 2,100 · 75세 이상 1,700mg/일)",
        "E": "국·찌개·김치류와 외식 중심의 식사",
        "E변수": ["eatout_d"], "측정": "24시간 회상법", "목표선": "na_lim",
        "제공": "나트륨 하루 2,300mg 이하 (65~74세 2,100mg, 75세 이상 1,700mg)",
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
        "P": "식이섬유 섭취 적음 (충분섭취량 미만)", "영역": "섭취", "지표": "fiber", "지표명": "식이섬유 섭취량", "단위": "g/일",
        "기준": "식이섬유 충분섭취량 미만 (남 30g·65세 이상 25g, 여 20g). 충분섭취량은 평균필요량이 아니라서 '부족'이라고 판정할 수는 없어요",
        "E": "채소·잡곡·콩류가 적은 식사", "E변수": ["eatout_d", "skip_bf"], "측정": "24시간 회상법",
        "제공": "식이섬유 하루 남 30g(65세 이상 25g), 여 20g 수준",
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
        "P": "알코올 섭취 과다 (고위험음주)", "영역": "섭취", "지표": "drink", "지표명": "고위험음주 비율", "단위": "",
        "기준": "주 2회 이상, 한 번에 남 7잔·여 5잔 이상 (국민건강통계 고위험음주 정의)", "E": "잦은 회식·술자리", "E변수": ["eatout_d"], "측정": "건강 설문",
        "제공": "음주 횟수를 주 1회 이하로, 한 번에 마시는 양도 줄이기",
        "교육": ["빈속에 마시지 않도록 술자리 전에 식사를 해요.",
               "안주는 튀김·찌개 대신 두부·생선구이·채소로 골라요.",
               "술 한 잔에 물 한 잔을 같이 마셔요.",
               "2차는 가지 않는 것을 원칙으로 해요."],
        "상담": ["1~2주: 음주 날짜 기록", "3~4주: 음주 주 2회 이하", "2개월: 음주 주 1회 이하"],
        "협의_키": None, "협의": "음주 조절이 어렵다면 보건소 절주 상담을 연계해요.",
    },
    "p_en_hi": {
        "P": "에너지 섭취 과다", "영역": "섭취", "지표": "en_eer", "지표명": "필요추정량 대비 에너지 섭취", "단위": "%",
        "기준": f"에너지 섭취가 성별·나이별 필요추정량의 {EN_HIGH}% 초과", "E": "한 끼 양이 많거나 간식·야식·음주로 열량이 늘어남",
        "E변수": [], "측정": "24시간 회상법", "목표선": 100,
        "제공": "에너지 섭취를 필요추정량 수준으로 (예: 30~49세 남자 2,500kcal, 여자 1,900kcal)",
        "교육": ["밥은 2/3공기, 고기·생선은 손바닥 크기를 한 끼 기준으로 익혀요.",
               "음료·술·야식의 열량을 알아보고 가장 자주 먹는 것부터 줄여요.",
               "큰 그릇 대신 작은 그릇에 덜어 먹어요."],
        "상담": ["1~2주: 사흘 동안 먹은 것 기록", "3~4주: 야식·간식 주 2회 이하", "2개월: 한 끼 적정량 지키기 주 5일"],
        "협의_키": None, "협의": "체중이 함께 늘고 있다면 비만 관리 프로그램과 연계해요.",
    },
    "inactive": {
        "P": "신체활동 부족", "영역": "행동·환경", "지표": "inactive", "지표명": "유산소 신체활동 미실천 비율", "단위": "",
        "기준": "중강도 주 150분·고강도 주 75분 이상의 유산소 신체활동을 하지 않음 (국민건강통계 유산소 신체활동 실천율 정의)",
        "E": "운동할 시간·장소 부족, 좌식 업무", "E변수": [], "측정": "건강 설문",
        "제공": "중강도 유산소 신체활동 주 150분 이상",
        "교육": ["하루 30분 빠르게 걷기를 주 5일 해요. 10분씩 나눠도 돼요.",
               "식후 10~15분 걷기는 혈당을 낮추는 데 도움이 돼요.",
               "엘리베이터 대신 계단, 가까운 거리는 걸어서 가요."],
        "상담": ["1~2주: 하루 걸음 수 기록", "3~4주: 하루 30분 걷기 주 3회", "2개월: 주 150분"],
        "협의_키": None, "협의": "보건소 운동 프로그램·걷기 동아리와 연계해요.",
    },
}


PRIORITY_KEYS = [k for k in DX if k != "obese"] + ["smoke"]


def name_of(key):
    if key in DX:
        return DX[key]["P"]
    return E_LIB[key][0]


def priority(df, case):
    """문제 후보마다: 대상 그룹 해당 비율, 성별·나이 보정 오즈비, 인구기여위험분율(PAF).
    PAF = 대상 그룹 중 해당 비율 × (오즈비 − 1) / 오즈비 (Miettinen 공식). 오즈비가 확실히 1보다 클 때만 계산해요."""
    w = df["wt"]
    cs, nor = df["group"] == case, df["group"] == "정상"
    keys = [k for k in PRIORITY_KEYS if n_known(df.loc[cs, k]) >= 30]
    ors = adjusted_or(df, case, keys) if keys else pd.DataFrame(columns=OR_COLS)
    rows = []
    for r_ in ors.itertuples():
        x = df[r_.key]
        pc, pn = wmean(x[cs], w[cs]), wmean(x[nor], w[nor])
        sig = r_.low > 1
        rows.append({"key": r_.key, "이름": name_of(r_.key), "대상 해당": pc, "정상 해당": pn, "OR": r_.OR, "low": r_.low,
                     "high": r_.high, "PAF": pc * (r_.OR - 1) / r_.OR if sig else 0.0, "근거": sig,
                     "영양": r_.key not in NON_NUTRITION})
    return pd.DataFrame(rows).sort_values("PAF", ascending=False) if rows else pd.DataFrame()


def pick_auto(pr, df, case, k=3):
    """공통 처방(비만)은 대상 그룹 20% 이상이면 넣고, 나머지는 영양 문제 중 근거가 있는 것을 PAF 순으로 k개."""
    cs = df["group"] == case
    common = ["obese"] if wmean(df.loc[cs, "obese"], df.loc[cs, "wt"]) >= 0.2 else []
    if pr.empty:
        return common, []
    top = pr[pr["근거"] & pr["영양"]]["key"].head(k).tolist()
    return common, top


def causes(df, case, key):
    """문제(P)에 해당하는 것과 관련된 생활·사회경제 요인(E)을 대상 그룹 안에서 찾아요 (성별·나이 보정)."""
    sub = df[df["group"] == case]
    keys = [k for k in E_LIB if k != key and n_known(sub[k]) >= 30]
    if n_known(sub[key]) < 100 or not keys:
        return pd.DataFrame(columns=OR_COLS)
    res = assoc_or(sub, key, tuple(keys))
    return res.sort_values("OR", ascending=False)


def segments(df, case, key):
    """대상 그룹 안에서 성별 × 나이대별 해당 비율."""
    sub = df[(df["group"] == case) & df[key].notna()]
    band = pd.cut(sub["age"], [18, 39, 64, 200], labels=["19~39세", "40~64세", "65세 이상"])
    out = []
    for sx, sx_name in [(1, "남자"), (2, "여자")]:
        for b_ in ["19~39세", "40~64세", "65세 이상"]:
            m = (sub["sex"] == sx) & (band == b_)
            n = int(m.sum())
            out.append({"성별": sx_name, "나이": b_, "해당": wmean(sub.loc[m, key], sub.loc[m, "wt"]) if n >= 30 else np.nan, "n": n,
                        "해당자": (sub.loc[m, key] * sub.loc[m, "wt"]).sum()})
    res = pd.DataFrame(out)
    res["해당자"] = res["해당자"] / res["해당자"].sum() if res["해당자"].sum() > 0 else np.nan
    return res


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
    DX["drink"].update({"P": "음주 (양·빈도는 상담에서 확인)", "기준": "음주 여부 (빈도·양 정보 없음)", "지표명": "음주 비율"})
elif meta.get("drink_label") == "주 2회 이상 음주":
    DX["drink"].update({"P": "잦은 음주 (양은 상담에서 확인)", "기준": "주 2회 이상 음주 (한 번 음주량 정보 없음)", "지표명": "잦은 음주 비율"})
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

t1, t2, t3, t4 = st.tabs(["① 영양판정", "② 영양진단", "③ 영양중재", "④ 평가계획"])

# ---------------------------------------------------------------- ① 영양판정
with t1:
    st.subheader("이 집단은 지금 어떤 상태인가")
    st.write("영양판정은 세 단계로 봐요. **A** 건강 결과로 누가 얼마나 문제인지 보고, **B** 식사·생활을 섭취기준과 비교하고, "
             "**C** 성별·나이를 맞춘 뒤에도 대사증후군과 관련 있는 요인을 찾아요. B와 C는 ② 영양진단에서 문제를 고르는 근거가 돼요.")

    # ---------- A 건강 결과
    st.markdown("#### A. 건강 결과 · 누가, 얼마나")
    shares = {g_: wmean((df["group"] == g_).astype(float), w) for g_ in GROUPS}
    counts = df["group"].value_counts()
    cols = st.columns(4)
    cols[0].metric("분석 인원 (성인)", f"{len(df):,}명")
    for c_, g_, lab in zip(cols[1:], GROUPS, ["기준 0개", "기준 1~2개 · 예방 대상", "기준 3개 이상 · 관리 대상"]):
        c_.metric(f"{g_} ({counts.get(g_, 0):,}명)", f"{shares[g_]:.1%}", help=lab)
    st.caption("대사증후군 판정 기준 5개 중 몇 개에 해당하는지로 나눴어요. 0개 정상 · 1~2개 위험군(예방 대상) · 3개 이상 대사증후군(관리 대상). "
               "여기서 '정상'은 5개 기준에 해당하지 않는다는 뜻이지 건강하다는 뜻은 아니에요.")

    a1, a2 = st.columns(2)
    with a1:
        fig = go.Figure(go.Bar(x=[shares[g_] for g_ in GROUPS], y=GROUPS, orientation="h",
                               marker_color=[COLORS[g_] for g_ in GROUPS],
                               text=[f"{shares[g_]:.1%}" for g_ in GROUPS], textposition="outside"))
        fig.update_layout(title="그룹 분포", height=230, margin=dict(t=40, b=10, l=10, r=40), xaxis_tickformat=".0%",
                          xaxis_range=[0, max(shares.values()) * 1.25], yaxis_autorange="reversed")
        st.plotly_chart(fig, width="stretch")
    with a2:
        crit = pd.DataFrame([{"항목": name, "기준": rule, "해당": wmean(df[col].astype(float), w)} for col, name, rule in CRITERIA])
        crit = crit.sort_values("해당")
        fig = go.Figure(go.Bar(x=crit["해당"], y=crit["항목"], orientation="h", marker_color="#8C6BB1",
                               text=[f"{v:.1%}" for v in crit["해당"]], textposition="outside",
                               customdata=crit["기준"], hovertemplate="%{y}: %{x:.1%}<br>기준: %{customdata}<extra></extra>"))
        fig.update_layout(title="판정 기준별 해당 비율 (성인 전체)", height=230, margin=dict(t=40, b=10, l=10, r=40),
                          xaxis_tickformat=".0%", xaxis_range=[0, crit["해당"].max() * 1.3])
        st.plotly_chart(fig, width="stretch")

    with st.expander("혈당 단계 자세히 보기 (국민건강통계 당뇨 정의)"):
        stage = {s_: wmean((df["glu_stage"] == s_).astype(float), w) for s_ in ["정상", "당뇨 전단계", "당뇨"]}
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("혈당 정상", f"{stage['정상']:.1%}")
        c2.metric("당뇨 전단계", f"{stage['당뇨 전단계']:.1%}", help="공복혈당 100~125 또는 당화혈색소 5.7~6.4%")
        c3.metric("당뇨", f"{stage['당뇨']:.1%}", help="공복혈당 126 이상, 당화혈색소 6.5% 이상, 의사진단, 당뇨약·인슐린 중 하나")
        if df["dm_dx"].notna().any():
            c4.metric("수치는 높은데 진단 없음", f"{int(df['undx_dm'].sum()):,}명", help="③ 영양중재의 다분야 협의(진료 연계) 대상이에요")
        if df["hba1c"].isna().all():
            st.caption("당화혈색소·진단·약 정보가 없으면 공복혈당만으로 판정해서, 공복혈당은 정상인 당뇨인을 놓쳐요.")

    # ---------- B 식사·생활 판정
    st.markdown("#### B. 식사·생활 판정 · 섭취기준과 비교")
    rows = []
    for col, name, area, rule in JUDGE:
        x = df[col]
        if n_known(x) < 30:
            continue
        row = {"영역": area, "항목": name, "판정 기준": rule, "성인 전체": wmean(x, w)}
        for g_ in GROUPS:
            m = df["group"] == g_
            row[g_] = np.nan if n_known(x[m]) < 5 else wmean(x[m], w[m])
        rows.append(row)
    if rows:
        judge = pd.DataFrame(rows)
        if meta.get("drink_label") != "고위험음주":
            rule = "주 2회 이상 (한 번 음주량 정보가 없어 빈도로만 판정)" if meta.get("drink_label") == "주 2회 이상 음주" else "음주 여부 (빈도·양 정보 없음)"
            judge.loc[judge["항목"] == "고위험음주", ["항목", "판정 기준"]] = [meta.get("drink_label", "음주"), rule]
        for c_ in ["성인 전체"] + GROUPS:
            judge[c_] = judge[c_] * 100
        pct = st.column_config.NumberColumn(format="%.1f%%")
        st.dataframe(judge, width="stretch", hide_index=True, height=35 * (len(judge) + 1) + 3,
                     column_config={"판정 기준": st.column_config.TextColumn(width="large"),
                                    "성인 전체": pct, "정상": pct, "위험군": pct, "대사증후군": pct})
        st.caption(f"숫자는 기준에 해당하는 사람의 비율이에요. 기준은 국건영 제9기가 쓰는 {KDRI_NAME}이고, 성별·나이별 값을 사람마다 따로 적용했어요. "
                   "24시간 회상법은 하루치 기록이라 '평소에 기준을 넘는 사람'보다 높게 나올 수 있어요. "
                   "식이섬유 충분섭취량은 평균필요량이 아니라서, 미만이어도 '부족'이라고 판정하지는 않아요.")
        if not has_diet:
            st.info("식사 자료가 없어 식사 항목은 빠졌어요. 국건영 영양조사 자료를 올리면 나타나요.")
    else:
        st.info("식사·생활 자료가 없어 이 단계를 건너뛰어요.")

    # ---------- C 관련성
    st.markdown("#### C. 관련성 · 성별·나이를 맞춰도 남는 차이")
    st.write("B에서 그룹 간 차이가 나도 나이 차이 때문일 수 있어요. 대사증후군 그룹이 정상 그룹보다 평균 나이가 훨씬 많거든요. "
             "그래서 같은 성별·나이끼리 비교했을 때도 차이가 남는지 봐요.")
    age_gap = {g_: wmean(df.loc[df["group"] == g_, "age"], w[df["group"] == g_]) for g_ in GROUPS}
    st.caption("그룹별 평균 나이: " + " · ".join(f"{g_} {age_gap[g_]:.1f}세" for g_ in GROUPS))
    case = st.radio("어느 그룹을 정상 그룹과 비교할까", ["대사증후군", "위험군"], horizontal=True,
                    format_func=lambda g_: f"{g_} vs 정상" + (" (관리 대상)" if g_ == "대사증후군" else " (예방 대상)"))
    keys = [col for col, *_ in JUDGE if n_known(df[col]) >= 30]
    ors = adjusted_or(df, case, keys) if keys else pd.DataFrame()
    st.session_state["or_table"] = ors
    if ors.empty:
        st.info("보정 분석을 하기에 자료가 부족해요.")
    else:
        names = {col: name for col, name, *_ in JUDGE}
        if meta.get("drink_label") != "고위험음주":
            names["drink"] = meta.get("drink_label", "음주")
        ors = ors.assign(이름=ors["key"].map(names), 확실=(ors["low"] > 1) | (ors["high"] < 1)).sort_values("OR")
        color = np.where(~ors["확실"], "#A0A0A0", np.where(ors["OR"] > 1, "#C8553D", "#4C9F70"))
        fig = go.Figure(go.Scatter(
            x=ors["OR"], y=ors["이름"], mode="markers", marker=dict(size=11, color=color),
            error_x=dict(type="data", symmetric=False, array=ors["high"] - ors["OR"], arrayminus=ors["OR"] - ors["low"], color="#888"),
            hovertemplate="%{y}<br>오즈비 %{x:.2f}<extra></extra>"))
        fig.add_vline(x=1, line_dash="dash", line_color="#666")
        fig.update_layout(height=60 + 34 * len(ors), margin=dict(t=10, b=40, l=10, r=10), xaxis_type="log",
                          xaxis_title=f"오즈비 (95% 신뢰구간) · 1보다 크면 {case} 그룹과 관련 높음")
        st.plotly_chart(fig, width="stretch")
        st.caption("빨간 점: 해당하는 사람이 같은 성별·나이의 비해당자보다 " + case + " 그룹일 가능성이 높아요. "
                   "초록 점: 낮아요. 회색 점: 신뢰구간이 1을 걸쳐서 차이가 있다고 말하기 어려워요.")
        with st.expander("숫자로 보기 · 오즈비 읽는 법"):
            tbl = ors[["이름", "OR", "low", "high", "p", "n"]].rename(columns={"이름": "항목", "OR": "오즈비", "low": "95% 하한", "high": "95% 상한", "p": "p값", "n": "분석 인원"})
            st.dataframe(tbl.sort_values("오즈비", ascending=False), width="stretch", hide_index=True, height=35 * (len(tbl) + 1) + 3,
                         column_config={c_: st.column_config.NumberColumn(format="%.2f") for c_ in ["오즈비", "95% 하한", "95% 상한"]}
                         | {"p값": st.column_config.NumberColumn(format="%.3f")})
            st.markdown(md(
                "- **오즈비 1.5**: 성별·나이가 같을 때, 그 항목에 해당하는 사람은 해당하지 않는 사람보다 "
                f"{case} 그룹일 오즈가 1.5배라는 뜻이에요.\n"
                "- **95% 신뢰구간**이 1을 포함하지 않으면 우연으로 보기 어려운 차이예요.\n"
                "- 나트륨과 식이섬유는 하루 총량 기준이라 많이 먹는 사람일수록 해당하기 쉬워서, 에너지 섭취량도 함께 맞췄어요.\n"
                "- 가중치를 반영했고, 같은 조사구 사람끼리 비슷한 점을 고려해 신뢰구간을 넓게 잡았어요 (층 정보까지 쓰는 정식 복합표본 분석보다 조금 보수적).\n"
                "- 비만(BMI)은 판정 기준인 허리둘레와 거의 같이 움직여서 여기서 뺐어요."))
        st.warning("같은 날 조사한 단면 자료라 원인이 아니라 **관련성**이에요. 이미 진단받고 식사를 바꾼 사람이 섞이면 방향이 거꾸로 나올 수 있어요 "
                   "(예: 당류를 줄인 당뇨 환자). 위의 '진단받은 사람 빼고 보기'를 켜서 결과가 유지되는지 확인해 보세요.")

    with st.expander("상세 표 · 그룹별 평균 비교"):
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
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True,
                     column_config={"효과크기": st.column_config.NumberColumn(format="%.2f",
                                    help="(대사증후군 평균 − 정상 평균) ÷ 표준편차. 단위가 다른 지표의 차이 크기를 같은 잣대로 보는 값이에요. 0.2 작음 · 0.5 중간 · 0.8 큼")})
        st.caption("판정 기준이 붙은 지표는 그룹을 나눈 기준이라 차이가 큰 게 당연해요. 이 표는 나이 보정 전 단순 비교라서 결론은 C를 기준으로 봐요. "
                   "식사 지표의 효과크기는 하루치 기록의 날마다 변동 때문에 실제보다 작게 나오는 경향이 있어요.")

# ---------------------------------------------------------------- ② 영양진단
with t2:
    st.subheader("무엇이 문제이고, 왜 그런가")
    st.write("문제 후보마다 **얼마나 흔한지**와 **성별·나이를 맞춰도 관련이 남는지**를 계산하고, 둘을 합친 "
             "**인구기여위험분율(PAF)** 순서로 우선순위를 정해요. 그다음 문제마다 관련된 생활·사회경제 요인을 찾아 원인(E)으로 써요. "
             "문제와 원인이 미리 정해진 게 아니라, 데이터를 바꾸면 결과도 바뀌어요.")
    case = st.radio("어느 대상을 위한 영양진단인가", ["대사증후군", "위험군"], horizontal=True, key="track",
                    format_func=lambda g_: f"{TRACK[g_][0]} · {g_} (vs 정상)")
    st.caption(f"{TRACK[case][0]}: {TRACK[case][1]}")
    cs = df["group"] == case
    pr = priority(df, case)
    common_keys, top_keys = pick_auto(pr, df, case)

    st.markdown("#### 1. 우선순위 · 흔하고, 관련이 큰 문제부터")
    if pr.empty:
        st.info("우선순위를 계산할 자료가 부족해요. 식사·생활 변수가 있는 국건영 자료를 올려 주세요.")
    else:
        kind = np.where(~pr["영양"], "영양 외 요인", np.where(pr["key"].isin(top_keys), "선택된 영양진단",
                        np.where(pr["근거"], "관련 있음", "관련 불확실")))
        palette = {"선택된 영양진단": "#C8553D", "관련 있음": "#E8A33D", "관련 불확실": "#A0A0A0", "영양 외 요인": "#5B7DB1"}
        fig = go.Figure()
        for k_, col_ in palette.items():
            m = kind == k_
            if not m.any():
                continue
            sub = pr[m]
            fig.add_trace(go.Scatter(
                x=sub["대상 해당"] * 100, y=sub["OR"], mode="markers+text", name=k_, text=sub["이름"], textposition="top center",
                marker=dict(size=12 + 120 * sub["PAF"], color=col_, opacity=0.85, line=dict(width=1, color="white")),
                error_y=dict(type="data", symmetric=False, array=sub["high"] - sub["OR"], arrayminus=sub["OR"] - sub["low"], color="#BBB", thickness=1),
                customdata=np.stack([sub["low"], sub["high"], sub["PAF"] * 100], axis=-1),
                hovertemplate="%{text}<br>해당 %{x:.1f}%<br>오즈비 %{y:.2f} (%{customdata[0]:.2f}~%{customdata[1]:.2f})<br>PAF %{customdata[2]:.1f}%<extra></extra>"))
        fig.add_hline(y=1, line_dash="dash", line_color="#666")
        fig.update_layout(height=470, margin=dict(t=10, b=50, l=10, r=10), yaxis_type="log", legend=dict(orientation="h", y=-0.15),
                          xaxis_title=f"{case} 그룹에서 해당하는 비율 (%) → 흔할수록 오른쪽",
                          yaxis_title="성별·나이 보정 오즈비 → 관련 클수록 위")
        st.plotly_chart(fig, width="stretch")
        st.caption("원 크기 = 인구기여위험분율(PAF). 오른쪽 위에 크게 있을수록 우선이에요. 세로 막대는 95% 신뢰구간이고, 점선(1)에 걸치면 관련이 불확실해요.")

        tbl = pr.assign(**{"대상 해당": pr["대상 해당"] * 100, "정상 해당": pr["정상 해당"] * 100, "PAF": pr["PAF"] * 100,
                           "오즈비 (95% CI)": [f"{a:.2f} ({b:.2f}–{c:.2f})" for a, b, c in zip(pr["OR"], pr["low"], pr["high"])],
                           "판단": kind})
        st.dataframe(tbl[["이름", "대상 해당", "정상 해당", "오즈비 (95% CI)", "PAF", "판단"]], width="stretch", hide_index=True,
                     height=35 * (len(tbl) + 1) + 3,
                     column_config={"이름": "문제 후보", "대상 해당": st.column_config.NumberColumn(f"{case} 해당", format="%.1f%%"),
                                    "정상 해당": st.column_config.NumberColumn("정상 해당", format="%.1f%%"),
                                    "PAF": st.column_config.NumberColumn("PAF", format="%.1f%%",
                                        help="이 문제와 관련된 몫. 이 문제가 원인이라고 가정하면, 없앴을 때 대사증후군(또는 위험군)이 이만큼 줄어든다는 뜻이에요.")})
        with st.expander("PAF(인구기여위험분율)가 뭐야?"):
            st.markdown(md(
                "- 오즈비는 **한 사람 기준**으로 '해당하면 몇 배 관련 있나'예요. 그런데 아주 드문 문제는 오즈비가 커도 집단 전체에 미치는 영향은 작아요.\n"
                "- PAF는 **흔한 정도와 관련 크기를 곱한 값**이라서, '집단에서 이 문제를 해결하면 얼마나 줄어드나'를 보여줘요. "
                "지역사회영양에서 문제의 크기와 심각도로 우선순위를 정하는 것과 같은 원리예요.\n"
                "- 계산: PAF = (대상 그룹 중 해당 비율) × (오즈비 − 1) ÷ 오즈비. 보정 오즈비의 신뢰구간이 1보다 확실히 클 때만 계산해요.\n"
                "- 주의: 대사증후군처럼 흔한 결과에서는 오즈비가 실제 위험비보다 커서 PAF가 다소 크게 나와요. 그리고 인과관계를 가정한 값이에요."))
        st.caption("고르는 규칙: 과체중·비만은 판정 기준(허리둘레)과 겹쳐서 우선순위 계산에서 빼고, 대상 그룹 20% 이상이면 공통 처방으로 넣어요. "
                   "나머지는 영양 문제 중 관련이 확실한 것을 PAF 순서로 3개까지 골라요. 흡연은 영양진단이 아니라서 다분야 협의로 보내요.")

    st.markdown("#### 2. 집단 PES · 원인(E)도 데이터에서")
    options = ["obese"] + (pr[pr["영양"]]["key"].tolist() if not pr.empty else [])
    chosen = st.multiselect("영양진단으로 쓸 문제 (기본은 자동 선택)", options=options, default=common_keys + top_keys,
                            format_func=lambda k: DX[k]["P"], key=f"dx_pick_{case}_{uploaded}_{len(df)}")
    prow = pr.set_index("key") if not pr.empty else pd.DataFrame()
    for key in chosen:
        d = DX[key]
        p_c = wmean(df.loc[cs, key], w[cs])
        p_n = wmean(df.loc[df["group"] == "정상", key], w[df["group"] == "정상"])
        s_ = f"{case} 그룹의 {d['지표명'] if d['단위'] == '' else d['P'] + ' 해당 비율'} {p_c:.1%} (정상 그룹 {p_n:.1%})"
        if key in prow.index and prow.loc[key, "근거"]:
            r_ = prow.loc[key]
            s_ += f", 성별·나이 보정 오즈비 {r_['OR']:.2f} (95% CI {r_['low']:.2f}–{r_['high']:.2f}), 인구기여위험분율 {r_['PAF']:.0%}"
        ec = causes(df, case, key)
        sig = ec[ec["low"] > 1]
        e_data = [f"{E_LIB[k_][0]} (오즈비 {o:.2f}, {lo:.2f}–{hi:.2f})" for k_, o, lo, hi in zip(sig["key"], sig["OR"], sig["low"], sig["high"])][:3]
        with st.container(border=True):
            st.markdown(f"**{d['P']}** · {d['영역']} 영역" + (" · 공통 처방" if key in common_keys else ""))
            st.markdown(f"- **P (문제)** {d['P']}\n"
                        + (f"- **E (원인, 데이터)** {', '.join(e_data)}\n    - {case} 그룹 안에서 이 문제에 해당하는 것과 관련된 요인이에요 (성별·나이 보정)\n"
                           if e_data else "- **E (원인, 데이터)** 뚜렷하게 관련된 생활·사회경제 요인이 나오지 않았어요\n")
                        + f"- **E (원인, 전공 지식)** {d['E']} · 24시간 회상 음식 자료가 없어 상담에서 확인해요\n"
                        + f"- **S (근거)** {s_}")
            st.caption(md(f"판정 기준: {d['기준']}"))
            e_sent = ", ".join(E_LIB[k_][0] for k_ in sig["key"][:3]) if e_data else d["E"]
            st.markdown(md(f"> {case} 그룹의 영양 문제는 '**{d['P']}**'이다. 관련 요인은 '**{e_sent}**'이고, 근거는 '**{s_}**'이다."))
            with st.expander("원인 후보 전체 보기"):
                if ec.empty:
                    st.caption("계산할 수 있는 원인 후보가 없어요.")
                else:
                    et = ec.assign(이름=ec["key"].map(lambda k_: E_LIB[k_][0]), 관련=np.where(ec["low"] > 1, "있음", np.where(ec["high"] < 1, "반대 방향", "불확실")))
                    st.dataframe(et[["이름", "OR", "low", "high", "관련", "n"]], width="stretch", hide_index=True,
                                 column_config={"이름": "원인 후보", "OR": st.column_config.NumberColumn("오즈비", format="%.2f"),
                                                "low": st.column_config.NumberColumn("95% 하한", format="%.2f"),
                                                "high": st.column_config.NumberColumn("95% 상한", format="%.2f"), "n": "분석 인원"})
    st.session_state["chosen"] = chosen

# ---------------------------------------------------------------- ③ 영양중재
with t3:
    st.subheader("누구에게, 무엇부터, 어떻게")
    st.write(f"② 영양진단의 결과로 중재를 짜요. **누구에게**는 {case} 그룹 안의 성별·나이대별 해당 비율로, **무엇부터**는 데이터가 찾은 원인(E)으로 정해요. "
             "영양중재 4영역(식품·영양소 제공, 영양교육, 영양상담, 다분야 협의)이 틀이에요.")
    st.info(f"**{TRACK[case][0]}** ({TRACK[case][1]}) · {TRACK[case][2]}")
    mets = df["group"] == "대사증후군"
    if case == "대사증후군" and mets.sum() >= 5:
        lean = wmean((df.loc[mets, "bmi"] < 25).astype(float), w[mets])
        st.caption(f"대사증후군 그룹의 {lean:.1%}는 BMI 25 미만이에요. 이 사람들에게는 체중 감량보다 식사의 질이 중심이에요.")
    if not chosen:
        st.info("② 영양진단 탭에서 문제를 하나 이상 골라 주세요.")
    for key in chosen:
        d = DX[key]
        with st.container(border=True):
            st.markdown(f"### {d['P']}")
            seg = segments(df, case, key)
            ok = seg.dropna(subset=["해당"])
            a, b = st.columns([1, 1])
            with a:
                st.markdown("**누구에게 먼저** · 성별 × 나이대별 해당 비율")
                if ok.empty:
                    st.caption("하위 집단별로 나눌 만큼 자료가 없어요.")
                else:
                    piv = seg.pivot(index="성별", columns="나이", values="해당").reindex(index=["남자", "여자"], columns=["19~39세", "40~64세", "65세 이상"])
                    nn = seg.pivot(index="성별", columns="나이", values="n").reindex(index=["남자", "여자"], columns=["19~39세", "40~64세", "65세 이상"])
                    txt = [[("표본 부족" if pd.isna(v) else f"{v:.0%}") + f"<br>(n={int(n)})" for v, n in zip(rv, rn)] for rv, rn in zip(piv.values, nn.values)]
                    fig = go.Figure(go.Heatmap(z=piv.values * 100, x=piv.columns, y=piv.index, text=txt, texttemplate="%{text}",
                                               colorscale="Reds", showscale=False, hoverinfo="skip"))
                    fig.update_layout(height=200, margin=dict(t=10, b=10, l=10, r=10), yaxis_autorange="reversed")
                    st.plotly_chart(fig, width="stretch", key=f"seg_{key}")
                    big = ok[ok["n"] >= 50]
                    if not big.empty:
                        hi_ = big.sort_values("해당", ascending=False).iloc[0]
                        many = big.sort_values("해당자", ascending=False).iloc[0]
                        st.markdown(f"→ 고위험 접근 (비율이 가장 높은 집단): **{hi_['나이']} {hi_['성별']}** {hi_['해당']:.0%}  \n"
                                    f"→ 인구집단 접근 (해당자가 가장 많은 집단): **{many['나이']} {many['성별']}**, {case} 그룹 해당자의 {many['해당자']:.0%}")
                    st.caption("표본 50명 미만인 칸은 우선 대상을 고를 때 빼요.")
            ec = causes(df, case, key)
            sig = ec[ec["low"] > 1]
            with b:
                st.markdown("**무엇부터** · 데이터가 찾은 원인에 맞춘 중재")
                if sig.empty:
                    st.caption("뚜렷한 원인 요인이 없어 아래 기본 중재로 시작해요.")
                for k_, o in zip(sig["key"], sig["OR"]):
                    nm, act, area = E_LIB[k_]
                    tag = "대상 선정" if k_ in SOCIO else area
                    st.markdown(f"- **{nm}** (오즈비 {o:.2f}) → {act} `{tag}`")
            c, e = st.columns(2)
            with c:
                st.markdown("**식품·영양소 제공** · 목표")
                st.write(d["제공"])
                st.markdown("**영양교육** · 구체적 실천 방안")
                st.markdown("\n".join(f"- {md(t)}" for t in d["교육"]))
            with e:
                st.markdown("**영양상담** · 작게 시작하는 단계별 목표")
                st.markdown("\n".join(f"- {md(t)}" for t in d["상담"]))
                st.markdown("**영양관리를 위한 다분야 협의** · 연계")
                ck = d.get("협의_키")
                if ck and df[ck].notna().any() and df[{"undx_ht": "ht_dx", "undx_dm": "dm_dx", "undx_dl": "dl_dx"}[ck]].notna().any():
                    st.write(f"{d['협의']} {case} 그룹 해당 {int((cs & df[ck]).sum()):,}명 (표본 기준)")
                else:
                    st.write(d["협의"])
    if not pr.empty and "smoke" in set(pr.loc[pr["근거"], "key"]):
        r_ = pr.set_index("key").loc["smoke"]
        st.warning(f"영양 외 요인 · 현재 흡연도 {case} 그룹과 관련이 있어요 (오즈비 {r_['OR']:.2f}). 보건소 금연클리닉 연계를 다분야 협의에 넣어요.")
    st.caption("교육·상담 문구는 식품영양학과 검토 초안이에요. 섭취기준·진료지침 원문으로 목표값을 확인해야 해요.")

# ---------------------------------------------------------------- ④ 평가계획
with t4:
    st.subheader("효과를 무엇으로, 어떻게 확인할까")
    st.write("국건영은 매년 다른 사람을 조사해서 같은 사람의 변화를 따라갈 수 없어요. 그래서 개인 모니터링 대신 **평가계획**을 세워요. "
             "② 영양진단의 근거(S)가 그대로 평가 지표가 되고, 지금 값이 기준선이에요.")
    cut = st.slider("목표: 해당 비율을 지금보다 몇 % 줄일까 (상대 감소)", 5, 50, 20, 5)
    plan, eff = [], []
    share = wmean(cs.astype(float), w)
    for key in chosen:
        d = DX[key]
        now = wmean(df.loc[cs, key], w[cs])
        plan.append({"영양진단": d["P"], "지표": f"{d['지표명']} 기준 해당 비율" if d["단위"] else d["지표명"],
                     "기준선 (지금)": now * 100, "목표": now * (1 - cut / 100) * 100, "측정 방법": d["측정"], "재평가": "1년 (국건영 조사 주기)"})
        if key in prow.index and prow.loc[key, "PAF"] > 0:
            paf = prow.loc[key, "PAF"]
            eff.append({"영양진단": d["P"], "PAF": paf, "예상 감소": paf * cut / 100})
    if plan:
        st.dataframe(pd.DataFrame(plan), width="stretch", hide_index=True,
                     column_config={"기준선 (지금)": st.column_config.NumberColumn(format="%.1f%%"),
                                    "목표": st.column_config.NumberColumn(format="%.1f%%")})
    st.markdown(f"**기대 효과** · 해당 비율을 {cut}% 줄이면 {case} 그룹은 얼마나 줄까")
    if eff:
        et = pd.DataFrame(eff)
        both = 1 - np.prod(1 - et["예상 감소"])
        et = pd.concat([et, pd.DataFrame([{"영양진단": "모두 함께", "PAF": np.nan, "예상 감소": both}])], ignore_index=True)
        et["성인 100명당"] = share * et["예상 감소"] * 100
        et["PAF"] = [("-" if pd.isna(v) else f"{v:.1%}") for v in et["PAF"]]
        st.dataframe(et.assign(**{"예상 감소": et["예상 감소"] * 100}), width="stretch", hide_index=True,
                     column_config={
                                    "예상 감소": st.column_config.NumberColumn(f"{case} 그룹 상대 감소", format="%.1f%%"),
                                    "성인 100명당": st.column_config.NumberColumn(f"성인 100명당 줄어드는 {case} (명)", format="%.2f")})
        st.caption(f"계산: 예상 감소 = PAF × 줄이는 비율. 지금 {case} 그룹은 성인의 {share:.1%}예요. '모두 함께'는 문제들이 서로 독립이라고 가정해 합쳤어요. "
                   "관련성을 인과로 가정한 최대 추정치라서, 실제 효과는 이보다 작을 수 있어요.")
    else:
        st.caption("PAF를 계산할 수 있는 영양진단이 없어요 (과체중·비만은 판정 기준과 겹쳐서 계산에서 빠져요).")
    st.markdown("**한계**")
    st.markdown("- 같은 날 잰 단면 자료라, 식사와 대사 위험이 함께 나타나는 관계(연관성)만 볼 수 있어요.\n"
                "- 24시간 회상법은 하루치라 한 사람의 평소 식사를 대표하지 못해요. 집단 평균·비율로만 해석해요.\n"
                "- 이미 진단받은 사람은 식사를 바꿨을 수 있어요. 위의 '진단받은 사람 빼고 보기'로 비교해 보세요.\n"
                "- 연도별 비교는 '같은 사람이 좋아졌다'가 아니라 '집단 전체 수준이 변했다'는 뜻이에요.")
