"""VibeData AI 공용 이상 탐지 엔진 (Python)

데모 사이트의 engine.js를 Python으로 옮긴 것입니다. 함수 이름과 결과 구조는 원본과 같게 두었습니다.
화면(Streamlit) 코드는 없고, 계산만 합니다. 각 팀원 페이지에서 불러서 씁니다.

가장 간단한 사용법 (Streamlit 페이지 안에서):

    import streamlit as st
    from engine import load_csv, run_engine, events_table

    file = st.file_uploader("CSV 올리기", type="csv")
    if file is not None:
        table = load_csv(file)          # CSV 읽기 (UTF-8 / CP949 자동)
        result = run_engine(table)      # 이상 탐지
        st.dataframe(events_table(result))

엔진이 하는 일 (4가지 방법을 함께 보고 판정):
  1. 튀는 값 탐지: 주변 값의 중앙값과 비교해 혼자 튀는 값 (robust Z)
  2. 관리도: 값이 기준선에서 서서히 멀어지는지 (EWMA)
  3. 변동성 감시: 값이 평소보다 크게 흔들리는지 (이동 표준편차)
  4. 다변량: 변수끼리의 관계가 평소와 다른지 (Mahalanobis 거리)
  → 2개 이상이 동의하거나 한 방법이 아주 강하게 반응하면 '이상'으로 판정하고,
    이어진 이상 행들을 묶어 '이벤트'(튀는 값 / 수준 이동 / 서서히 변화 / 변동성 증가 / 관계 이상)로 만듭니다.
  기준 구간: 데이터 앞쪽 30%(최소 30행)를 '정상'으로 보고 기준선을 만듭니다.
"""
from __future__ import annotations

import csv
import io
import math
import re
from datetime import datetime, timedelta

import numpy as np

NaN = float("nan")
_EPOCH = datetime(1970, 1, 1)


# ---------------------------------------------------------------- 작은 수학 도우미
def is_num(v) -> bool:
    """유한한 숫자인지 (NaN, 무한대, None이면 False)"""
    return isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool) and math.isfinite(v)


def js_round(x: float) -> float:
    """JavaScript Math.round와 같은 반올림 (0.5는 위로). 파이썬 round는 짝수 쪽으로 반올림해서 다름."""
    return int(math.floor(x + 0.5))


def round_to(v, d: int):
    """소수 d자리 반올림"""
    if not is_num(v):
        return v
    f = 10 ** d
    r = math.floor(v * f + 0.5) / f
    return int(r) if float(r).is_integer() else r   # 180.0 대신 180 (JS와 같게)


def _finite(a):
    return [v for v in a if is_num(v)]


def median(a) -> float:
    b = sorted(_finite(a))
    n = len(b)
    if not n:
        return NaN
    return b[(n - 1) // 2] if n % 2 else (b[n // 2 - 1] + b[n // 2]) / 2


def quantile(a, q: float) -> float:
    b = sorted(_finite(a))
    if not b:
        return NaN
    pos = (len(b) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return b[lo] + (b[hi] - b[lo]) * (pos - lo)


def mean(a) -> float:
    b = _finite(a)
    return sum(b) / len(b) if b else NaN


def std(a) -> float:
    """표본 표준편차 (n-1로 나눔)"""
    b = _finite(a)
    if len(b) <= 1:
        return 0.0
    m = sum(b) / len(b)
    return math.sqrt(sum((v - m) ** 2 for v in b) / (len(b) - 1))


def mad(a, m=None) -> float:
    """중앙값 절대 편차: 값과 중앙값의 차이(절댓값)의 중앙값"""
    if m is None:
        m = median(a)
    return median([abs(v - m) for v in a if is_num(v)])


def robust(a) -> dict:
    """튀는 값에 흔들리지 않는 중심(m)과 퍼짐(s). s = 1.4826 × MAD"""
    m = median(a)
    s = 1.4826 * mad(a, m)
    if not s > 0:
        s = std(a)
    if not s > 0:
        s = abs(m) * 1e-3 if is_num(m) and abs(m) * 1e-3 else 1e-6
    return {"m": m, "s": s}


def interpolate(x):
    """빈 값(NaN)을 앞뒤 값으로 직선 보간. 맨 앞·맨 뒤는 가장 가까운 값으로 채움"""
    n = len(x)
    y = list(x)
    last = -1
    for i in range(n):
        if is_num(y[i]):
            if last == -1 and i > 0:
                for j in range(i):
                    y[j] = y[i]
            elif last >= 0 and i - last > 1:
                for j in range(last + 1, i):
                    y[j] = y[last] + (y[i] - y[last]) * (j - last) / (i - last)
            last = i
    if last == -1:
        return [0.0] * n
    for j in range(last + 1, n):
        y[j] = y[last]
    return y


def rolling_median_centered(x, w: int):
    """가운데 정렬 이동 중앙값 (창 크기 w, 양 끝은 창이 짧아짐)"""
    n, h = len(x), w // 2
    arr = np.asarray(x, dtype=float)
    out = [0.0] * n
    if n >= w:
        win = np.lib.stride_tricks.sliding_window_view(arr, w)
        mid = np.median(win, axis=1)  # 창 [i-h, i+h] 가 온전한 i = h .. n-h-1
        for k, i in enumerate(range(h, n - h)):
            out[i] = float(mid[k])
    for i in list(range(0, min(h, n))) + list(range(max(h, n - h), n)):
        out[i] = median(x[max(0, i - h):min(n, i + h + 1)])
    if n < w:
        out = [median(x[max(0, i - h):min(n, i + h + 1)]) for i in range(n)]
    return out


def rolling_robust_std(x, w: int):
    """뒤를 보는 이동 robust 표준편차 (최근 w개). 값이 6개 미만이면 NaN"""
    n = len(x)
    out = [NaN] * n
    need = min(6, w)
    arr = np.asarray(x, dtype=float)
    if n >= w:
        win = np.lib.stride_tricks.sliding_window_view(arr, w)
        med = np.median(win, axis=1)
        md = np.median(np.abs(win - med[:, None]), axis=1)
        for k in range(len(md)):
            out[k + w - 1] = float(1.4826 * md[k])
    for i in range(min(w - 1, n)):
        s = x[max(0, i - w + 1):i + 1]
        if len(s) >= need:
            out[i] = 1.4826 * mad(s)
    return out


def linreg(xs, ys):
    """단순 선형 회귀. 기울기 b, 절편 a, 기울기의 t값"""
    n = len(xs)
    if n < 3:
        return None
    mx, my = mean(xs), mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if not sxx > 0:
        return None
    b = sxy / sxx
    a = my - b * mx
    sse = sum((y - a - b * x) ** 2 for x, y in zip(xs, ys))
    se = math.sqrt(sse / max(1, n - 2) / sxx)
    t = b / se if se > 0 else (math.inf if b else 0.0)
    return {"a": a, "b": b, "se": se, "t": t}


def chi2q(p: int, z: float) -> float:
    """카이제곱 분위수 근사 (Wilson–Hilferty)"""
    a = 2 / (9 * p)
    return p * (1 - a + z * math.sqrt(a)) ** 3


# ---------------------------------------------------------------- 시간
def _pad(n):
    return str(n).zfill(2)


def fmt_time(ms, with_sec=False) -> str:
    if not is_num(ms):
        return ""
    d = _EPOCH + timedelta(milliseconds=ms)
    s = f"{d.year}-{_pad(d.month)}-{_pad(d.day)} {_pad(d.hour)}:{_pad(d.minute)}"
    return s + (f":{_pad(d.second)}" if with_sec else "")


def fmt_short(ms, with_sec=False) -> str:
    if not is_num(ms):
        return ""
    d = _EPOCH + timedelta(milliseconds=ms)
    s = f"{_pad(d.month)}/{_pad(d.day)} {_pad(d.hour)}:{_pad(d.minute)}"
    return s + (f":{_pad(d.second)}" if with_sec else "")


_T1 = re.compile(r"^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})(?:[ T](\d{1,2}):(\d{2})(?::(\d{2}))?)?")
_T2 = re.compile(r"^(\d{4})(\d{2})(\d{2})(?:[ T]?(\d{2}):?(\d{2})(?::?(\d{2}))?)?$")


def parse_time(s) -> float:
    """'2026-10-19 08:30', '2026/10/19', '20261019 0830' 같은 글자를 밀리초 숫자로. 못 읽으면 NaN"""
    if s is None:
        return NaN
    if isinstance(s, datetime):
        return (s - _EPOCH).total_seconds() * 1000
    s = str(s).strip()
    if not s:
        return NaN
    m = _T1.match(s) or _T2.match(s)
    if not m:
        return NaN
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if mo < 1 or mo > 12 or d < 1 or d > 31:
        return NaN
    hh, mm, ss = int(m.group(4) or 0), int(m.group(5) or 0), int(m.group(6) or 0)
    try:
        dt = datetime(y, mo, 1) + timedelta(days=d - 1, hours=hh, minutes=mm, seconds=ss)
    except (ValueError, OverflowError):
        return NaN
    return (dt - _EPOCH).total_seconds() * 1000


# ---------------------------------------------------------------- CSV
def detect_delimiter(text: str) -> str:
    line = next((l for l in re.split(r"\r?\n", text) if l.strip()), "")
    best, bc = ",", -1
    for c in [",", "\t", ";", "|"]:
        k = len(line.split(c))
        if k > bc:
            bc, best = k, c
    return best


def decode_bytes(buf: bytes):
    """UTF-8로 먼저 읽고, 안 되면 CP949(한글 엑셀 CSV)로 읽음"""
    try:
        return buf.decode("utf-8"), "UTF-8"
    except UnicodeDecodeError:
        try:
            return buf.decode("cp949"), "CP949(EUC-KR)"
        except UnicodeDecodeError:
            return buf.decode("utf-8", errors="replace"), "UTF-8(일부 깨짐)"


def parse_csv(text: str) -> dict:
    """CSV 글자를 {'columns': [...], 'rows': [[...], ...]} 로. 모든 칸은 글자 그대로 둠"""
    if text and text[0] == "﻿":
        text = text[1:]
    delim = detect_delimiter(text)
    rows = [r for r in csv.reader(io.StringIO(text, newline=""), delimiter=delim)
            if len(r) > 1 or (len(r) == 1 and r[0] != "")]
    if not rows:
        return {"columns": [], "rows": [], "delimiter": delim}
    header = [(h or "").strip() or f"열{i + 1}" for i, h in enumerate(rows[0])]
    body = [[(r[i] if i < len(r) else "") for i in range(len(header))] for r in rows[1:]]
    return {"columns": header, "rows": body, "delimiter": delim}


def load_csv(source) -> dict:
    """파일 경로, 바이트, Streamlit 업로드 파일 무엇이든 받아서 표로 읽음"""
    if hasattr(source, "getvalue"):
        buf = source.getvalue()
    elif hasattr(source, "read"):
        buf = source.read()
    elif isinstance(source, (bytes, bytearray)):
        buf = bytes(source)
    else:
        with open(source, "rb") as f:
            buf = f.read()
    text, enc = decode_bytes(buf)
    table = parse_csv(text)
    table["encoding"] = enc
    return table


def table_from_dataframe(df) -> dict:
    """pandas DataFrame을 엔진이 쓰는 표 형태로 바꿈"""
    cols = [str(c) for c in df.columns]
    rows = [["" if (v is None or (isinstance(v, float) and math.isnan(v))) else str(v) for v in r]
            for r in df.itertuples(index=False, name=None)]
    return {"columns": cols, "rows": rows}


# ---------------------------------------------------------------- 도메인 팩 (열 이름으로 의미 추정 + 기준 규칙 예시)
DICT = [
    (re.compile(r"냉장|냉동|chill|fridge|refrig", re.I), re.compile(r"온도|temp", re.I), "냉장 보관 온도", "℃", "CCP-2 냉장 보관"),
    (re.compile(r"가열|중심온도|살균|cook|core", re.I), None, "가열 중심온도", "℃", "CCP-1 가열"),
    (re.compile(r"노즐|nozzle|hotend", re.I), None, "노즐 온도", "℃", ""),
    (re.compile(r"베드|bed", re.I), None, "베드 온도", "℃", ""),
    (re.compile(r"체온", re.I), None, "체온", "℃", ""),
    (re.compile(r"온도|temp", re.I), None, "온도", "℃", ""),
    (re.compile(r"습도|humid", re.I), None, "상대 습도", "%", ""),
    (re.compile(r"컴프레서|compressor", re.I), re.compile(r"전류|current|amp", re.I), "컴프레서 전류", "A", ""),
    (re.compile(r"압출|extru", re.I), re.compile(r"전류|current|amp", re.I), "압출 모터 전류", "A", ""),
    (re.compile(r"전류|current|amp", re.I), None, "전류", "A", ""),
    (re.compile(r"진동|vib|rms", re.I), None, "진동 세기", "g", ""),
    (re.compile(r"압력|press", re.I), None, "압력", "kPa", ""),
    (re.compile(r"전압|volt", re.I), None, "전압", "V", ""),
    (re.compile(r"유량|flow", re.I), None, "유량", "", ""),
    (re.compile(r"매출|sales|revenue", re.I), None, "매출액", "원", ""),
    (re.compile(r"금액|amount|price|가격", re.I), None, "금액", "원", ""),
    (re.compile(r"심박|heart|bpm", re.I), None, "심박수", "bpm", ""),
    (re.compile(r"혈당|glucose", re.I), None, "혈당", "mg/dL", ""),
]

PACKS = {
    "food": {"id": "food", "name": "식품 안전 (HACCP)", "short": "HACCP",
             "desc": "가열·냉장 중요관리점(CCP) 기준과 공정 센서 패턴을 함께 봐요.",
             "rules": [
                 {"meaning": "냉장 보관 온도", "op": ">", "value": 5, "durationMin": 30, "label": "냉장 온도 5℃ 초과 30분 지속", "ccp": "CCP-2"},
                 {"meaning": "가열 중심온도", "op": "<", "value": 75, "durationMin": 0, "label": "가열 중심온도 75℃ 미만", "ccp": "CCP-1"},
             ]},
    "phm": {"id": "phm", "name": "제조 설비 (PHM)", "short": "PHM",
            "desc": "설비 센서의 마모·열화 추세를 잡아 고장 전에 알려요.",
            "rules": [
                {"meaning": "노즐 온도", "op": "<", "value": 200, "durationMin": 0, "label": "노즐 온도 200℃ 미만"},
                {"meaning": "노즐 온도", "op": ">", "value": 220, "durationMin": 0, "label": "노즐 온도 220℃ 초과"},
                {"meaning": "진동 세기", "op": ">", "value": 0.3, "durationMin": 0, "label": "진동 0.3g 초과"},
            ]},
    "general": {"id": "general", "name": "범용 분석", "short": "범용", "desc": "도메인 기준 없이 통계 패턴만으로 이상을 찾아요.", "rules": []},
}


def display_name(col) -> str:
    s = re.sub(r"_+", " ", re.sub(r"\([^)]*\)", "", str(col))).strip()
    return s or str(col)


def unit_from_name(col) -> str:
    m = re.search(r"\(([^)]{1,8})\)", str(col))
    return m.group(1) if m else ""


def infer_meaning(col) -> dict:
    for rx, also, meaning, unit, ccp in DICT:
        if rx.search(col) and (also is None or also.search(col)):
            return {"meaning": meaning, "unit": unit_from_name(col) or unit, "ccp": ccp}
    return {"meaning": "", "unit": unit_from_name(col), "ccp": ""}


# ---------------------------------------------------------------- 데이터 점검 (열 역할 판별)
NAME_HINT = re.compile(r"이름|성명|작업자|담당자|환자|고객|name|worker|operator", re.I)
PHONE_RE = re.compile(r"^0\d{1,2}-?\d{3,4}-?\d{4}$")
RRN_RE = re.compile(r"^\d{6}-?[1-4]\d{6}$")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
KNAME_RE = re.compile(r"^[가-힣]{2,4}$")
LABEL_HINT = re.compile(r"정답|label|이상여부|is_?anomaly|target|ground.?truth", re.I)
COUNTER_HINT = re.compile(r"(^|_)id$|번호|순번|레이어|layer|index|^no$", re.I)
_NUM_RE = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")


def to_number(s) -> float:
    """'1,234.5' 같은 글자를 숫자로. 숫자가 아니면 NaN"""
    if isinstance(s, (int, float)) and not isinstance(s, bool):
        return float(s)
    if s is None:
        return NaN
    t = str(s).strip().replace(",", "")
    if not t or not _NUM_RE.match(t):
        return NaN
    return float(t)


def _top_values(strs):
    cnt = {}
    for s in strs:
        cnt[s] = cnt.get(s, 0) + 1
    return sorted(cnt.items(), key=lambda kv: -kv[1])[:4]


def profile(table: dict) -> dict:
    """열마다 역할(숫자·시간·개인정보·정답 라벨·순번·범주)을 판별하고 결측·중복·측정 간격을 셈"""
    columns, rows = table["columns"], table["rows"]
    n = len(rows)
    cols = []
    for ci, name in enumerate(columns):
        raw = [r[ci] for r in rows]
        non_empty = [v for v in raw if v is not None and str(v).strip() != ""]
        k = max(1, len(non_empty))
        nums = [to_number(v) for v in non_empty]
        num_ratio = sum(1 for v in nums if is_num(v)) / k
        time_ratio = sum(1 for v in non_empty if is_num(parse_time(v))) / k
        strs = [str(v).strip() for v in non_empty]
        uniq = len(set(strs))
        c = {"index": ci, "name": name, "display": display_name(name), "missing": n - len(non_empty), "unique": uniq,
             "role": "text", "kind": "", "meaning": "", "unit": "", "ccp": "", "stats": None}

        def ratio(rx):
            return sum(1 for s in strs if rx.search(s)) / k

        if ratio(PHONE_RE) >= 0.8:
            c["role"], c["kind"] = "personal", "전화번호"
        elif ratio(RRN_RE) >= 0.8:
            c["role"], c["kind"] = "personal", "주민등록번호"
        elif ratio(EMAIL_RE) >= 0.8:
            c["role"], c["kind"] = "personal", "이메일"
        elif ratio(KNAME_RE) >= 0.8 and (NAME_HINT.search(name) or (5 <= uniq <= n * 0.5)):
            c["role"], c["kind"] = "personal", "이름"
        elif time_ratio >= 0.9 and num_ratio < 0.9:
            c["role"] = "time"
        elif num_ratio >= 0.95:
            vals = [v for v in nums if is_num(v)]
            uniq_nums = set(vals)
            all_int = all(abs(v - round(v)) < 1e-9 for v in vals)
            nondecr = all(vals[i] >= vals[i - 1] for i in range(1, len(vals)))
            if LABEL_HINT.search(name) and all(v in (0, 1) for v in uniq_nums):
                c["role"] = "label"
            elif all_int and (COUNTER_HINT.search(name) or (nondecr and len(uniq_nums) > 5)):
                c["role"] = "counter"
            elif len(uniq_nums) <= 1:
                c["role"] = "constant"
            else:
                c["role"] = "numeric"
            c["stats"] = {"min": min(vals), "max": max(vals), "mean": mean(vals), "median": median(vals), "std": std(vals),
                          "q1": quantile(vals, 0.25), "q3": quantile(vals, 0.75)}
            if c["role"] == "numeric":
                c.update(infer_meaning(name))
        elif uniq <= 20 or uniq / k < 0.05:
            c["role"] = "category"
            c["top"] = _top_values(strs)
        cols.append(c)
    # 완전히 같은 행(중복)
    seen, dup_rows = set(), []
    for i, r in enumerate(rows):
        key = "\u0001".join(r)
        if key in seen:
            dup_rows.append(i)
        else:
            seen.add(key)
    time_col = next((c for c in cols if c["role"] == "time"), None)
    time = None
    if time_col:
        ts = sorted(v for v in (parse_time(r[time_col["index"]]) for r in rows) if is_num(v))
        diffs = [ts[i] - ts[i - 1] for i in range(1, len(ts))]
        posd = [d for d in diffs if d > 0]
        med = median(posd)
        time = {"col": time_col["name"], "start": ts[0], "end": ts[-1], "intervalMs": med,
                "gaps": sum(1 for d in posd if d > med * 1.5), "dupStamps": sum(1 for d in diffs if d == 0)}
    return {"n": n, "cols": cols, "duplicates": len(dup_rows), "dupRows": dup_rows, "time": time,
            "totalMissing": sum(c["missing"] for c in cols)}


def mask_value(v, kind) -> str:
    s = str(v or "")
    if not s:
        return s
    if kind == "이름":
        return s[0] + "*" if len(s) <= 2 else s[0] + "*" * (len(s) - 2) + s[-1]
    if kind == "전화번호":
        return re.sub(r"^(\d{2,3})-?(\d{3,4})-?(\d{4})$", lambda m: m.group(1) + "-****-" + m.group(3), s)
    if kind == "주민등록번호":
        return s[:6] + "-*******"
    if kind == "이메일":
        return re.sub(r"^(.)[^@]*(@.*)$", r"\1***\2", s)
    return "***"


def guess_pack(prof) -> str:
    names = " ".join(c["meaning"] for c in prof["cols"])
    if re.search(r"냉장|가열", names):
        return "food"
    if re.search(r"노즐|베드|진동|모터|컴프레서|전류|압력", names):
        return "phm"
    return "general"


def recommend_columns(prof):
    """분석할 숫자 열 추천 (변하는 숫자 열 최대 6개)"""
    return [c["name"] for c in prof["cols"] if c["role"] == "numeric" and c["stats"] and c["stats"]["std"] > 0][:6]


def pack_rules(pack_id, prof):
    pack = PACKS.get(pack_id)
    if not pack:
        return []
    out = []
    for i, r in enumerate(pack["rules"]):
        col = next((c for c in prof["cols"] if c["role"] == "numeric" and c["meaning"] == r["meaning"]), None)
        if col:
            out.append({"id": f"{pack_id}-{i}", "col": col["name"], "op": r["op"], "value": r["value"], "durationMin": r["durationMin"],
                        "label": r["label"], "ccp": r.get("ccp", ""), "source": pack["short"] + " 팩 예시 기준", "on": True})
    return out


# ---------------------------------------------------------------- 핵심 분석
METHODS = [
    {"key": "point", "name": "튀는 값 탐지", "tech": "이동 중앙값 잔차 robust Z (|z|≥4)"},
    {"key": "ewma", "name": "관리도", "tech": "EWMA 관리도 (λ=0.2, 3σ 한계)"},
    {"key": "vari", "name": "변동성 감시", "tech": "이동 robust 표준편차 (기준 대비 2.5배)"},
    {"key": "mv", "name": "다변량", "tech": "Mahalanobis 거리 (χ² 99.9%)"},
]


def prepare(table, prof, selected):
    """중복 행을 빼고 시간순으로 정렬한 뒤, 고른 열을 숫자로 바꿈"""
    time_col = next((c for c in prof["cols"] if c["role"] == "time"), None)
    label_col = next((c for c in prof["cols"] if c["role"] == "label"), None)
    dup = set(prof["dupRows"])
    rows = table["rows"]
    idx = [i for i in range(len(rows)) if i not in dup]
    times = None
    if time_col:
        pairs = [(i, parse_time(rows[i][time_col["index"]])) for i in idx]
        pairs = sorted((p for p in pairs if is_num(p[1])), key=lambda p: p[1])
        idx = [p[0] for p in pairs]
        times = [p[1] for p in pairs]
    cols = []
    for name in selected:
        c = next(x for x in prof["cols"] if x["name"] == name)
        cols.append({"name": name, "display": c["display"], "unit": c["unit"], "meaning": c["meaning"], "ccp": c["ccp"],
                     "values": [to_number(rows[i][c["index"]]) for i in idx]})
    labels = [1 if to_number(rows[i][label_col["index"]]) == 1 else 0 for i in idx] if label_col else None
    return {"idx": idx, "rowNums": [i + 2 for i in idx], "times": times, "cols": cols, "labels": labels,
            "labelName": label_col["name"] if label_col else "", "intervalMs": prof["time"]["intervalMs"] if prof["time"] else NaN}


def analyze_base(prep):
    """열마다 4가지 방법의 점수를 계산. 점수 1.0 = 그 방법의 기준선에 닿음 (k로 문턱 조절)"""
    n = len(prep["idx"])
    nb = min(n, max(30, math.floor(n * 0.3)))      # 기준 구간: 앞쪽 30% (최소 30행)
    lam = 0.2
    w = min(15, n if n % 2 else n - 1)
    for c in prep["cols"]:
        x = interpolate(c["values"])
        c["filled"] = x
        rb = robust(x[:nb])
        c["mu"], c["sigma"] = rb["m"], rb["s"]
        c["band"] = [rb["m"] - 3 * rb["s"], rb["m"] + 3 * rb["s"]]
        # ① 튀는 값: 가운데 이동 중앙값에서 얼마나 벗어났나
        rm = rolling_median_centered(x, w)
        r = [v - rm[i] for i, v in enumerate(x)]
        rr = robust(r)
        c["resid"] = r
        c["point"] = [abs(v - rr["m"]) / rr["s"] / 4 for v in r]
        # ② 관리도(EWMA): 튀는 값은 잘라 낸 뒤 계산해서 한 번 튄 값이 오래 남지 않게
        xw = [min(rm[i] + 4 * rr["s"], max(rm[i] - 4 * rr["s"], v)) for i, v in enumerate(x)]
        se = rb["s"] * math.sqrt(lam / (2 - lam))
        z = rb["m"]
        c["ewmaVal"], c["ewma"] = [], []
        for i in range(n):
            z = lam * xw[i] + (1 - lam) * z
            c["ewmaVal"].append(z)
            raw = abs(z - rb["m"]) / (3 * se)
            # 지금 값 자체도 기준에서 1σ 이상 떨어져 있어야 온전히 인정
            c["ewma"].append(raw if abs(xw[i] - rb["m"]) >= rb["s"] else raw * 0.5)
        c["ewmaLimits"] = [rb["m"] - 3 * se, rb["m"] + 3 * se]
        # ③ 변동성: 최근 16행의 흔들림이 기준 구간의 몇 배인가
        rs = rolling_robust_std(r, 16)
        s_base = robust(r[:nb])["s"]
        c["vari"] = [v / s_base / 2.5 if is_num(v) else 0.0 for v in rs]
        c["missingIdx"] = [i for i, v in enumerate(c["values"]) if not is_num(v)]
    # ④ 다변량: 변수 사이 관계가 평소와 다른가 (변수 2개 이상일 때)
    p = len(prep["cols"])
    mv = contrib = None
    if p >= 2:
        Z = np.array([[(c["filled"][i] - c["mu"]) / c["sigma"] for c in prep["cols"]] for i in range(n)])
        B = Z[:nb]
        mu = np.array([median(list(B[:, j])) for j in range(p)])
        D = B - mu
        S = D.T @ D / (len(B) - 1)
        alpha = 0.1
        S = np.where(np.eye(p, dtype=bool), S, (1 - alpha) * S)   # 대각선은 그대로, 나머지는 0.9배 (안정화)
        try:
            if np.linalg.cond(S) > 1e12:
                raise np.linalg.LinAlgError
            Si = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            Si = None
        if Si is not None:
            thr = chi2q(p, 3.09)
            Dz = Z - mu
            Sd = Dz @ Si.T
            parts = np.maximum(0, Dz * Sd)
            D2 = np.sum(Dz * Sd, axis=1)
            mv = list(np.sqrt(np.maximum(0, D2) / thr))
            tot = parts.sum(axis=1)
            tot[tot == 0] = 1
            contrib = [list(row) for row in (parts / tot[:, None])]
    return {"n": n, "nb": nb, "mv": [float(v) for v in mv] if mv is not None else None,
            "contrib": [[float(v) for v in r] for r in contrib] if contrib is not None else None}


def eval_rules(prep, rules):
    """기준 규칙(예: 냉장 5℃ 초과 30분 지속)을 어긴 행 표시"""
    n = len(prep["idx"])
    hits = [[] for _ in range(n)]
    iv = prep["intervalMs"]
    for rule in rules:
        if rule.get("on") is False:
            continue
        c = next((x for x in prep["cols"] if x["name"] == rule["col"]), None)
        if not c:
            continue
        need = max(1, math.ceil(rule["durationMin"] * 60000 / iv)) if rule["durationMin"] > 0 and is_num(iv) and iv > 0 else 1
        op, val = rule["op"], rule["value"]

        def ok(v):
            if not is_num(v):
                return False
            return v > val if op == ">" else v >= val if op == ">=" else v < val if op == "<" else v <= val

        cond = [ok(v) for v in c["values"]]
        i = 0
        while i < n:
            if not cond[i]:
                i += 1
                continue
            j = i
            while j < n and cond[j]:
                j += 1
            if j - i >= need:
                for t in range(i, j):
                    hits[t].append(rule["id"])
            i = j
    return hits


def classify(prep, base, rules, k=1.0, mult=None, dismissed=None):
    """행마다 4가지 점수를 모아 이상 여부·등급을 정하고, 이벤트로 묶음.
    k: 문턱 배수 (기본 1.0, 크면 덜 민감)  mult: {열 이름: 배수} 열별 민감도  dismissed: 사용자가 '정상'이라고 표시한 행 번호 집합"""
    mult = mult or {}
    n, cols = base["n"], prep["cols"]
    kc = [k * mult.get(c["name"], 1) for c in cols]
    rule_hits = eval_rules(prep, rules)
    # 관리도 신호는 3행 이상 이어질 때만 인정
    ew_eff = []
    for cj, c in enumerate(cols):
        out, kk = list(c["ewma"]), kc[cj]
        i = 0
        while i < n:
            if c["ewma"][i] < kk:
                i += 1
                continue
            j = i
            while j < n and c["ewma"][j] >= kk:
                j += 1
            if j - i < 3:
                for t in range(i, j):
                    out[t] = min(out[t], kk * 0.99)
            i = j
        ew_eff.append(out)
    rows = []
    for i in range(n):
        m = {"point": 0.0, "ewma": 0.0, "vari": 0.0, "mv": base["mv"][i] if base["mv"] else 0.0}
        hit = {"point": False, "ewma": False, "vari": False}
        col_flag, col_base = [], []
        for j, c in enumerate(cols):
            ew, kk = ew_eff[j][i], kc[j]
            m["point"] = max(m["point"], c["point"][i])
            m["ewma"] = max(m["ewma"], ew)
            m["vari"] = max(m["vari"], c["vari"][i])
            if c["point"][i] >= kk:
                hit["point"] = True
            if ew >= kk:
                hit["ewma"] = True
            if c["vari"][i] >= kk:
                hit["vari"] = True
            b = max(c["point"][i], ew, c["vari"][i])
            cv = (c["point"][i] >= kk) + (ew >= kk) + (c["vari"][i] >= kk)
            col_flag.append(cv >= 2 or b >= 1.5 * kk)
            col_base.append(b)
        top = max(m["point"], m["ewma"], m["vari"], m["mv"])
        votes = hit["point"] + hit["ewma"] + hit["vari"] + (m["mv"] >= k)
        cause = [j for j in range(len(cols)) if col_flag[j]]
        strong = list(cause)
        if m["mv"] >= k and base["contrib"]:
            for j, sh in enumerate(base["contrib"][i]):
                zj = abs(cols[j]["filled"][i] - cols[j]["mu"]) / cols[j]["sigma"]
                if sh >= 0.3 and zj >= 2 and j not in cause:
                    cause.append(j)
        stat = len(cause) > 0 or m["mv"] >= 1.5 * k
        rule = len(rule_hits[i]) > 0
        user_dismissed = bool(dismissed and i in dismissed and not rule)
        if user_dismissed:
            stat, cause, strong = False, [], []
        flagged = stat or rule
        sev = ""
        if flagged:
            sev = "critical" if rule else ("serious" if top >= 2.5 * k or votes >= 3 else "warning")
        score = min(99, js_round(100 * (1 - math.exp(-0.8 * top)) + 4 * max(0, votes - 1)))
        for rid in rule_hits[i]:
            r = next((x for x in rules if x["id"] == rid), None)
            j = next((jj for jj, c in enumerate(cols) if c["name"] == (r and r["col"])), -1)
            if j >= 0:
                if j not in cause:
                    cause.append(j)
                if j not in strong:
                    strong.append(j)
        rows.append({"i": i, "m": m, "colBase": col_base, "top": top, "votes": votes, "stat": stat, "rule": rule, "flagged": flagged,
                     "sev": sev, "score": score, "cause": cause, "strong": strong, "ruleIds": rule_hits[i], "dismissed": user_dismissed})
    events = build_events(prep, base, rows, rules, k)
    return {"rows": rows, "events": events, "k": k}


TYPE_INFO = {
    "spike": {"name": "튀는 값", "hint": "순간적인 외부 요인(문 열림, 센서 접촉 불량 등)일 가능성이 있어요."},
    "shift": {"name": "수준 이동", "hint": "설정값 변경이나 부품 이상으로 기준선 자체가 바뀌었을 가능성이 있어요."},
    "drift": {"name": "서서히 변화", "hint": "마모·성능 저하처럼 점점 나빠지는 패턴이에요. 고장 전조일 가능성이 있어요."},
    "vari": {"name": "변동성 증가", "hint": "센서 고장이나 제어 불안정으로 값이 크게 흔들릴 가능성이 있어요."},
    "relation": {"name": "관계 이상", "hint": "개별 값은 정상 범위인데 변수 사이의 관계가 평소와 달라요."},
}
_SEV_RANK = {"critical": 3, "serious": 2, "warning": 1}


def build_events(prep, base, rows, rules, k):
    """이상 행을 구간(이벤트)으로 묶음. 4행까지 끊겨도 같은 구간으로 보고, 많이 겹치는 구간은 합침"""
    n, cols, segs = len(rows), prep["cols"], []

    def segment(fl, j):
        i = 0
        while i < n:
            if not fl(i):
                i += 1
                continue
            t, gap, end = i, 0, i
            while t < n:
                if fl(t):
                    end, gap = t, 0
                else:
                    gap += 1
                    if gap > 4:
                        break
                t += 1
            segs.append({"s": i, "e": end, "cols": [j] if j >= 0 else []})
            i = end + 1

    for j in range(len(cols)):
        segment(lambda i, j=j: rows[i]["flagged"] and j in rows[i]["strong"], j)
    segment(lambda i: rows[i]["flagged"] and len(rows[i]["strong"]) == 0, -1)
    segs.sort(key=lambda g: g["s"])
    merged = []
    for g in segs:
        hit = None
        for mg in merged:
            ov = min(mg["e"], g["e"]) - max(mg["s"], g["s"]) + 1
            if ov > 0 and ov >= 0.5 * max(mg["e"] - mg["s"] + 1, g["e"] - g["s"] + 1):
                hit = mg
                break
        if hit:
            hit["s"], hit["e"] = min(hit["s"], g["s"]), max(hit["e"], g["e"])
            for j in g["cols"]:
                if j not in hit["cols"]:
                    hit["cols"].append(j)
        else:
            merged.append({"s": g["s"], "e": g["e"], "cols": list(g["cols"])})
    # 1~2행짜리 약한 깜빡임은 버림 (규칙 위반이 아니고 점수가 약하면)
    out = [make_event(prep, base, rows, rules, g["s"], g["e"], g["cols"], k) for g in merged]
    out = [e for e in out if not (e["len"] <= 2 and e["sev"] != "critical" and e["evTop"] < 2 * k)]
    out.sort(key=lambda e: (-_SEV_RANK[e["sev"]], -e["peakScore"], e["start"]))
    for idx, e in enumerate(out):
        e["id"] = f"E{idx + 1}"
    for r in rows:
        r["events"] = []
    for e in out:
        for t in range(e["start"], e["end"] + 1):
            if rows[t]["flagged"]:
                rows[t]["events"].append(e["id"])
    for r in rows:
        if r["flagged"] and not r["events"] and not r["rule"]:
            r["flagged"], r["sev"] = False, ""
    return out


def make_event(prep, base, rows, rules, s, e, ev_cols, k):
    """이벤트 하나의 유형(튀는 값/수준 이동/서서히 변화/변동성 증가/관계 이상), 등급, 주요 변수를 정함"""
    cols, ln = prep["cols"], e - s + 1
    relation = len(ev_cols) == 0
    mx = {"point": 0.0, "ewma": 0.0, "vari": 0.0, "mv": 0.0}
    col_sum = [0.0] * len(cols)
    ev_top, max_votes = 0.0, 0
    rule_set = []
    for t in range(s, e + 1):
        r = rows[t]
        for j in ev_cols:
            c = cols[j]
            mx["point"] = max(mx["point"], c["point"][t])
            mx["ewma"] = max(mx["ewma"], c["ewma"][t])
            mx["vari"] = max(mx["vari"], c["vari"][t])
            col_sum[j] += r["colBase"][j]
            ev_top = max(ev_top, r["colBase"][j])
        if base["mv"]:
            mx["mv"] = max(mx["mv"], base["mv"][t])
        if relation and base["contrib"]:
            for j, v in enumerate(base["contrib"][t]):
                col_sum[j] += v
        if relation:
            ev_top = max(ev_top, r["m"]["mv"])
        max_votes = max(max_votes, r["votes"])
        for rid in r["ruleIds"]:
            ru = next((x for x in rules if x["id"] == rid), None)
            if ru and (relation or any(cols[j]["name"] == ru["col"] for j in ev_cols)) and rid not in rule_set:
                rule_set.append(rid)
    rule_list = [r for r in (next((x for x in rules if x["id"] == rid), None) for rid in rule_set) if r]
    main = col_sum.index(max(col_sum))
    if rule_list:
        j = next((jj for jj, c in enumerate(cols) if c["name"] == rule_list[0]["col"]), -1)
        if j >= 0:
            main = j
    if not relation and main not in ev_cols:
        main = ev_cols[0]
    c = cols[main]
    keys = ["mv"] if relation else ["point", "ewma", "vari"]
    dom = keys[0]
    for kk in keys[1:]:
        dom = dom if mx[dom] >= mx[kk] else kk
    ev_vals = c["filled"][s:e + 1]
    ev_sd = std(ev_vals) or c["sigma"]
    if relation:
        typ = "relation"
    elif mx["vari"] >= k and ln >= 6 and abs(mean(ev_vals) - c["mu"]) / ev_sd < 0.8:
        typ = "vari"
    elif ln <= 5 or (dom == "point" and ln <= 12):
        typ = "spike"
    else:
        lr = linreg(list(range(s, e + 1)), c["filled"][s:e + 1])
        rise = abs(lr["b"] * ln) if lr else 0
        typ = "drift" if lr and abs(lr["t"]) > 5 and ln >= 12 and rise > 2 * c["sigma"] else "shift"
    peak = s
    for t in range(s, e + 1):
        if abs(c["filled"][t] - c["mu"]) > abs(c["filled"][peak] - c["mu"]):
            peak = t
    peak_score = min(99, js_round(100 * (1 - math.exp(-0.8 * ev_top))))
    sev = "critical" if rule_list else ("serious" if ev_top >= 2.5 * k or max_votes >= 3 else "warning")
    pool = list(range(len(cols))) if relation else list(ev_cols)
    involved = [j for j, _ in sorted(((j, col_sum[j]) for j in pool), key=lambda o: -o[1])]
    if involved[0] != main:
        involved.remove(main)
        involved.insert(0, main)
    return {"start": s, "end": e, "len": ln, "peak": peak, "sev": sev, "type": typ, "typeName": TYPE_INFO[typ]["name"],
            "hint": TYPE_INFO[typ]["hint"], "main": main, "cols": involved[:3] if relation else involved,
            "peakVal": c["filled"][peak], "devSigma": (c["filled"][peak] - c["mu"]) / c["sigma"], "peakScore": peak_score,
            "rules": [r["label"] for r in rule_list], "dominant": dom, "methods": mx, "evTop": ev_top}


def forecasts(prep, base, rules):
    """기준 규칙이 있는 변수가 지금 추세로 몇 시간 뒤 기준에 닿을지 (최근 구간 직선 추세)"""
    out = []
    n = base["n"]
    if not prep["times"] or n < 30:
        return out
    W = max(24, js_round(n * 0.12))
    for rule in rules:
        if rule.get("on") is False:
            continue
        j = next((jj for jj, c in enumerate(prep["cols"]) if c["name"] == rule["col"]), -1)
        if j < 0:
            continue
        c = prep["cols"][j]
        xs, ys = [], []
        for t in range(n - W, n):
            if is_num(c["values"][t]):
                xs.append((prep["times"][t] - prep["times"][n - 1]) / 3600000)
                ys.append(c["values"][t])
        lr = linreg(xs, ys)
        if not lr:
            continue
        now = lr["a"]
        up = rule["op"] in (">", ">=")
        safe = now < rule["value"] if up else now > rule["value"]
        toward = lr["b"] > 0 if up else lr["b"] < 0
        if not safe or not toward or abs(lr["t"]) < 4:
            continue
        hours = (rule["value"] - now) / lr["b"]
        span_h = (prep["times"][n - 1] - prep["times"][0]) / 3600000
        if 0 < hours <= span_h * 1.5:
            out.append({"col": j, "colName": c["name"], "display": c["display"], "unit": c["unit"], "rule": rule["label"],
                        "threshold": rule["value"], "now": now, "slopePerHour": lr["b"], "hours": hours,
                        "lastTime": prep["times"][n - 1], "window": W})
    return out


def _score(pred, L):
    tp = sum(1 for p, l in zip(pred, L) if p and l)
    fp = sum(1 for p, l in zip(pred, L) if p and not l)
    fn = sum(1 for p, l in zip(pred, L) if not p and l)
    pr = tp / (tp + fp) if tp + fp else 0
    rc = tp / (tp + fn) if tp + fn else 0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": pr, "recall": rc, "f1": 2 * pr * rc / (pr + rc) if pr + rc else 0}


def evaluate(prep, base, rules, k=1.0, **opts):
    """정답 열(정답_이상여부 등)이 있으면 방법별 정밀도·재현율·F1을 계산"""
    if not prep["labels"]:
        return None
    res = classify(prep, base, rules, k, **opts)
    L = prep["labels"]
    methods = [dict(key=m["key"], name=m["name"], tech=m["tech"], **_score([r["m"][m["key"]] >= k for r in res["rows"]], L))
               for m in METHODS if m["key"] != "mv" or base["mv"]]
    rule_only = dict(key="rule", name="기준 규칙만", tech="고정 기준치", **_score([r["rule"] for r in res["rows"]], L))
    ens = dict(key="ens", name="VibeData 종합", tech="2개 이상 합의 또는 강한 단일 신호 + 규칙", **_score([r["flagged"] for r in res["rows"]], L))
    return {"methods": methods, "ruleOnly": rule_only, "ens": ens, "positives": sum(L), "n": len(L)}


def sweep(prep, base, rules, **opts):
    """k를 0.6~2.0으로 바꿔 가며 종합 판정의 정밀도·재현율·F1"""
    if not prep["labels"]:
        return None
    pts, k = [], 0.6
    while k <= 2.001:
        e = evaluate(prep, base, rules, k, **opts)
        pts.append({"k": round_to(k, 1), "precision": e["ens"]["precision"], "recall": e["ens"]["recall"], "f1": e["ens"]["f1"]})
        k += 0.1
    return pts


# ---------------------------------------------------------------- AI 요약과 숫자 검증
def build_summary(prep, base, res, fc, rules, pack, dataset_name="", excluded=None):
    """AI에게 보낼 요약 (원본 행은 넣지 않음)"""
    sev_ko = {"critical": "위험", "serious": "경고", "warning": "주의"}
    with_sec = is_num(prep["intervalMs"]) and prep["intervalMs"] < 60000

    def t(i):
        return fmt_time(prep["times"][i], with_sec) if prep["times"] else f"행 {prep['rowNums'][i]}"

    counts = {"critical": 0, "serious": 0, "warning": 0}
    for e in res["events"]:
        counts[e["sev"]] += 1
    n = base["n"]
    return {
        "데이터": {"이름": dataset_name, "분석_행수": n, "기간": f"{t(0)} ~ {t(n - 1)}" if prep["times"] else None,
                 "측정_간격_분": round_to(prep["intervalMs"] / 60000, 2) if is_num(prep["intervalMs"]) else None, "제외한_열": excluded or []},
        "도메인": pack["name"],
        "분석_방법": [m["tech"] for m in METHODS if m["key"] != "mv" or base["mv"]],
        "변수": [{"이름": c["display"], "의미": c["meaning"] or None, "단위": c["unit"] or None, "기준구간_중앙값": round_to(c["mu"], 2),
                "정상범위": [round_to(c["band"][0], 2), round_to(c["band"][1], 2)], "최솟값": round_to(min(c["filled"]), 2),
                "최댓값": round_to(max(c["filled"]), 2), "마지막값": round_to(c["filled"][n - 1], 2)} for c in prep["cols"]],
        "기준_규칙": [{"규칙": r["label"], "출처": r["source"]} for r in rules if r.get("on") is not False],
        "이벤트_수": {"전체": len(res["events"]), "위험": counts["critical"], "경고": counts["serious"], "주의": counts["warning"]},
        "이벤트": [{"id": e["id"], "등급": sev_ko[e["sev"]], "유형": e["typeName"], "시작": t(e["start"]), "끝": t(e["end"]), "행수": e["len"],
                  "주요_변수": prep["cols"][e["main"]]["display"], "관련_변수": [prep["cols"][j]["display"] for j in e["cols"]],
                  "가장_벗어난_값": round_to(prep["cols"][e["main"]]["filled"][e["peak"]], 2),
                  "정상범위": [round_to(prep["cols"][e["main"]]["band"][0], 2), round_to(prep["cols"][e["main"]]["band"][1], 2)],
                  "이탈_정도_시그마": round_to(e["devSigma"], 1), "이상_점수": e["peakScore"], "규칙_위반": e["rules"] or None}
                 for e in res["events"][:8]],
        "추세_예측": [{"변수": f["display"], "기준": f["rule"], "현재_추정값": round_to(f["now"], 2), "시간당_변화": round_to(f["slopePerHour"], 3),
                   "기준_도달까지_시간": round_to(f["hours"], 1)} for f in fc],
    }


_NUM_PLAIN = re.compile(r"(^|[^A-Za-z0-9_.])(-?\d+(?:\.\d+)?)")
_NUM_GROUPED = re.compile(r"(^|[^A-Za-z0-9_.])(-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?)")


def _numbers_in(text, grouped=False):
    rx = _NUM_GROUPED if grouped else _NUM_PLAIN
    return [(m.group(2), float(m.group(2).replace(",", ""))) for m in rx.finditer(text)]


def _num_key(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else repr(float(v))


def verify_numbers(report: str, summary_text: str) -> dict:
    """AI 리포트 속 숫자가 요약(계산값)에 있는지 대조. 10 이하 정수(목록 번호 등)는 건너뜀"""
    allowed = [v for _, v in _numbers_in(summary_text)]
    keys = {_num_key(v) for v in allowed}
    checked, bad = [], []
    for raw, v in _numbers_in(report, True):
        if float(v).is_integer() and abs(v) <= 10 and "." not in raw and "," not in raw:
            continue
        ok = _num_key(v) in keys or any(abs(a - v) <= max(0.051, abs(a) * 0.006) or abs(abs(a) - abs(v)) <= 0.051 for a in allowed)
        if not ok and abs(v) > 0.05:  # 두 계산값의 차이를 인용한 경우도 인정
            ok = any(i != j and abs(abs(a - b) - abs(v)) <= 0.051 for i, a in enumerate(allowed) for j, b in enumerate(allowed))
        checked.append(raw)
        if not ok:
            bad.append(raw)
    return {"total": len(checked), "ok": len(checked) - len(bad), "bad": list(dict.fromkeys(bad))}


def template_report(S: dict) -> str:
    """AI 없이 요약만으로 만드는 기본 리포트"""
    L = []
    ev = S.get("이벤트") or []
    c = S["이벤트_수"]
    L.append("## 요약")
    L.append(f"- {S['데이터']['기간'] or ''} 동안 {S['데이터']['분석_행수']}행을 분석해 이상 이벤트 {c['전체']}건(위험 {c['위험']}, 경고 {c['경고']}, 주의 {c['주의']})을 찾았어요.")
    for f in S["추세_예측"]:
        L.append(f"- **{f['변수']}**는 시간당 {f['시간당_변화']}씩 변하고 있어, 이 추세가 이어지면 약 {f['기준_도달까지_시간']}시간 뒤 \"{f['기준']}\" 기준에 도달할 수 있어요.")
    L += ["", "## 주요 이상 이벤트"]
    for e in ev[:5]:
        rule = (", 규칙 위반: " + ", ".join(e["규칙_위반"])) if e["규칙_위반"] else ""
        L.append(f"- **[{e['id']}] {e['등급']} · {e['유형']}** — {e['주요_변수']}, {e['시작']} ~ {e['끝']}. 가장 벗어난 값 {e['가장_벗어난_값']} "
                 f"(정상범위 {e['정상범위'][0]}~{e['정상범위'][1]}, {e['이탈_정도_시그마']}σ){rule}.")
    L += ["", "## 가능한 원인"]
    seen = set()
    for e in ev[:5]:
        if e["유형"] in seen:
            continue
        seen.add(e["유형"])
        hint = next(v["hint"] for v in TYPE_INFO.values() if v["name"] == e["유형"])
        L.append(f"- {e['유형']}: {hint}")
    L += ["", "## 권장 조치"]
    if any(e["규칙_위반"] for e in ev):
        L.append("- 규칙 위반 구간의 제품은 출하 전에 격리하고 기록을 확인하세요.")
    if S["추세_예측"]:
        L.append("- 추세가 이어지는 설비는 기준에 닿기 전에 점검 일정을 잡으세요.")
    L.append("- 변동성이 커진 센서는 교정하거나 예비 센서 값과 비교하세요.")
    L += ["", "## 한계", "- 통계적 이상은 원인을 확정하지 않아요. 현장 기록과 함께 판단해야 해요."]
    return "\n".join(L)


# ---------------------------------------------------------------- 한 번에 실행 (페이지에서 쓰기 쉽게)
def run_engine(table, columns=None, pack=None, k=1.0, rules=None, mult=None, dismissed=None, dataset_name=""):
    """CSV 표 하나를 넣으면 점검 → 준비 → 4가지 점수 → 판정·이벤트 → 추세 → (정답 열 있으면) 성능까지 한 번에.

    table   : load_csv()의 결과 또는 pandas DataFrame
    columns : 분석할 열 이름 목록 (없으면 자동 추천)
    pack    : 'phm' / 'food' / 'general' (없으면 열 이름으로 추정). 팀 페이지는 보통 'general' + 자기 규칙(rules)
    rules   : [{"id": "r1", "col": "열 이름", "op": ">", "value": 5, "durationMin": 0, "label": "설명", "source": "출처"}]
    """
    if hasattr(table, "columns") and hasattr(table, "itertuples"):
        table = table_from_dataframe(table)
    prof = profile(table)
    selected = columns or recommend_columns(prof)
    if not selected:
        raise ValueError("분석할 숫자 열이 없어요. 값이 변하는 숫자 열이 하나 이상 있어야 해요.")
    pack_id = pack or guess_pack(prof)
    if rules is None:
        rules = pack_rules(pack_id, prof)
    prep = prepare(table, prof, selected)
    if len(prep["idx"]) < 10:
        raise ValueError("행이 너무 적어요. 최소 10행 이상이 필요해요.")
    base = analyze_base(prep)
    res = classify(prep, base, rules, k, mult=mult, dismissed=dismissed)
    fc = forecasts(prep, base, rules)
    ev = evaluate(prep, base, rules, k, mult=mult, dismissed=dismissed)
    excluded = [c["name"] for c in prof["cols"] if c["role"] == "personal"]
    summary = build_summary(prep, base, res, fc, rules, PACKS[pack_id], dataset_name, excluded)
    return {"profile": prof, "prep": prep, "base": base, "result": res, "forecasts": fc, "evaluation": ev,
            "rules": rules, "pack": pack_id, "summary": summary}


def events_table(out) -> list:
    """이벤트 목록을 표(st.dataframe에 바로 넣을 수 있는 dict 목록)로"""
    sev_ko = {"critical": "위험", "serious": "경고", "warning": "주의"}
    prep = out["prep"]
    with_sec = is_num(prep["intervalMs"]) and prep["intervalMs"] < 60000

    def t(i):
        return fmt_time(prep["times"][i], with_sec) if prep["times"] else f"행 {prep['rowNums'][i]}"

    rows = []
    for e in out["result"]["events"]:
        c = prep["cols"][e["main"]]
        rows.append({"id": e["id"], "등급": sev_ko[e["sev"]], "유형": e["typeName"], "주요 변수": c["display"],
                     "시작": t(e["start"]), "끝": t(e["end"]), "행 수": e["len"], "가장 벗어난 값": round_to(e["peakVal"], 2),
                     "정상 범위": f"{round_to(c['band'][0], 2)} ~ {round_to(c['band'][1], 2)}", "이탈(σ)": round_to(e["devSigma"], 1),
                     "이상 점수": e["peakScore"], "규칙 위반": ", ".join(e["rules"]), "가능한 원인": e["hint"]})
    return rows


def scores_table(out) -> list:
    """행마다 4가지 점수와 판정 (그래프 그릴 때 사용)"""
    prep, res = out["prep"], out["result"]
    with_sec = is_num(prep["intervalMs"]) and prep["intervalMs"] < 60000
    rows = []
    for r in res["rows"]:
        i = r["i"]
        row = {"순서": i, "원래 행": prep["rowNums"][i], "시각": fmt_time(prep["times"][i], with_sec) if prep["times"] else ""}
        for c in prep["cols"]:
            row[c["display"]] = c["values"][i] if is_num(c["values"][i]) else None
        row.update({"튀는값": round_to(r["m"]["point"], 3), "관리도": round_to(r["m"]["ewma"], 3), "변동성": round_to(r["m"]["vari"], 3),
                    "다변량": round_to(r["m"]["mv"], 3), "이상": r["flagged"], "등급": r["sev"], "이벤트": ",".join(r.get("events", []))})
        rows.append(row)
    return rows
