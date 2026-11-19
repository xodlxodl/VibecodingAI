"""엔진 실험실 (담당: 수학과 B)
내 CSV 하나를 올리면, 그 파일이 공용 엔진에서 제대로 돌아가는지 점검해요.
1) CSV 입력 → 2) 탭별 점검(기본 실행 · 데이터 진단 · 이상 심기 240회 · 기준값 k · robust Z 대조 · 숫자 검증기) → 3) 종합 판정(AI 리포트).
모든 점검은 올린 파일의 실제 값으로 그 자리에서 계산해요. 같은 파일·같은 설정이면 저장해 둔 결과를 다시 써요.
AI 리포트: 점검 요약(원본 행 없음)만 AI에게 보내고, 리포트 속 숫자를 verify_numbers로 대조해 틀리면 최대 2번 다시 쓰게 해요.
API 키는 Streamlit Secrets의 ANTHROPIC_API_KEY에 넣어요(코드·저장소에 넣지 않음). 키가 없으면 기본 판정과 '가짜 AI'로 흐름만 시험할 수 있어요.
"""
import datetime
import hashlib
import json
import math
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from engine import check_table, events_table, load_csv, profile, report_text, run_engine, verify_numbers  # noqa: E402

st.set_page_config(page_title="엔진 실험실", layout="wide")

# ---------------------------------------------------------------- 설정
SAMPLES = {"3D 프린터 센서 (시계열)": "printer.csv", "식품 공정 (시계열)": "food.csv", "일반 표": "generic.csv"}
TS_WINDOW = 600            # 이상 심기 실험에 쓰는 구간 길이(시계열)
TABLE_SAMPLE = 1500        # 이상 심기 실험에 쓰는 표본 행 수(표)
REPS = 20                  # 칸마다 반복 수 → 4종류 × 3크기 × 20 = 240회(시계열), 3종류 × 4크기 × 20 = 240회(표)
TS_KINDS = ["튀는 값", "수준 이동", "변동성 증가", "서서히 흐름"]
TS_SIZES = [2, 4, 6]
TABLE_KINDS = ["높은 쪽 극단값", "낮은 쪽 극단값", "자릿수 실수"]
TABLE_SIZES = {"높은 쪽 극단값": [2, 3, 4, 6], "낮은 쪽 극단값": [2, 3, 4, 6], "자릿수 실수": ["×10", "×100", "÷10", "÷100"]}
K_GRID = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
BIG_ROWS = 50000
CODE_NAME_RE = re.compile(r"코드|번호|구분|ID|Id|id|code|Code|no\.?$|No\.?$")
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#9a9890"

AI_URL = "https://api.anthropic.com/v1/messages"
AI_DEFAULT_MODEL = "claude-haiku-5-5"   # Secrets의 AI_MODEL로 바꿀 수 있음
AI_MAX_RETRIES = 2                      # 틀린 숫자 다시 쓰기 최대 2번
AI_CALL_LIMIT = 20                      # 접속(세션)당 호출 상한
AI_GLOBAL_LIMIT = 200                   # 앱 전체(모든 접속 합계) 하루 호출 상한 — 새로고침으로 세션 상한을 우회하는 것 막기
AI_MAX_TOKENS = 3000                    # 5개 절 한국어 리포트가 잘리지 않을 만큼
AI_SYSTEM = (
    "너는 이상 탐지 엔진이 특정 데이터 파일에서 제대로 작동하는지 비전문가에게 판정해 주는 한국어 리포트 작성자다. "
    "사용자가 주는 JSON 점검 요약에 있는 숫자만 쓰고, 새 숫자를 계산하거나 지어내지 마라. "
    "점검 결과는 심어 본 이상과 통계 대조에 근거한 것이며 실제 이상의 원인을 확정하지 않는다는 한계를 반드시 적어라. "
    "형식: ## 한 줄 판정 / ## 잘 되는 것 / ## 주의할 것 / ## 권장 설정 / ## 한계 (마크다운)."
)
AI_USER = "다음은 내 CSV 파일을 공용 이상 탐지 엔진으로 점검한 결과 요약이다. 원본 데이터는 없다.\n\n```json\n{summary}\n```\n\n이 요약만으로 엔진이 이 파일에서 믿을 만한지 판정하는 리포트를 써라."
AI_FIX = "다음 숫자는 요약에 없다: {bad}. 요약에 있는 숫자만 써서 리포트 전체를 다시 써라."


# ================================================================ 1. 파일 읽기 · 엔진 실행 (저장해 두고 다시 씀)
@st.cache_resource(max_entries=4, show_spinner=False)
def file_profile(data: bytes):
    table = load_csv(data)
    return table, profile(table)


@st.cache_resource(max_entries=4, show_spinner=False)
def analyze_file(data: bytes, mode: str, k: float, columns: tuple, name: str):
    t0 = time.time()
    table, _ = file_profile(data)
    out = run_engine(table, columns=list(columns) or None, mode=mode, k=k, dataset_name=name)
    return {"table": table, "out": out, "seconds": round(time.time() - t0, 1)}


def numeric_options(prof):
    """분석 열로 고를 수 있는 열(값이 숫자인 열)"""
    return [c["name"] for c in prof["cols"] if c.get("stats") and c["role"] != "time"]


# ================================================================ 2-1. 데이터 진단
def _looks_like_code(table, col_name):
    """앞자리 0(010, 007)이 많거나 이름에 코드·번호·구분·ID가 있으면 '이름표 같은 숫자'일 수 있음"""
    ci = table["columns"].index(col_name)
    vals = [str(r[ci]).strip() for r in table["rows"][:5000] if str(r[ci]).strip()]
    lead0 = sum(1 for v in vals if re.match(r"^0\d", v)) / max(1, len(vals))
    return lead0 >= 0.3 or bool(CODE_NAME_RE.search(col_name)), lead0


def diagnose(res, requested_mode):
    """파일이 엔진에 맞는지 항목별로 점검. 상태: 좋음 / 주의 / 문제"""
    table, out, prof = res["table"], res["out"], res["out"]["profile"]
    n, mode, items = prof["n"], out["mode"], []

    def add(name, status, detail, advice="", key="", **extra):
        items.append(dict({"항목": name, "상태": status, "내용": detail, "권장": advice, "_key": key}, **extra))

    add("파일 읽기", "좋음", f"{n:,}행 × {len(table['columns'])}열 · 글자 인코딩 {table.get('encoding', '?')}", key="read")
    role_ko = {"numeric": "숫자", "time": "시간", "personal": "개인정보", "label": "정답 라벨", "counter": "번호·순번",
               "constant": "값이 하나뿐", "category": "범주", "text": "글자"}
    skipped = [f"{c['name']}({role_ko.get(c['role'], c['role'])})" for c in prof["cols"]
               if c["name"] not in out["columns"] and c["role"] != "time"]
    add("분석한 열", "좋음", f"{len(out['columns'])}개: {', '.join(out['columns'])}",
        ("분석에서 뺀 열: " + ", ".join(skipped[:8]) + (" 외" if len(skipped) > 8 else "")) if skipped else "", key="cols")

    # 모드가 데이터 모양과 맞는지
    tinfo = prof["time"]
    if mode == "timeseries" and tinfo:
        same = tinfo["dupStamps"] / max(1, n)
        if same >= 0.2:
            per = n / max(1, n - tinfo["dupStamps"])
            add("모드와 데이터 모양", "문제",
                f"시간 열({tinfo['col']})이 있어 시계열로 돌렸지만, 같은 시각에 평균 {per:.1f}행이 있어요(같은 시각 비율 {same:.0%}).",
                "한 행이 한 사람·한 건인 기록으로 보여요. 표 모드로 돌리거나, 날짜별로 모은 값(건수·평균)을 시계열로 보세요.",
                key="mode_shape")
        else:
            add("모드와 데이터 모양", "좋음", f"시간 열 {tinfo['col']} 기준 시계열 · 같은 시각 비율 {same:.0%}", key="mode")
    elif mode == "timeseries":
        add("모드와 데이터 모양", "좋음", "시간 열이 없어 행 순서를 시간으로 봤어요(직접 '시계열'을 고름).", key="mode")
    else:
        note = "시간 열이 없어 자동으로 표 모드가 됐어요." if requested_mode == "auto" else "표 모드로 돌렸어요."
        add("모드와 데이터 모양", "좋음", note + " 행마다 따로 이상치를 찾고 이벤트로 묶지 않아요.",
            "센서처럼 행 순서가 시간이라면 '시계열'을 고르세요." if requested_mode == "auto" else "", key="mode")

    # 이름표 같은 숫자 열
    codes, code_names = [], []
    for name in out["columns"]:
        is_code, lead0 = _looks_like_code(table, name)
        if is_code:
            code_names.append(name)
            codes.append(f"{name}" + (f"(앞자리 0 {lead0:.0%})" if lead0 >= 0.3 else ""))
    if codes:
        add("이름표 같은 숫자 열", "주의", "숫자 크기에 뜻이 없을 수 있는 열을 숫자로 분석했어요: " + ", ".join(codes),
            "센터·지역·상품 코드라면 위의 '분석할 열'에서 빼세요.", key="code_cols", _cols=code_names)
    else:
        add("이름표 같은 숫자 열", "좋음", "코드·번호처럼 보이는 열은 분석하지 않았어요.", key="code_cols")

    # 중복 행
    dup = prof["duplicates"]
    if mode == "timeseries" and dup / max(1, n) > 0.01:
        add("중복 행", "주의", f"완전히 같은 행 {dup:,}개({dup / n:.1%})를 빼고 분석했어요.",
            "같은 값의 서로 다른 사람·건이라면 빼면 안 돼요. 표 모드는 중복을 빼지 않아요.", key="dup")
    else:
        add("중복 행", "좋음", f"완전히 같은 행 {dup:,}개" + (" (표 모드라 빼지 않음)" if mode == "table" and dup else ""), key="dup")

    # 빈칸
    miss = [(c["name"], c["missing"] / max(1, n)) for c in prof["cols"] if c["name"] in out["columns"]]
    worst = max(miss, key=lambda x: x[1]) if miss else ("", 0)
    if worst[1] >= 0.5:
        add("빈칸", "문제", f"{worst[0]} 열의 {worst[1]:.0%}가 비어 있어요.", "빈칸이 많은 열은 분석 열에서 빼세요.",
            key="missing", _cols=[w for w, r in miss if r >= 0.5])
    elif worst[1] >= 0.2:
        add("빈칸", "주의", f"{worst[0]} 열의 {worst[1]:.0%}가 비어 있어요.", "시계열은 빈칸을 앞뒤 값으로 채워서 봐요.",
            key="missing", _cols=[w for w, r in miss if r >= 0.2])
    else:
        add("빈칸", "좋음", f"분석 열의 빈칸 비율 최대 {worst[1]:.1%}", key="missing")

    # 잡힌 비율
    rows = out["result"]["rows"]
    flagged = sum(1 for r in rows if r["flagged"])
    ratio = flagged / max(1, len(rows))
    if mode == "timeseries":
        n_ev = len(out["result"]["events"])
        per1k = 1000 * n_ev / max(1, len(rows))
        detail = f"이벤트 {n_ev:,}건(1,000행당 {per1k:.1f}건), 이상으로 표시된 행 {flagged:,}개({ratio:.1%})"
        too_many = per1k > 10 or ratio > 0.5   # 이벤트는 여러 행에 걸치므로 행 비율보다 이벤트 수로 봄
    else:
        detail = f"확인이 필요한 행 {flagged:,}개({ratio:.1%})"
        too_many = ratio > 0.05
    if too_many:
        add("잡힌 양", "주의", detail + "로 많은 편이에요.",
            "정상 데이터에서도 많이 잡히면 결과를 믿기 어려워요. '기준값 k' 탭에서 k를 키운 결과를 보세요.", key="flags")
    else:
        add("잡힌 양", "좋음", detail, key="flags")

    # 기준 구간 대표성(시계열): 앞 30%와 나머지의 수준 차이
    if mode == "timeseries":
        nb = out["base"]["nb"]
        moved = []
        for c in out["prep"]["cols"]:
            rest = np.array(c["filled"][nb:], dtype=float)
            if len(rest) and c["sigma"] > 0:
                shift = abs(np.nanmedian(rest) - c["mu"]) / c["sigma"]
                if shift >= 2:
                    moved.append(f"{c['name']}({shift:.1f}σ)")
        if moved:
            add("기준 구간", "주의", f"엔진은 앞쪽 {nb:,}행을 정상 기준으로 삼는데, 나머지 구간의 수준이 크게 달라요: " + ", ".join(moved),
                "앞부분이 정상 운전이 아니면 이벤트가 많이 나와요. 정상 구간이 앞에 오도록 파일을 자르세요.", key="baseline")
        else:
            add("기준 구간", "좋음", f"앞쪽 {nb:,}행을 정상 기준으로 삼았고, 나머지 구간과 수준 차이가 크지 않아요.", key="baseline")

    if n < 100:
        add("행 수", "주의", f"{n}행뿐이라 판정이 흔들릴 수 있어요.", "가능하면 수백 행 이상을 넣으세요.", key="rows")
    if res["seconds"] > 20:
        add("실행 시간", "주의", f"한 번 분석에 {res['seconds']}초 걸렸어요.", f"{BIG_ROWS:,}행이 넘으면 필요한 기간·열만 잘라서 넣는 게 좋아요.", key="time")
    else:
        add("실행 시간", "좋음", f"한 번 분석에 {res['seconds']}초", key="time")
    return items


# ================================================================ 2-2. 이상 심기 240회 (내 파일의 실제 값 위에)
def _overlap(a, b):
    return a[0] <= b[1] and b[0] <= a[1]


def _robust_sigma(y):
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if not len(y):
        return 1.0
    s = 1.4826 * np.median(np.abs(y - np.median(y)))
    if not s > 0:
        s = np.std(y, ddof=1) if len(y) > 1 else 0.0
    return float(s) if s > 0 else max(abs(float(np.median(y))) * 1e-3, 1e-6)


def _ts_segments(df, k):
    o = run_engine(df, columns=list(df.columns), mode="timeseries", k=k)
    rn = o["prep"]["rowNums"]
    return [(rn[e["start"]] - 2, rn[e["end"]] - 2) for e in o["result"]["events"]]


def ts_window(out):
    """정상 기준이 오염되지 않게, 이상 표시가 가장 적은 연속 구간을 실험 창으로 고름"""
    prep = out["prep"]
    n = len(prep["idx"])
    N = min(n, TS_WINDOW)
    flags = np.array([1 if r["flagged"] else 0 for r in out["result"]["rows"]], dtype=float)
    start = 0
    if n > N:
        sums = np.convolve(flags, np.ones(N), mode="valid")
        start = int(np.argmin(sums))
    cols = {c["name"]: np.asarray(c["filled"][start:start + N], dtype=float) for c in prep["cols"]}
    return start, pd.DataFrame(cols)


def _inject_ts(y, kind, size, sigma, s, L, rng):
    y = y.copy()
    sign = rng.choice([-1, 1])
    if kind == "튀는 값":
        y[s] += sign * size * sigma
    elif kind == "수준 이동":
        y[s:s + L] += sign * size * sigma
    elif kind == "변동성 증가":
        y[s:s + L] += rng.normal(0, size * sigma, L)
    else:
        y[s:s + L] += sign * np.linspace(0, size * sigma, L)
    return y


def ts_trials(win, target, k, kinds, sizes, reps):
    """시계열: 실험 창의 대상 열에 이상을 하나씩 심고 엔진을 다시 돌려 잡는지 봄"""
    N = len(win)
    base_ev = _ts_segments(win, k)
    sigma = _robust_sigma(win[target].values)
    L_long = max(5, min(40, N // 10))
    recs = []
    for ki, kind in enumerate(kinds):
        L = 1 if kind == "튀는 값" else L_long
        for si, size in enumerate(sizes):
            for rep in range(reps):
                rng = np.random.default_rng(10000 * ki + 100 * si + rep)
                lo, hi = int(N * 0.35), N - L - 1
                s = int(rng.integers(lo, hi)) if hi > lo else lo
                for _ in range(50):   # 원래 있던 이벤트와 겹치지 않는 자리에 심음
                    if not any(_overlap((s, s + L - 1), b) for b in base_ev):
                        break
                    s = int(rng.integers(lo, hi)) if hi > lo else lo
                df = win.copy()
                df[target] = _inject_ts(win[target].values, kind, size, sigma, s, L, rng)
                ev = _ts_segments(df, k)
                seg = (s, s + L - 1)
                hits = [e for e in ev if _overlap(e, seg)]
                fp = sum(1 for e in ev if not _overlap(e, seg) and not any(_overlap(e, b) for b in base_ev))
                recs.append({"종류": kind, "크기": size, "반복": rep, "시작": s, "길이": L, "잡음": bool(hits), "오탐": fp,
                             "지연": (min(e[0] for e in hits) - s) if hits else None, "잡은 구간": ev,
                             "y": df[target].values if rep == 0 else None})   # 첫 번째 반복만 그래프용으로 보관
    return recs, base_ev, sigma


def table_sample(out):
    prep = out["prep"]
    n = len(prep["idx"])
    pick = np.sort(np.random.default_rng(0).choice(n, size=min(n, TABLE_SAMPLE), replace=False))
    data = {c["name"]: np.asarray(c["values"], dtype=float)[pick] for c in prep["cols"]}
    return pd.DataFrame(data)


def _table_flags(df, k):
    o = run_engine(df, columns=list(df.columns), mode="table", k=k)
    return {r["i"] for r in o["result"]["outliers"]}


def table_trials(sample, target, k, kinds, sizes_of, reps):
    """표: 표본의 대상 열에서 한 행의 값을 극단값·자릿수 실수로 바꾸고 엔진이 그 행을 잡는지 봄"""
    base = _table_flags(sample, k)
    y = sample[target].values
    fin = y[np.isfinite(y)]
    med, sigma, nonneg = float(np.median(fin)), _robust_sigma(fin), bool(len(fin) and fin.min() >= 0)
    ok_rows = [i for i in range(len(y)) if np.isfinite(y[i]) and i not in base]
    recs = []
    for ki, kind in enumerate(kinds):
        for si, size in enumerate(sizes_of[kind]):
            if kind == "낮은 쪽 극단값" and nonneg and med - size * sigma < 0:
                recs += [{"종류": kind, "크기": size, "반복": r, "해당 없음": True} for r in range(reps)]
                continue
            for rep in range(reps):
                rng = np.random.default_rng(10000 * ki + 100 * si + rep)
                rows = [i for i in ok_rows if y[i] != 0] if kind == "자릿수 실수" else ok_rows
                if not rows:
                    recs.append({"종류": kind, "크기": size, "반복": rep, "해당 없음": True})
                    continue
                i = int(rng.choice(rows))
                new = y.copy()
                if kind == "높은 쪽 극단값":
                    new[i] = med + size * sigma
                elif kind == "낮은 쪽 극단값":
                    new[i] = med - size * sigma
                else:
                    f = {"×10": 10, "×100": 100, "÷10": 0.1, "÷100": 0.01}[size]
                    new[i] = y[i] * f
                df = sample.copy()
                df[target] = new
                flags = _table_flags(df, k)
                recs.append({"종류": kind, "크기": size, "반복": rep, "행": i, "원래 값": float(y[i]), "바꾼 값": float(new[i]),
                             "잡음": i in flags, "오탐": len(flags - base - {i}), "해당 없음": False})
    return recs, base, sigma


@st.cache_resource(max_entries=4, show_spinner=False)
def run_trials(data, mode, k, columns, name, target):
    out = analyze_file(data, mode, k, columns, name)["out"]
    t0 = time.time()
    if out["mode"] == "timeseries":
        start, win = ts_window(out)
        recs, base_ev, sigma = ts_trials(win, target, k, TS_KINDS, TS_SIZES, REPS)
        info = {"창 시작 행": start, "창 길이": len(win), "원래 이벤트": len(base_ev), "σ": sigma}
        return {"mode": "timeseries", "recs": recs, "win": win, "base": base_ev, "info": info, "seconds": round(time.time() - t0, 1)}
    sample = table_sample(out)
    recs, base, sigma = table_trials(sample, target, k, TABLE_KINDS, TABLE_SIZES, REPS)
    info = {"표본 행": len(sample), "원래 이상치": len(base), "σ": sigma}
    return {"mode": "table", "recs": recs, "win": sample, "base": base, "info": info, "seconds": round(time.time() - t0, 1)}


def trial_scores(recs):
    done = [r for r in recs if not r.get("해당 없음")]
    tp = sum(r["잡음"] for r in done)
    fp = sum(r["오탐"] for r in done)
    big = [r for r in done if (isinstance(r["크기"], (int, float)) and r["크기"] >= 4) or isinstance(r["크기"], str)]
    rec = tp / len(done) if done else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    return {"실험_수": len(done), "재현율": round(rec, 3), "정밀도": round(prec, 3),
            "F1": round(2 * rec * prec / (rec + prec), 3) if rec + prec else 0.0,
            "큰_이상_재현율": round(sum(r["잡음"] for r in big) / len(big), 3) if big else 0.0, "오탐_합계": fp}


def trial_grid(recs):
    rows = []
    for kind in dict.fromkeys(r["종류"] for r in recs):
        for size in dict.fromkeys(r["크기"] for r in recs if r["종류"] == kind):
            cell = [r for r in recs if r["종류"] == kind and r["크기"] == size]
            done = [r for r in cell if not r.get("해당 없음")]
            if not done:
                rows.append({"이상 종류": kind, "크기": f"{size}σ" if isinstance(size, (int, float)) else size,
                             "재현율(%)": None, "실험 수": 0, "평균 오탐": None, "평균 지연(행)": None, "비고": "음수가 돼서 해당 없음"})
                continue
            delays = [r["지연"] for r in done if r.get("지연") is not None]
            rows.append({"이상 종류": kind, "크기": f"{size}σ" if isinstance(size, (int, float)) else size,
                         "재현율(%)": round(100 * sum(r["잡음"] for r in done) / len(done)),
                         "실험 수": len(done), "평균 오탐": round(float(np.mean([r["오탐"] for r in done])), 2),
                         "평균 지연(행)": round(float(np.median(delays)), 1) if delays else None, "비고": ""})
    return rows


# ================================================================ 2-3. 기준값 k
@st.cache_resource(max_entries=4, show_spinner=False)
def run_k_sweep(data, mode, k, columns, name, target):
    """k마다: 심은 이상(크기 4)을 몇 개 잡나 + 실험 구간 원래 데이터에서 몇 개 표시되나"""
    out = analyze_file(data, mode, k, columns, name)["out"]
    pts = []
    if out["mode"] == "timeseries":
        _, win = ts_window(out)
        for kk in K_GRID:
            recs, base_ev, _ = ts_trials(win, target, kk, TS_KINDS, [4], 5)
            sc = trial_scores(recs)
            pts.append({"k": kk, "재현율": sc["재현율"], "정밀도": sc["정밀도"], "F1": sc["F1"], "원래 데이터 표시 수": len(base_ev)})
    else:
        sample = table_sample(out)
        sizes = {"높은 쪽 극단값": [4], "자릿수 실수": ["×10"]}
        for kk in K_GRID:
            recs, base, _ = table_trials(sample, target, kk, list(sizes), sizes, 10)
            sc = trial_scores(recs)
            pts.append({"k": kk, "재현율": sc["재현율"], "정밀도": sc["정밀도"], "F1": sc["F1"], "원래 데이터 표시 수": len(base)})
    best = max(pts, key=lambda p: (p["F1"], -abs(p["k"] - 1.0)))
    return pts, best


# ================================================================ 2-4. robust Z 손 재현
def robust_check(out):
    """엔진이 낸 중앙값·퍼짐·Z를 numpy로 따로 계산해서 맞춰 봄"""
    rows, examples = [], []
    if out["mode"] == "table":
        for c in out["prep"]["cols"]:
            v = np.asarray(c["values"], dtype=float)
            fin = v[np.isfinite(v)]
            if c.get("discrete"):
                rows.append({"열": c["name"], "척도": "-", "손 계산 중앙값": None, "엔진 중앙값": None, "손 계산 σ": None,
                             "엔진 σ": None, "Z 최대 차이": None, "일치": "건너뜀(절반 넘게 같은 값 · 엔진도 Z를 안 씀)"})
                continue
            log = c.get("scale") == "log"
            shift = 0.0 if (not log or fin.min() > 0) else 1.0
            t = np.log(v + shift) if log else v
            m = float(np.nanmedian(t))
            s = 1.4826 * float(np.nanmedian(np.abs(t - m)))
            if not s > 0:
                s = float(np.nanstd(t, ddof=1))
            z_hand = (t - m) / s
            z_eng = np.asarray(c["z"], dtype=float)
            diff = float(np.nanmax(np.abs(z_hand - z_eng))) if np.isfinite(z_eng).any() else float("nan")
            mu_hand = math.exp(m) - shift if log else m
            ok = diff < 1e-6 and abs(mu_hand - c["mu"]) < 1e-6 * max(1, abs(c["mu"])) and abs(s - c["sigma"]) < 1e-6 * max(1, s)
            rows.append({"열": c["name"], "척도": "로그" if log else "원래", "손 계산 중앙값": round(mu_hand, 4), "엔진 중앙값": round(c["mu"], 4),
                         "손 계산 σ": round(s, 4), "엔진 σ": round(c["sigma"], 4), "Z 최대 차이": diff, "일치": "일치" if ok else "다름"})
        cols = out["prep"]["cols"]
        for r in out["result"]["outliers"][:5]:
            for j in r["cause"][:1]:
                c = cols[j]
                if c.get("discrete") or not math.isfinite(c["values"][r["i"]]):
                    continue
                examples.append({"원래 행": out["prep"]["rowNums"][r["i"]], "열": c["name"], "값": c["values"][r["i"]],
                                 "척도": "로그" if c.get("scale") == "log" else "원래", "엔진 Z": round(c["z"][r["i"]], 3)})
    else:
        nb = out["base"]["nb"]
        for c in out["prep"]["cols"]:
            x = np.asarray(c["filled"][:nb], dtype=float)
            m = float(np.nanmedian(x))
            s = 1.4826 * float(np.nanmedian(np.abs(x - m)))
            if not s > 0:
                s = float(np.nanstd(x, ddof=1))
            ok = abs(m - c["mu"]) < 1e-6 * max(1, abs(m)) and abs(s - c["sigma"]) < 1e-6 * max(1, s)
            rows.append({"열": c["name"], "척도": f"앞 {nb}행", "손 계산 중앙값": round(m, 4), "엔진 중앙값": round(c["mu"], 4),
                         "손 계산 σ": round(s, 4), "엔진 σ": round(c["sigma"], 4),
                         "정상범위(손)": f"{m - 3 * s:.2f} ~ {m + 3 * s:.2f}", "정상범위(엔진)": f"{c['band'][0]:.2f} ~ {c['band'][1]:.2f}",
                         "일치": "일치" if ok else "다름"})
        for e in out["result"]["events"][:5]:
            c = out["prep"]["cols"][e["main"]]
            examples.append({"이벤트": e["id"], "열": c["name"], "가장 벗어난 값": e["peakVal"],
                             "손 계산 Z": round((e["peakVal"] - c["mu"]) / c["sigma"], 2)})
    return rows, examples


# ================================================================ 2-5. 숫자 검증기 (내 파일의 기본 리포트로)
NUM_RE = re.compile(r"(?<![\w.])(\d+\.\d+|\d{2,})(?![\w.])")


def verifier_check(out):
    base = report_text(out)
    s_text = json.dumps(out["summary"], ensure_ascii=False)
    clean = verify_numbers(base, s_text)
    cases = []
    for m in list(NUM_RE.finditer(base))[:40]:
        v = float(m.group(1))
        wrong = round(v * 1.37 + 1, 2)
        tampered = base[:m.start(1)] + str(wrong) + base[m.end(1):]
        caught = bool(verify_numbers(tampered, s_text)["bad"])
        cases.append({"원래 숫자": m.group(1), "바꾼 숫자": str(wrong), "잡음": caught,
                      "앞뒤 글": base[max(0, m.start(1) - 18):m.end(1) + 10].replace("\n", " ")})
    return {"clean": clean, "cases": cases, "report": base}


# ================================================================ 3. 종합 판정
def fitness_summary(res, diag, trials, kpts, kbest, rz_rows, ver, k):
    out = res["out"]
    sc = trial_scores(trials["recs"])
    grid, fpgrid = {}, {}
    for row in trial_grid(trials["recs"]):
        if row["재현율(%)"] is not None:
            grid.setdefault(row["이상 종류"], {})[row["크기"]] = row["재현율(%)"]
            fpgrid.setdefault(row["이상 종류"], {})[row["크기"]] = row["평균 오탐"]
    compared = [r for r in rz_rows if r["일치"] in ("일치", "다름")]
    caught = sum(c["잡음"] for c in ver["cases"])
    return {
        "파일": {"행": out["profile"]["n"], "열": len(res["table"]["columns"]), "모드": "시계열" if out["mode"] == "timeseries" else "표",
               "분석_열": out["columns"], "실행_초": res["seconds"]},
        "데이터_진단": [{"항목": d["항목"], "상태": d["상태"], "내용": d["내용"], "권장": d["권장"]} for d in diag],
        "이상_심기": {"대상_열": trials["target"], "실험_수": sc["실험_수"], "재현율_퍼센트": round(100 * sc["재현율"], 1),
                  "큰_이상_재현율_퍼센트": round(100 * sc["큰_이상_재현율"], 1), "정밀도_퍼센트": round(100 * sc["정밀도"], 1),
                  "종류별_크기별_재현율_퍼센트": grid, "종류별_크기별_평균_새_오탐": fpgrid},
        "기준값_k": {"지금_k": k, "F1이_가장_높은_k": kbest["k"], "그때_F1": kbest["F1"],
                  "k별": [{"k": p["k"], "재현율": p["재현율"], "정밀도": p["정밀도"], "원래_데이터_표시_수": p["원래 데이터 표시 수"]} for p in kpts]},
        "robust_Z_대조": {"비교한_열": len(compared), "일치한_열": sum(r["일치"] == "일치" for r in compared)},
        "숫자_검증기": {"바꾼_숫자": len(ver["cases"]), "잡은_수": caught, "깨끗한_리포트에서_틀렸다고_한_숫자": len(ver["clean"]["bad"])},
    }


def grade_of(S):
    probs = [d for d in S["데이터_진단"] if d["상태"] == "문제"]
    warns = [d for d in S["데이터_진단"] if d["상태"] == "주의"]
    big = S["이상_심기"]["큰_이상_재현율_퍼센트"]
    rz_bad = S["robust_Z_대조"]["비교한_열"] - S["robust_Z_대조"]["일치한_열"]
    if probs or big < 50 or rz_bad:
        return "맞지 않음", probs, warns
    if warns or big < 80 or S["이상_심기"]["정밀도_퍼센트"] < 70:
        return "주의 필요", probs, warns
    return "잘 맞음", probs, warns


GRADE_SENTENCE = {"잘 맞음": "공용 엔진과 잘 맞아요. 결과를 그대로 참고해도 돼요.",
                  "주의 필요": "공용 엔진으로 돌릴 수 있지만, 아래 '주의할 것'을 감안해서 결과를 읽어야 해요.",
                  "맞지 않음": "이대로는 공용 엔진 결과를 믿기 어려워요. 아래 '권장 설정'대로 고친 뒤 다시 돌려 보세요."}


def verdict_text(S):
    """AI 없이 점검 요약에서 바로 만드는 기본 판정"""
    g, probs, warns = grade_of(S)
    T, F, K = S["이상_심기"], S["파일"], S["기준값_k"]
    lines = ["## 한 줄 판정", f"- **{g}** — 이 파일은 {GRADE_SENTENCE[g]}",
             f"- {F['행']:,}행 · {F['모드']} 모드 · 분석 열 {len(F['분석_열'])}개({', '.join(F['분석_열'])})", ""]
    lines += ["## 잘 되는 것"]
    good = [d for d in S["데이터_진단"] if d["상태"] == "좋음"]
    lines += [f"- 크기가 큰(4σ 이상·자릿수 실수) 심은 이상을 {T['큰_이상_재현율_퍼센트']}% 잡았어요. 전체 {T['실험_수']}회 중 재현율 {T['재현율_퍼센트']}%, 정밀도 {T['정밀도_퍼센트']}%예요."]
    best = [(kind, size, v) for kind, d in T["종류별_크기별_재현율_퍼센트"].items() for size, v in d.items()]
    strong = [f"{kind} {size}" for kind, size, v in best if v >= 90]
    weak = [f"{kind} {size}({v}%)" for kind, size, v in best if v < 50 and (size.endswith("σ") and float(size[:-1]) >= 4)]
    if strong:
        lines.append("- 거의 다 잡는 경우: " + ", ".join(strong[:6]))
    rz = S["robust_Z_대조"]
    lines.append(f"- robust Z를 손으로 다시 계산해 보니 {rz['비교한_열']}개 열 중 {rz['일치한_열']}개가 엔진과 일치했어요.")
    lines += [f"- {d['항목']}: {d['내용']}" for d in good if d["항목"] in ("모드와 데이터 모양", "이름표 같은 숫자 열", "기준 구간")]
    noisy = [f"{kind} {size}(평균 {v}개)" for kind, d in T["종류별_크기별_평균_새_오탐"].items() for size, v in d.items() if v and v >= 1]
    lines += ["", "## 주의할 것"]
    if not probs and not warns and not weak and not noisy:
        lines.append("- 특별히 주의할 점이 보이지 않았어요.")
    lines += [f"- **{d['항목']}** ({d['상태']}): {d['내용']}" for d in probs + warns]
    if weak:
        lines.append("- 크기가 커도 잘 못 잡는 경우: " + ", ".join(weak))
    if noisy:
        lines.append("- 이런 값이 하나 들어가면 다른 정상 행까지 새로 표시돼요: " + ", ".join(noisy[:6])
                     + ". 값 하나 때문에 열 전체의 판정 기준(척도)이 바뀐 경우예요.")
    lines += ["", "## 권장 설정"]
    if K["F1이_가장_높은_k"] != K["지금_k"]:
        lines.append(f"- 이 파일에서는 k = {K['F1이_가장_높은_k']}일 때 F1이 가장 높았어요(F1 {K['그때_F1']}, 지금 k = {K['지금_k']}).")
    else:
        lines.append(f"- 지금 k = {K['지금_k']}일 때 이 파일에서 F1이 가장 높았어요(F1 {K['그때_F1']}).")
    advice = [d for d in S["데이터_진단"] if d["상태"] != "좋음" and d.get("권장")]
    lines += [f"- {d['권장']}" for d in advice]
    lines += ["", "## 한계",
              "- 이 판정은 파일의 실제 값 위에 이상을 심어 본 실험과 계산 대조에 근거해요. 파일에 원래 있던 이상의 원인은 알려 주지 않아요.",
              "- 심은 이상과 실제로 생기는 이상의 모양이 다를 수 있어요."]
    return "\n".join(lines)


# ================================================================ 3-1. 이렇게 바꿔 보세요 (판정이 나쁠 때 고칠 방법)
MODE_LABEL = {"auto": "자동", "timeseries": "시계열 (행 순서 = 시간)", "table": "표 (한 행 = 한 건)"}


@st.cache_resource(max_entries=4, show_spinner=False)
def date_aggregate(data: bytes):
    """같은 시각에 여러 행이 있는 기록을 시각별로 모음: 건수 + 숫자 열의 평균 (코드 같은 열은 뺌) → 새 CSV 바이트"""
    table, prof = file_profile(data)
    tcol = prof["time"]["col"]
    df = pd.DataFrame(table["rows"], columns=table["columns"])
    when = pd.to_datetime(df[tcol], errors="coerce")
    agg = pd.DataFrame({tcol: when})
    used = []
    for c in prof["cols"]:
        if c["role"] == "numeric" and not _looks_like_code(table, c["name"])[0]:
            agg[f"평균_{c['name']}"] = pd.to_numeric(df[c["name"]].astype(str).str.replace(",", ""), errors="coerce")
            used.append(c["name"])
    g = agg.dropna(subset=[tcol]).groupby(tcol)
    outdf = g.size().rename("건수").to_frame().join(g.mean(numeric_only=True)).reset_index()
    outdf[tcol] = outdf[tcol].dt.strftime("%Y-%m-%d %H:%M:%S")
    return outdf.round(4).to_csv(index=False).encode("utf-8"), used, len(outdf)


def suggest_fixes(diag, out, kbest, k, cols_now, agg_on):
    """진단 결과마다 고칠 방법을 붙여요.
    설정: 버튼 하나로 바꿔 다시 점검 · 데이터 바꾸기: 페이지가 데이터를 새로 만들어 보여 줌(질문이 바뀜) · 데이터 준비: 사용자가 해야 함"""
    bad = {d["_key"]: d for d in diag if d["상태"] != "좋음"}
    fixes, combo = [], {}
    cur_cols = list(cols_now) or list(out["columns"])
    if "mode_shape" in bad:
        fixes.append({"종류": "설정", "제목": "표 모드로 다시 점검", "apply": {"mode": "table"}, "짧게": "표 모드",
                      "이유": "한 행이 한 사람·한 건인 기록이라 행마다 따로 보는 표 모드가 맞아요. 같은 값의 행도 빼지 않아요."})
        combo["mode"] = "table"
        if not agg_on:
            fixes.append({"종류": "데이터 바꾸기", "제목": "시각별로 모아서 시계열로 보기", "apply": {"agg": True},
                          "이유": "같은 시각의 행을 모아 '건수'와 숫자 열의 평균으로 바꾼 뒤 시계열로 봐요. "
                                  "'이상한 사람'이 아니라 '평소와 다른 날'을 찾는 질문으로 바뀌어요."})
    elif "dup" in bad and out["mode"] == "timeseries":
        fixes.append({"종류": "설정", "제목": "표 모드로 다시 점검 (중복 행 유지)", "apply": {"mode": "table"}, "짧게": "표 모드",
                      "이유": "시계열 모드는 완전히 같은 행을 빼요. 서로 다른 사람·건이라면 표 모드로 봐야 해요."})
        combo["mode"] = "table"
    drop = []
    if "code_cols" in bad:
        drop += bad["code_cols"].get("_cols", [])
    if "missing" in bad:
        drop += bad["missing"].get("_cols", [])
    drop = list(dict.fromkeys(drop))
    keep = [c for c in cur_cols if c not in drop]
    if drop and keep:
        fixes.append({"종류": "설정", "제목": f"{', '.join(drop)} 빼고 다시 점검", "apply": {"cols": keep}, "짧게": f"{', '.join(drop)} 빼기",
                      "이유": "코드·번호처럼 숫자 크기에 뜻이 없거나 빈칸이 많은 열은 엔진 판정을 흐려요."})
        combo["cols"] = keep
    elif drop:
        fixes.append({"종류": "데이터 준비", "제목": "분석할 숫자 열이 있는 파일 준비",
                      "이유": f"문제 있는 열({', '.join(drop)})을 빼면 남는 숫자 열이 없어요. 값의 크기에 뜻이 있는 숫자 열이 필요해요."})
    if "flags" in bad:
        bigger = kbest["k"] if kbest["k"] > k else next((x for x in K_GRID if x > k), None)
        if bigger:
            fixes.append({"종류": "설정", "제목": f"k = {bigger}로 다시 점검", "apply": {"k": bigger}, "짧게": f"k {bigger}",
                          "이유": f"정상 데이터에서도 많이 잡혀요. k를 키우면 덜 민감해져요(지금 k = {k})."})
            combo["k"] = bigger
    if "baseline" in bad:
        fixes.append({"종류": "데이터 준비", "제목": "정상 구간이 앞에 오도록 파일 자르기",
                      "이유": "엔진은 앞쪽 30%를 정상 기준으로 삼아요. 고장·시운전 구간이 앞에 있으면 그 뒤가 전부 이상해 보여요."})
    if "rows" in bad:
        fixes.append({"종류": "데이터 준비", "제목": "행을 더 모으기", "이유": "행이 적으면 판정이 흔들려요. 수백 행 이상을 권해요."})
    if "missing" in bad and bad["missing"]["상태"] == "문제" and not keep:
        fixes.append({"종류": "데이터 준비", "제목": "빈칸을 채운 파일 준비", "이유": bad["missing"]["내용"]})
    if "mode" in combo and "k" in combo:
        combo.pop("k")   # 추천 k는 지금 모드에서 잰 값이라, 모드를 바꿀 때는 같이 묶지 않음
    parts = [f for f in fixes if f["종류"] == "설정" and set(f["apply"]) <= set(combo)]
    if len(parts) >= 2:
        fixes.insert(0, {"종류": "설정", "제목": "추천 설정 모두 적용해서 다시 점검", "apply": dict(combo),
                         "이유": " + ".join(f["짧게"] for f in parts)
                                 + (" (k는 모드를 바꾼 뒤 다시 점검해서 정하세요)" if any("k" in f["apply"] for f in fixes) and "mode" in combo else "")})
    return fixes


def apply_fix(change):
    """버튼 콜백: 위쪽 설정 위젯 값을 바꾼 뒤 페이지가 다시 돌아요(원래 판정은 점검 기록에 남음)."""
    if "mode" in change:
        st.session_state["mode_label"] = MODE_LABEL[change["mode"]]
    if "cols" in change:
        st.session_state["cols_pick"] = list(change["cols"])
    if "k" in change:
        st.session_state["k_val"] = change["k"]
    if change.get("agg") is True:
        st.session_state["agg_for"] = st.session_state.get("_orig_hash")
        if "mode" not in change:
            st.session_state["mode_label"] = MODE_LABEL["timeseries"]
        if "cols" not in change:
            st.session_state["cols_pick"] = []
    elif change.get("agg") is False:
        st.session_state.pop("agg_for", None)
    st.session_state.pop("target", None)


def leave_agg():
    st.session_state.pop("agg_for", None)
    st.session_state["mode_label"] = MODE_LABEL["auto"]
    st.session_state["cols_pick"] = []
    st.session_state.pop("target", None)


GRADE_RANK = {"잘 맞음": 2, "주의 필요": 1, "맞지 않음": 0}


def record_history(orig_hash, setting, grade, big, warns, apply):
    """이 파일로 해 본 설정과 판정을 쌓아 둠. _apply = 그 설정으로 돌아갈 때 쓸 값"""
    hist = st.session_state.setdefault("history", {}).setdefault(orig_hash, [])
    if not any(h["설정"] == setting for h in hist):
        hist.append({"순서": len(hist) + 1, "설정": setting, "판정": grade, "큰 이상 재현율": f"{big:.0%}", "진단 경고": warns,
                     "_big": big, "_apply": apply})
    return hist


def best_of(hist, agg_on):
    """원래 데이터끼리, 시각별 모음끼리만 비교 (둘은 묻는 질문이 달라서)"""
    same = [h for h in hist if h["_apply"]["agg"] == agg_on]
    return max(same, key=lambda h: (GRADE_RANK[h["판정"]], -h["진단 경고"], h["_big"]))


# ---------------------------------------------------------------- AI 호출 (비용 장치 포함)
def ai_settings():
    """Secrets에서 키와 모델을 읽어요. secrets.toml이 없으면 st.secrets 접근 자체가 오류라 감싸요."""
    try:
        return st.secrets.get("ANTHROPIC_API_KEY"), st.secrets.get("AI_MODEL", AI_DEFAULT_MODEL)
    except Exception:
        return None, AI_DEFAULT_MODEL


@st.cache_resource
def global_counter():
    """모든 접속이 함께 쓰는 호출 카운터. 앱이 다시 시작되거나 날짜가 바뀌면 0부터 다시 세요."""
    return {"lock": threading.Lock(), "day": None, "used": 0}


def _take_global_call():
    g = global_counter()
    with g["lock"]:
        today = datetime.date.today().isoformat()
        if g["day"] != today:
            g["day"], g["used"] = today, 0
        if g["used"] >= AI_GLOBAL_LIMIT:
            return False
        g["used"] += 1
        return True


def _give_back_global_call():
    g = global_counter()
    with g["lock"]:
        g["used"] = max(0, g["used"] - 1)


def claude_writer(key, model):
    """AI 호출 함수. 접속당 상한과 앱 전체 하루 상한을 지키고, 돈이 들지 않은 실패 호출은 횟수를 돌려줘요."""
    def write(messages):
        used = st.session_state.get("ai_calls", 0)
        if used >= AI_CALL_LIMIT:
            raise RuntimeError(f"이 접속에서 AI 호출 {AI_CALL_LIMIT}번을 모두 썼어요.")
        if not _take_global_call():
            raise RuntimeError(f"오늘 앱 전체 AI 호출 {AI_GLOBAL_LIMIT}번을 모두 썼어요. 내일 다시 쓸 수 있어요.")
        st.session_state["ai_calls"] = used + 1

        def give_back():
            st.session_state["ai_calls"] = max(0, st.session_state.get("ai_calls", 1) - 1)
            _give_back_global_call()

        body = json.dumps({"model": model, "max_tokens": AI_MAX_TOKENS, "system": AI_SYSTEM, "messages": messages}).encode("utf-8")
        req = urllib.request.Request(AI_URL, data=body, headers={
            "x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if 400 <= e.code < 500:
                give_back()
            raise RuntimeError(f"AI 호출 실패 (HTTP {e.code}): {e.read().decode('utf-8', 'ignore')[:200]}") from e
        except urllib.error.URLError as e:
            give_back()
            raise RuntimeError(f"AI 서버에 연결하지 못했어요: {e.reason}") from e
        if data.get("stop_reason") == "max_tokens":
            raise RuntimeError("AI 리포트가 길이 제한에 걸려 중간에 잘렸어요.")
        return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
    return write


def fake_writer(base_report, summary):
    """API 키 없이 흐름을 시험하는 가짜 AI: 첫 번째엔 숫자 하나를 틀리게 쓰고, 고쳐 달라고 하면 바르게 써요."""
    summary_text = json.dumps(summary, ensure_ascii=False)

    def write(messages):
        if len(messages) > 1:
            return base_report
        for m in NUM_RE.finditer(base_report):
            for factor in (1.1, 1.37, 2.9, 7.3):
                wrong = round(float(m.group(1)) * factor + 1, 2)
                candidate = base_report[:m.start(1)] + str(wrong) + base_report[m.end(1):]
                if verify_numbers(candidate, summary_text)["bad"]:
                    return candidate
        return base_report
    return write


def write_verified_report(summary, writer, max_retries=AI_MAX_RETRIES):
    """요약 → AI 리포트 → 숫자 검증 → 틀리면 틀린 숫자를 알려 주고 다시 쓰기(최대 max_retries번)."""
    summary_text = json.dumps(summary, ensure_ascii=False)
    messages = [{"role": "user", "content": AI_USER.format(summary=summary_text)}]
    attempts = []
    for _ in range(max_retries + 1):
        report = writer(messages)
        check = verify_numbers(report, summary_text)
        attempts.append({"report": report, "check": check})
        if not check["bad"]:
            break
        messages += [{"role": "assistant", "content": report},
                     {"role": "user", "content": AI_FIX.format(bad=", ".join(check["bad"]))}]
    return {"report": report, "check": check, "attempts": attempts, "passed": not check["bad"]}


def report_key(summary, writer_name):
    return hashlib.sha256((writer_name + json.dumps(summary, ensure_ascii=False, sort_keys=True)).encode("utf-8")).hexdigest()


def cached_report(summary, writer_name, writer):
    """같은 요약 + 같은 작성자면 저장해 둔 결과를 재사용해요(호출 수 절약)."""
    key = report_key(summary, writer_name)
    cache = st.session_state.setdefault("ai_cache", {})
    if key not in cache:
        cache[key] = write_verified_report(summary, writer)
    return cache[key]


# ================================================================ 화면
STATUS_ICON = {"좋음": "✅ 좋음", "주의": "⚠️ 주의", "문제": "❌ 문제"}

st.caption("엔진 실험실 · 담당: 수학과 B")
st.title("내 파일, 이 엔진으로 믿고 돌려도 될까?")
st.write("CSV를 올리면 공용 엔진으로 분석하고, 그 파일의 실제 값으로 엔진이 제대로 작동하는지 탭마다 점검해요. "
         "맨 아래에서 종합 판정을 읽을 수 있어요.")

# ---------------------------------------------------------------- 1. 파일 입력
st.header("① 파일 넣기")
src = st.radio("파일", ["내 CSV 올리기", "예시 파일로 해 보기"], horizontal=True, label_visibility="collapsed")
data, fname = None, ""
if src == "내 CSV 올리기":
    up = st.file_uploader("CSV 파일 (UTF-8·CP949 모두 돼요)", type=["csv"])
    if up is not None:
        data, fname = up.getvalue(), up.name
else:
    pick = st.selectbox("예시 파일", list(SAMPLES))
    p = ROOT / "tests" / "fixtures" / SAMPLES[pick]
    if p.exists():
        data, fname = p.read_bytes(), p.name
    else:
        st.error("예시 파일을 찾지 못했어요.")

if data is None:
    st.info("CSV를 올리면 분석이 시작돼요. 센서처럼 시간 순서가 있는 데이터도, 한 행이 한 사람·한 건인 표도 돼요.")
    st.stop()

orig_hash = hashlib.sha256(data).hexdigest()[:16]
st.session_state["_orig_hash"] = orig_hash
if st.session_state.get("_last_file") != orig_hash:          # 새 파일이면 설정을 처음으로
    st.session_state["_last_file"] = orig_hash
    for key_ in ("cols_pick", "target", "agg_for"):
        st.session_state.pop(key_, None)
    st.session_state["mode_label"] = MODE_LABEL["auto"]
    st.session_state["k_val"] = 1.0

try:
    with st.spinner("파일을 읽는 중이에요…"):
        table0, prof0 = file_profile(data)
        agg_on = st.session_state.get("agg_for") == orig_hash and bool(prof0["time"])
        if agg_on:   # '시각별로 모아서 보기'를 고른 경우: 모은 데이터로 바꿔서 점검
            data, agg_used, agg_n = date_aggregate(data)
            fname = f"{fname} (시각별 모음)"
            table0, prof0 = file_profile(data)
except ValueError as e:
    st.error(str(e))
    st.stop()

if agg_on:
    st.info(f"지금은 원래 파일을 **시각별로 모은 데이터**({agg_n:,}행: 건수 + {', '.join(agg_used) or '숫자 열 없음'}의 평균)를 점검하고 있어요. "
            "'이상한 사람'이 아니라 '평소와 다른 날'을 찾는 질문이에요.")
    st.button("원래 파일로 돌아가기", on_click=leave_agg)

n_rows = prof0["n"]
if n_rows > BIG_ROWS:
    st.warning(f"{n_rows:,}행짜리 큰 파일이에요. 처음 분석은 수십 초 걸릴 수 있어요(같은 설정으로 다시 볼 때는 바로 떠요).")

c1, c2, c3 = st.columns([2, 1, 3])
mode_ko = {v: kk for kk, v in MODE_LABEL.items()}
options = numeric_options(prof0)
st.session_state["cols_pick"] = [x for x in st.session_state.get("cols_pick", []) if x in options]   # 없는 열 정리
mode_label = c1.radio("모드", list(mode_ko), horizontal=True, key="mode_label",
                      help="자동: 시간 열이 있으면 시계열, 없으면 표. 시간 열 없이 행 순서가 시간인 센서 데이터는 '시계열'을 고르세요.")
k = c2.select_slider("민감도 k (크면 덜 민감)", options=K_GRID, key="k_val")
cols_pick = c3.multiselect("분석할 열 (비워 두면 엔진이 골라요)", options, key="cols_pick",
                           help="코드·번호처럼 숫자 크기에 뜻이 없는 열은 빼세요.")
mode = mode_ko[mode_label]
cols_key = tuple(cols_pick)

try:
    with st.spinner(f"공용 엔진으로 {n_rows:,}행을 분석하는 중이에요…"):
        res = analyze_file(data, mode, k, cols_key, fname)
except ValueError as e:
    st.error(str(e))
    st.stop()
out = res["out"]
diag = diagnose(res, mode)

if st.session_state.get("target") not in out["columns"]:
    st.session_state.pop("target", None)
target = st.selectbox("이상 심기 실험에 쓸 열", out["columns"], key="target",
                      help="이 열의 실제 값 위에 이상을 하나씩 심고 엔진이 잡는지 최대 240번 봐요(음수가 되는 실험은 빼요).")
with st.spinner("이상을 240번 심어 보고, 기준값 k를 바꿔 보는 중이에요… (큰 파일이면 30초쯤)"):
    trials = run_trials(data, mode, k, cols_key, fname, target)
    trials = dict(trials, target=target)
    kpts, kbest = run_k_sweep(data, mode, k, cols_key, fname, target)
rz_rows, rz_examples = robust_check(out)
ver = verifier_check(out)
S = fitness_summary(res, diag, trials, kpts, kbest, rz_rows, ver, k)
grade, _, _ = grade_of(S)
sc = trial_scores(trials["recs"])
setting_desc = (f"{'시각별 모음 · ' if agg_on else ''}{'시계열' if out['mode'] == 'timeseries' else '표'} 모드 · k {k} · "
                f"열 {', '.join(cols_pick) if cols_pick else '엔진 자동(' + ', '.join(out['columns']) + ')'}")
history = record_history(orig_hash, setting_desc, grade, sc["큰_이상_재현율"], sum(x["상태"] != "좋음" for x in diag),
                         {"mode": mode, "cols": list(cols_pick), "k": k, "agg": agg_on})

# ---------------------------------------------------------------- 요약 지표
a, b, c, d = st.columns(4)
a.metric("종합 판정", grade)
b.metric("실제로 돈 모드", "시계열" if out["mode"] == "timeseries" else "표", f"분석 열 {len(out['columns'])}개", delta_color="off")
c.metric("큰 이상 재현율", f"{sc['큰_이상_재현율']:.0%}", f"{sc['실험_수']}회 전체 {sc['재현율']:.0%}", delta_color="off",
         help="4σ 이상(표 모드는 자릿수 실수 포함)으로 심은 이상 중 엔진이 잡은 비율")
d.metric("진단 경고", f"{sum(x['상태'] != '좋음' for x in diag)}개", f"항목 {len(diag)}개 중", delta_color="off")

# ---------------------------------------------------------------- 2. 탭별 점검
st.header("② 점검 결과")
t_run, t_diag, t_inject, t_k, t_rob, t_ver = st.tabs(
    ["기본 실행 결과", "데이터 진단", "이상 심기 240회", "기준값 k", "robust Z 대조", "숫자 검증기"])

with t_run:
    st.caption(f"{out['notice']} · 분석 열: {', '.join(out['columns'])} · {res['seconds']}초")
    if mode == "auto" and out["mode"] == "table":
        st.info("시간 열을 찾지 못해 표 모드로 돌았어요. 센서처럼 행 순서가 시간인 데이터라면 위에서 '시계열'을 고르세요.")
    nres = len(out["result"]["events"]) if out["mode"] == "timeseries" else len(out["result"]["outliers"])
    st.markdown(f"**{'이벤트' if out['mode'] == 'timeseries' else '확인이 필요한 행'} {nres:,}개**")
    ev_rows = events_table(out)
    if len(ev_rows) > 500:
        st.caption(f"이상 점수가 높은 순으로 500개만 보여 줘요(전체 {len(ev_rows):,}개).")
        ev_rows = sorted(ev_rows, key=lambda r: -r.get("이상 점수", 0))[:500]
    st.dataframe(pd.DataFrame(ev_rows), width="stretch", hide_index=True)
    with st.expander("열마다 엔진이 본 역할"):
        st.dataframe(pd.DataFrame(check_table(out)), width="stretch", hide_index=True)
    with st.expander("기본 리포트 (AI 없이 엔진 요약에서 바로 만든 글)"):
        st.markdown(report_text(out))

with t_diag:
    st.subheader("이 파일이 엔진의 전제와 맞는지")
    st.dataframe(pd.DataFrame([{kk: (STATUS_ICON[v] if kk == "상태" else v) for kk, v in x.items() if not kk.startswith("_")} for x in diag]),
                 width="stretch", hide_index=True)
    st.caption("❌ 문제 = 이대로는 결과를 믿기 어려움 · ⚠️ 주의 = 결과를 읽을 때 감안 · ✅ 좋음")

with t_inject:
    st.subheader(f"'{target}' 열에 이상을 하나씩 심고 엔진이 잡는지 {sc['실험_수']}번")
    inf = trials["info"]
    if trials["mode"] == "timeseries":
        st.caption(f"실험 창: 정렬된 데이터의 {inf['창 시작 행']:,}번째부터 {inf['창 길이']}행(원래 이벤트가 가장 적은 구간) · "
                   f"σ = 이 구간의 1.4826×MAD = {inf['σ']:.4g} · 창 안에 원래 있던 이벤트 {inf['원래 이벤트']}건은 오탐에서 빼요 · {trials['seconds']}초")
    else:
        st.caption(f"표본 {inf['표본 행']:,}행 · σ = 1.4826×MAD = {inf['σ']:.4g} · 원래 표시되던 행 {inf['원래 이상치']}개는 오탐에서 빼요 · {trials['seconds']}초")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("재현율", f"{sc['재현율']:.0%}", help="심은 이상 중 잡은 비율")
    m2.metric("큰 이상 재현율", f"{sc['큰_이상_재현율']:.0%}")
    m3.metric("정밀도", f"{sc['정밀도']:.0%}", help="잡은 것 중 심은 이상의 비율(새로 생긴 오탐 포함)")
    m4.metric("F1", f"{sc['F1']:.2f}")
    st.dataframe(pd.DataFrame(trial_grid(trials["recs"])), width="stretch", hide_index=True,
                 column_config={"재현율(%)": st.column_config.ProgressColumn("재현율", min_value=0, max_value=100, format="%d%%")})
    if trials["mode"] == "timeseries":
        st.markdown("**예시 한 번 보기**")
        kinds = list(dict.fromkeys(r["종류"] for r in trials["recs"]))
        ex1, ex2 = st.columns(2)
        ex_kind = ex1.selectbox("이상 종류", kinds)
        ex_size = ex2.selectbox("크기(σ)", TS_SIZES, index=1)
        rec = next(r for r in trials["recs"] if r["종류"] == ex_kind and r["크기"] == ex_size and r["반복"] == 0)
        fig = go.Figure(go.Scatter(y=rec["y"], name=target, line=dict(color=BLUE, width=1.5),
                                   hovertemplate="창 안 %{x}행<br>%{y:.4g}<extra></extra>"))
        fig.add_vrect(x0=rec["시작"] - 0.5, x1=rec["시작"] + rec["길이"] - 0.5, fillcolor=GRAY, opacity=0.35, line_width=0)
        for s_, e_ in rec["잡은 구간"]:
            fig.add_vrect(x0=s_ - 0.5, x1=e_ + 0.5, fillcolor=ORANGE, opacity=0.3, line_width=0)
        fig.update_layout(height=300, margin=dict(t=20, b=40), xaxis_title="실험 창 안의 행", showlegend=False)
        st.plotly_chart(fig, width="stretch")
        st.caption(f"회색 = 심은 구간(정답), 주황 = 엔진이 잡은 이벤트 · 이 예시: {'잡음' if rec['잡음'] else '놓침'}, 새 오탐 {rec['오탐']}건")
    else:
        st.caption("표 모드는 한 행의 값만 바꿔서 그 행이 '확인이 필요한 행'에 들어가는지 봐요. 낮은 쪽 값이 음수가 되는 칸은 실험하지 않아요.")

with t_k:
    st.subheader("기준값 k를 바꾸면 이 파일에서 어떻게 달라지나")
    st.caption("k마다 크기 4σ(표 모드는 4σ·×10) 이상을 다시 심어 재현율·정밀도를 재고, 실험 구간의 원래 데이터에서 몇 개가 표시되는지 셌어요.")
    kp = pd.DataFrame(kpts)
    fig = go.Figure()
    for col, color, dash in (("재현율", BLUE, "solid"), ("정밀도", ORANGE, "solid"), ("F1", GRAY, "dot")):
        fig.add_trace(go.Scatter(x=kp["k"], y=kp[col], name=col, mode="lines+markers", line=dict(color=color, width=2, dash=dash),
                                 marker=dict(size=8), hovertemplate=f"k=%{{x}}<br>{col} %{{y:.2f}}<extra></extra>"))
    fig.add_vline(x=k, line_dash="dash", line_color=GRAY, annotation_text="지금 k")
    fig.update_layout(height=300, margin=dict(t=20, b=40), yaxis=dict(range=[0, 1.05], title=""), xaxis_title="k", legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch")
    fig2 = go.Figure(go.Bar(x=[str(x) for x in kp["k"]], y=kp["원래 데이터 표시 수"], marker_color=BLUE,
                            hovertemplate="k=%{x}<br>%{y}개<extra></extra>"))
    fig2.update_layout(height=220, margin=dict(t=30, b=40), title=dict(text="원래 데이터에서 표시된 수 (적을수록 조용함)", font=dict(size=14)),
                       xaxis_title="k")
    st.plotly_chart(fig2, width="stretch")
    st.metric("이 파일에서 F1이 가장 높은 k", kbest["k"], f"F1 {kbest['F1']:.2f}", delta_color="off")
    st.dataframe(kp, width="stretch", hide_index=True)

with t_rob:
    st.subheader("엔진의 robust Z를 손으로 다시 계산해 맞춰 보기")
    if out["mode"] == "table":
        st.caption("표 모드: 열 전체의 중앙값과 1.4826×MAD로 Z를 계산해요. 한쪽으로 치우친 0 이상 열은 로그로 바꿔서 계산해요.")
    else:
        st.caption(f"시계열 모드: 앞쪽 {out['base']['nb']}행(기준 구간)의 중앙값과 1.4826×MAD로 정상범위(중앙값 ± 3σ)를 잡아요. 빈칸은 앞뒤 값으로 채운 뒤 계산해요.")
    st.latex(r"z_i = \frac{x_i - \mathrm{median}(x)}{1.4826 \times \mathrm{MAD}}, \quad \mathrm{MAD} = \mathrm{median}(|x_i - \mathrm{median}(x)|)")
    st.dataframe(pd.DataFrame(rz_rows), width="stretch", hide_index=True)
    if rz_examples:
        st.markdown("**엔진이 표시한 값 몇 개**" + (" (엑셀로 같은 계산을 해서 맞춰 볼 수 있어요)" if out["mode"] == "table" else ""))
        st.dataframe(pd.DataFrame(rz_examples), width="stretch", hide_index=True)

with t_ver:
    st.subheader("이 파일의 리포트로 숫자 검증기 시험")
    st.caption("AI 리포트를 검사할 때 쓰는 공용 엔진 함수(verify_numbers)를 이 파일의 기본 리포트에 써 봐요. "
               "리포트의 숫자를 하나씩 일부러 틀리게 바꿨을 때 잡아내는지 봐요. (10 이하 정수는 목록 번호일 수 있어 검사하지 않아요.)")
    caught = sum(x["잡음"] for x in ver["cases"])
    v1, v2 = st.columns(2)
    v1.metric("바꾼 숫자 중 잡은 수", f"{caught} / {len(ver['cases'])}")
    v2.metric("원래 리포트에서 틀렸다고 한 숫자", f"{len(ver['clean']['bad'])}개", help="0이어야 정상(오경보 없음)")
    if ver["cases"]:
        st.dataframe(pd.DataFrame(ver["cases"]), width="stretch", hide_index=True)
        missed = [x for x in ver["cases"] if not x["잡음"]]
        if missed:
            st.caption("못 잡은 숫자는 바꾼 값이 요약의 다른 숫자(또는 두 숫자의 차이)와 우연히 가까운 경우예요. 숫자가 맞는 자리에 쓰였는지는 사람이 확인해야 해요.")
    with st.expander("시험에 쓴 기본 리포트"):
        st.markdown(ver["report"])

# ---------------------------------------------------------------- 3. 종합 판정
st.header("③ 종합 판정")
st.caption("위 점검 결과를 모은 요약(원본 행 없음)으로 판정해요. AI를 쓰지 않은 기본 판정이 먼저 나오고, 원하면 AI가 더 읽기 쉽게 다시 써요.")
base_verdict = verdict_text(S)
api_key, model = ai_settings()
writers = ["기본 판정 (AI 없음)", "가짜 AI (시험용 · 숫자 하나를 일부러 틀림)"] + ([f"Claude ({model})"] if api_key else [])
writer_label = st.radio("판정 작성자", writers, horizontal=True)
if not api_key:
    st.caption("Streamlit Secrets에 ANTHROPIC_API_KEY가 없어서 실제 AI는 꺼져 있어요.")
with st.expander("AI에게 보내는 내용 (원본 행 없음)"):
    st.json(S)

if writer_label.startswith("기본"):
    st.markdown(base_verdict)
else:
    cur_key = report_key(S, writer_label)
    if st.button("AI 판정 리포트 만들기"):
        writer = claude_writer(api_key, model) if writer_label.startswith("Claude") else fake_writer(base_verdict, S)
        try:
            st.session_state["ai_last"] = {"key": cur_key, "res": cached_report(S, writer_label, writer)}
        except RuntimeError as e:
            st.session_state.pop("ai_last", None)
            st.error(f"{e} 아래 기본 판정을 대신 보세요.")
    last = st.session_state.get("ai_last")
    rep = last["res"] if last and last.get("key") == cur_key else None
    if last and rep is None:
        st.caption("파일이나 설정이 바뀌어서 이전 AI 리포트는 숨겼어요. 'AI 판정 리포트 만들기'를 다시 누르세요.")
    if rep:
        x1, x2, x3 = st.columns(3)
        x1.metric("숫자 검증", "통과" if rep["passed"] else "실패")
        x2.metric("요약과 맞는 숫자", f"{rep['check']['ok']} / {rep['check']['total']}")
        x3.metric("쓴 횟수", f"{len(rep['attempts'])}번", f"이 접속 AI 호출 {st.session_state.get('ai_calls', 0)}/{AI_CALL_LIMIT}", delta_color="off")
        if rep["passed"]:
            st.success("리포트의 숫자가 모두 점검 요약에 있어요. (숫자가 맞는 자리에 쓰였는지는 사람이 확인해야 해요.)")
            st.markdown(rep["report"])
        else:
            st.warning(f"다시 써도 요약에 없는 숫자가 남았어요: {', '.join(rep['check']['bad'])}. 이 리포트 대신 기본 판정을 보세요.")
            st.markdown(base_verdict)
        with st.expander("시도별 기록"):
            for i, t in enumerate(rep["attempts"], 1):
                bad = t["check"]["bad"]
                st.markdown(f"**{i}번째** · " + (f"요약에 없는 숫자: {', '.join(bad)}" if bad else "통과"))
    else:
        st.markdown(base_verdict)

# ---------------------------------------------------------------- 3-1. 이렇게 바꿔 보세요
fixes = suggest_fixes(diag, out, kbest, k, cols_pick, agg_on)
st.subheader("이렇게 바꿔 보세요")
if not fixes:
    st.success("진단에서 고칠 점이 나오지 않았어요. 지금 설정 그대로 써도 돼요.")
else:
    if grade == "잘 맞음":
        st.caption("판정은 좋지만, 진단에서 손볼 만한 점이 있어요.")
    st.caption("원래 판정은 아래 '점검 기록'에 남아요. 버튼을 누르면 위의 설정이 바뀌고 같은 파일로 처음부터 다시 점검해요.")
    kind_icon = {"설정": "🔧 설정만 바꾸면 돼요", "데이터 바꾸기": "🔄 페이지가 데이터를 바꿔서 보여 줘요", "데이터 준비": "📁 파일을 손봐서 다시 올려야 해요"}
    for i, f in enumerate(fixes):
        with st.container(border=True):
            l, r = st.columns([4, 1])
            l.markdown(f"**{f['제목']}**  \n{kind_icon[f['종류']]} · {f['이유']}")
            if f.get("apply"):
                r.button("적용해서 다시 점검", key=f"fix_{i}", on_click=apply_fix, args=(f["apply"],), type="primary" if i == 0 else "secondary")
    weak_engine = [f"{kind} {size}" for kind, d in S["이상_심기"]["종류별_크기별_재현율_퍼센트"].items() for size, v in d.items()
                   if v < 50 and size.endswith("σ") and float(size[:-1]) >= 4]
    if weak_engine:
        st.caption(f"참고: {', '.join(weak_engine)}처럼 큰 이상도 잘 못 잡는 건 엔진 방식의 한계라 설정이나 데이터를 바꿔도 크게 달라지지 않아요.")

if len(history) > 1:
    st.markdown("**점검 기록 (이 파일)**")
    st.dataframe(pd.DataFrame([{kk: v for kk, v in h.items() if not kk.startswith("_")} for h in history]), width="stretch", hide_index=True)
    first, now, best = history[0], next(h for h in history if h["설정"] == setting_desc), best_of(history, agg_on)
    msg = f"처음 설정: **{first['판정']}** → 지금 설정: **{now['판정']}**."
    if best["설정"] == now["설정"]:
        msg += " 지금 설정이 해 본 것 중 가장 좋아요. 다른 분야 페이지에서 이 파일을 쓸 때도 이 설정을 쓰세요."
        st.caption(msg)
    else:
        st.caption(msg + f" {'시각별 모음' if agg_on else '원래 데이터'}로 해 본 것 중에는 {best['순서']}번째 설정({best['판정']})이 더 좋았어요.")
        st.button(f"{best['순서']}번째 설정으로 돌아가기", on_click=apply_fix, args=(best["_apply"],))

with st.expander("AI 사용 원칙과 비용 장치"):
    st.markdown("**보내지 않음**: 원본 행, 이름·연락처·주민번호 열, 한 사람·한 건 단위의 값  \n"
                "**보냄**: 위 점검 요약(진단 결과, 실험 재현율, k별 결과, 대조 결과)")
    st.markdown(f"- 틀린 숫자 다시 쓰기 최대 {AI_MAX_RETRIES}번 · 같은 요약은 저장해 재사용\n"
                f"- 접속당 {AI_CALL_LIMIT}번 + 앱 전체 하루 {AI_GLOBAL_LIMIT}번 상한 · 돈이 들지 않은 실패 호출은 횟수에서 뺌\n"
                "- 리포트가 잘리거나 다시 써도 틀리면 기본 판정을 보여 줌")
    st.caption("앱 전체 상한은 앱이 다시 시작되면 0부터 다시 세요. 돈이 드는 진짜 상한은 Anthropic 콘솔의 사용 한도로 꼭 걸어 두세요.")
    if not api_key:
        st.code('ANTHROPIC_API_KEY = "sk-ant-..."\nAI_MODEL = "claude-haiku-5-5"   # 생략하면 이 값', language="toml")
