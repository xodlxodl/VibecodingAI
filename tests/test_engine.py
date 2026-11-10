"""원본(JS) 엔진과 Python 엔진의 결과가 같은지 확인하는 검사.

실행:  python tests/test_engine.py      (저장소 맨 위 폴더에서)
fixtures/js_reference.json 은 데모 사이트의 JS 엔진으로 같은 CSV를 분석한 결과입니다.
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from engine import engine as E  # noqa: E402

FIX = os.path.join(HERE, "fixtures")
CASES = {"food": ("food.csv", 1.0), "printer": ("printer.csv", 1.0), "food_k15": ("food.csv", 1.5), "generic": ("generic.csv", 1.0)}


def close(a, b, tol=1e-6):
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def max_diff(xs, ys):
    return max((abs(x - y) for x, y in zip(xs, ys)), default=0.0)


def check(ref):
    name = ref["name"]
    path, k = CASES[name]
    table = E.load_csv(os.path.join(FIX, path))
    out = E.run_engine(table, k=k, dataset_name=name, mode="timeseries")  # 원본 JS는 항상 시계열 방식
    prof, prep, base, res = out["profile"], out["prep"], out["base"], out["result"]
    problems = []

    def expect(cond, msg):
        if not cond:
            problems.append(msg)

    expect([[c["name"], c["role"], c["kind"], c["meaning"]] for c in prof["cols"]] == ref["roles"], "열 역할 판별이 다름")
    expect(prof["duplicates"] == ref["dup"], f"중복 행 수 {prof['duplicates']} vs {ref['dup']}")
    expect(out["prep"]["cols"] and [c["name"] for c in prep["cols"]] == ref["selected"], "추천 열이 다름")
    expect(out["pack"] == ref["pack"] and [r["id"] for r in out["rules"]] == ref["rules"], "도메인 팩·규칙이 다름")
    expect(base["n"] == ref["n"] and base["nb"] == ref["nb"], "행 수 또는 기준 구간이 다름")
    for j, c in enumerate(prep["cols"]):
        expect(close(c["mu"], ref["mu"][j]) and close(c["sigma"], ref["sigma"][j]), f"{c['name']} 기준선이 다름")
        for key in ("point", "ewma", "vari"):
            d = max_diff(c[key], ref[key][j])
            expect(d < 1e-6, f"{c['name']} {key} 점수 최대 차이 {d:.2e}")
    if ref["mv"] is not None:
        d = max_diff(base["mv"], ref["mv"])
        expect(d < 1e-6, f"다변량 점수 최대 차이 {d:.2e}")
    flagged = [1 if r["flagged"] else 0 for r in res["rows"]]
    diff_rows = [i for i, (a, b) in enumerate(zip(flagged, ref["flagged"])) if a != b]
    expect(not diff_rows, f"이상 판정이 다른 행 {len(diff_rows)}개: {diff_rows[:10]}")
    expect([r["sev"] for r in res["rows"]] == ref["sev"], "등급이 다른 행이 있음")
    expect([int(r["score"]) for r in res["rows"]] == ref["score"], "이상 점수가 다른 행이 있음")
    mine = [{"id": e["id"], "sev": e["sev"], "type": e["type"], "start": e["start"], "end": e["end"], "main": e["main"],
             "cols": e["cols"], "peak": e["peak"], "peakScore": int(e["peakScore"]), "rules": e["rules"]} for e in res["events"]]
    expect(mine == ref["events"], f"이벤트가 다름\n  Python: {mine}\n  JS    : {ref['events']}")
    expect(len(out["forecasts"]) == len(ref["forecasts"]) and all(
        f["col"] == g["col"] and close(f["hours"], g["hours"]) for f, g in zip(out["forecasts"], ref["forecasts"])), "추세 예측이 다름")
    if ref["ens"]:
        ev = out["evaluation"]
        expect(all(close(ev["ens"][key], ref["ens"][key]) for key in ("precision", "recall", "f1")), "종합 성능이 다름")
        expect([[m["key"], round(m["precision"], 9), round(m["recall"], 9)] for m in ev["methods"]]
               == [[m[0], round(m[1], 9), round(m[2], 9)] for m in ref["methods"]], "방법별 성능이 다름")
        sw = E.sweep(prep, base, out["rules"])
        expect(all(close(a["f1"], b["f1"]) and close(a["k"], b["k"]) for a, b in zip(sw, ref["sweep"])), "k 스윕 결과가 다름")
    S = out["summary"]
    expect(json.dumps(S, ensure_ascii=False, sort_keys=True) == json.dumps(ref["summary"], ensure_ascii=False, sort_keys=True),
           "AI 요약(JSON)이 다름")
    expect(E.template_report(S) == ref["template"], "기본 리포트 문장이 다름")
    return problems


def test_verifier():
    S = '{"값": 12.34, "다른값": 98.7, "건수": 432}'
    r = E.verify_numbers("평균은 12.3이고 432건 중 3건이 이상, 99.9는 지어낸 값", S)
    assert r["bad"] == ["99.9"], r
    r = E.verify_numbers("합계는 4,062건이다", '{"합계": 4062}')
    assert r["bad"] == [], r
    return []


def test_cp949_and_dataframe():
    import pandas as pd
    raw = "시각,온도(℃)\n" + "\n".join(f"2026-10-01 00:{i:02d},{20 + (5 if i > 40 else 0) + (i % 3) * 0.1:.1f}" for i in range(60))
    t = E.load_csv(raw.encode("cp949", errors="ignore"))
    assert t["encoding"].startswith("CP949") and len(t["rows"]) == 60, t["encoding"]
    out = E.run_engine(pd.DataFrame({"번호": range(1, 61), "온도": [20 + (i % 3) * 0.1 for i in range(50)] + [27] * 10}), mode="timeseries")
    assert out["result"]["events"], "DataFrame 입력에서 이벤트를 못 찾음"
    assert E.events_table(out) and E.scores_table(out)
    return []


def test_table_mode():
    """v1.1: 시간 열 없는 표(건강검진·보험)는 표 모드 — 범주형 숫자 열 제외, 이벤트 묶기 없음, 중복 유지, 행 순서 무관"""
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(3)
    n = 2000
    df = pd.DataFrame({"ID": range(1, n + 1), "sex": rng.integers(1, 3, n), "smoke": rng.integers(0, 2, n),
                       "age_band": rng.integers(1, 6, n), "sbp": rng.normal(122, 14, n).round(0),
                       "fbg": rng.normal(95, 12, n).round(0), "bmi": rng.normal(23.5, 3.2, n).round(1)})
    df.loc[[10, 500, 1500], "fbg"] = [260, 300, 280]        # 극단값 3명
    df.loc[[700], "sbp"] = [40]
    df = pd.concat([df, df.iloc[[5]].assign(ID=n + 1)], ignore_index=True)
    out = E.run_engine(df, rules=[{"id": "dm", "col": "fbg", "op": ">=", "value": 126, "durationMin": 0, "label": "공복혈당 126 이상", "source": "대한당뇨병학회"}])
    assert out["mode"] == "table", out["mode"]
    assert set(out["columns"]) == {"sbp", "fbg", "bmi"}, out["columns"]           # sex·smoke·age_band(값 2~5개)는 자동 선택에서 빠짐
    assert out["result"]["events"] == [], "표 모드에서는 이벤트로 묶지 않아야 함"
    rows = {r["원래 행"] for r in E.events_table(out)}
    assert {12, 502, 1502, 702} <= rows, sorted(rows)[:20]                        # 심은 극단값 4명 (원래 행 = 인덱스 + 2)
    assert len(out["prep"]["idx"]) == n + 1                                       # 중복 행을 빼지 않음
    shuffled = df.sample(frac=1, random_state=1)
    out2 = E.run_engine(shuffled, rules=out["rules"])
    assert len(out2["result"]["outliers"]) == len(out["result"]["outliers"]), "행 순서를 바꾸면 결과가 같아야 함"
    S = out["summary"]
    assert S["분석_방식"] == "표" and S["기준_규칙"][0]["해당_행수"] > 0
    assert "sex" not in E.report_text(out)
    roles = {r["열"]: r["역할"] for r in E.check_table(out)}
    assert roles["sex"] == "범주(숫자로 적힘)" and roles["ID"] == "번호·순번", roles
    return []


def test_table_skewed():
    """v1.1: 한쪽으로 치우친 금액·횟수 열(보험금 등)은 로그 척도로 봐서 정상 고액 건을 무더기로 잡지 않아야 함"""
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(7)
    n = 5000
    df = pd.DataFrame({"계약번호": range(1, n + 1), "성별": rng.integers(0, 2, n),
                       "청구금액": rng.lognormal(13, 1.0, n).round(0), "보험료": rng.normal(80, 15, n).round(1)})
    df.loc[100, "청구금액"] = 9e9                                                   # 입력 실수(0 몇 개 더 붙음)
    out = E.run_engine(df)
    assert out["mode"] == "table" and set(out["columns"]) == {"청구금액", "보험료"}, out["columns"]
    rate = len(out["result"]["outliers"]) / n
    assert rate < 0.03, f"치우친 열에서 이상치 비율 {rate:.1%} — 너무 많이 잡음"
    assert 102 in {r["원래 행"] for r in E.events_table(out)}, "명백한 입력 실수를 못 잡음"
    c = next(c for c in out["prep"]["cols"] if c["name"] == "청구금액")
    assert c["scale"] == "log", c["scale"]
    # 사고 건수(0이 대부분, 0~4)는 '횟수'로 표시하고, 직접 골라 넣어도 통계적 이상치로는 안 잡고 기준 규칙만 적용
    df["사고건수"] = np.minimum(rng.poisson(0.12, n), 4)
    roles = {r["열"]: r["역할"] for r in E.check_table(E.run_engine(df))}
    assert roles["사고건수"].startswith("횟수") and roles["성별"] == "범주(숫자로 적힘)", roles
    rule = [{"id": "c3", "col": "사고건수", "op": ">=", "value": 2, "durationMin": 0, "label": "사고 2건 이상", "source": "-"}]
    out = E.run_engine(df, columns=["사고건수", "보험료"], rules=rule)
    many = int((df["사고건수"] >= 2).sum())
    rule_rows = sum(1 for r in out["result"]["outliers"] if r["rule"])
    assert rule_rows == many, (rule_rows, many)
    assert len(out["result"]["outliers"]) < many + 0.02 * n, "횟수 열 때문에 1건 사고자까지 무더기로 잡음"
    assert "기준 규칙만" in E.report_text(out)
    # 행 순서가 섞인 표에서도 ID 열(IDpol·이름 힌트 없는 연속 번호)은 분석에서 빠져야 함. 금액처럼 값이 다 달라도 띄엄띄엄이면 숫자로 유지
    mixed = df.sample(frac=1, random_state=3).assign(IDpol=lambda d: d["계약번호"] + 1000, pol=lambda d: d["계약번호"])
    out = E.run_engine(mixed)
    assert not {"계약번호", "IDpol", "pol"} & set(out["columns"]) and "청구금액" in out["columns"], out["columns"]
    return []


if __name__ == "__main__":
    refs = json.load(open(os.path.join(FIX, "js_reference.json"), encoding="utf-8"))
    total = 0
    for ref in refs:
        p = check(ref)
        total += len(p)
        print(("통과  " if not p else "실패  ") + ref["name"] + ("" if not p else "\n  - " + "\n  - ".join(p)))
    for fn in (test_verifier, test_cp949_and_dataframe, test_table_mode, test_table_skewed):
        try:
            fn()
            print("통과  " + fn.__name__)
        except AssertionError as e:
            total += 1
            print("실패  " + fn.__name__ + f": {e}")
    print("\n모두 통과" if total == 0 else f"\n실패 {total}건")
    sys.exit(1 if total else 0)
