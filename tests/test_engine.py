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
    out = E.run_engine(table, k=k, dataset_name=name)
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
    out = E.run_engine(pd.DataFrame({"번호": range(1, 61), "온도": [20 + (i % 3) * 0.1 for i in range(50)] + [27] * 10}))
    assert out["result"]["events"], "DataFrame 입력에서 이벤트를 못 찾음"
    assert E.events_table(out) and E.scores_table(out)
    return []


if __name__ == "__main__":
    refs = json.load(open(os.path.join(FIX, "js_reference.json"), encoding="utf-8"))
    total = 0
    for ref in refs:
        p = check(ref)
        total += len(p)
        print(("통과  " if not p else "실패  ") + ref["name"] + ("" if not p else "\n  - " + "\n  - ".join(p)))
    for fn in (test_verifier, test_cp949_and_dataframe):
        try:
            fn()
            print("통과  " + fn.__name__)
        except AssertionError as e:
            total += 1
            print("실패  " + fn.__name__ + f": {e}")
    print("\n모두 통과" if total == 0 else f"\n실패 {total}건")
    sys.exit(1 if total else 0)
