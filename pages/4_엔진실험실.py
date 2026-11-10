"""엔진 실험실 (담당: 수학과 B)
이상 심어 보기 · 240회 실험 결과 · 기준값 k · 숫자 검증기 · robust Z 손 재현.
240회 실험과 k 곡선, 검증기 시험 결과는 HTML 시안에 저장된 값(data/snap.json)을 그대로 보여 줘요.
'이상 심어 보기'는 단순 robust Z 탐지기와 공용 엔진(시계열 모드, 4개 방법 합의)을 골라 돌려 볼 수 있어요.
공용 엔진 v1.1부터 시간 열이 없으면 자동으로 표 모드가 되므로, 행 순서가 시간인 데이터는 mode="timeseries"를 꼭 지정해요.
AI 리포트: 엔진 요약(원본 행 없음)만 AI에게 보내고, 리포트 속 숫자를 verify_numbers로 대조해 틀리면 최대 2번 다시 쓰게 해요.
API 키는 Streamlit Secrets의 ANTHROPIC_API_KEY에 넣어요(코드·저장소에 넣지 않음). 키가 없으면 '가짜 AI'로 흐름만 시험할 수 있어요.
"""
import hashlib
import json
import re
import sys
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
from engine import check_table, events_table, load_csv, report_text, run_engine, verify_numbers  # noqa: E402

st.set_page_config(page_title="엔진 실험실", layout="wide")

# ---------------------------------------------------------------- AI 리포트 설정 ('AI 전송 설계' 탭의 장치를 코드로 옮긴 것)
AI_URL = "https://api.anthropic.com/v1/messages"
AI_DEFAULT_MODEL = "claude-haiku-5-5"   # Secrets의 AI_MODEL로 바꿀 수 있음
AI_MAX_RETRIES = 2                      # 틀린 숫자 다시 쓰기 최대 2번
AI_CALL_LIMIT = 20                      # 접속(세션)당 호출 20번 상한
AI_SYSTEM = (
    "너는 데이터 분석 결과를 비전문가에게 설명하는 한국어 리포트 작성자다. "
    "사용자가 주는 JSON 요약에 있는 숫자만 쓰고, 새 숫자를 계산하거나 지어내지 마라. "
    "통계적 이상은 원인을 확정하지 않는다는 한계를 반드시 적어라. "
    "형식: ## 요약 / ## 주요 이상 / ## 가능한 원인 / ## 권장 조치 / ## 한계 (마크다운)."
)
AI_USER = "다음은 이상 탐지 엔진의 계산 결과 요약이다. 원본 데이터는 없다.\n\n```json\n{summary}\n```\n\n이 요약만으로 리포트를 써라."
AI_FIX = "다음 숫자는 요약에 없다: {bad}. 요약에 있는 숫자만 써서 리포트 전체를 다시 써라."


@st.cache_data
def load_lab():
    return json.loads((ROOT / "data" / "snap.json").read_text(encoding="utf-8"))["lab"]


L = load_lab()


def ai_settings():
    """Secrets에서 키와 모델을 읽어요. secrets.toml이 없으면 st.secrets 접근 자체가 오류라 감싸요."""
    try:
        return st.secrets.get("ANTHROPIC_API_KEY"), st.secrets.get("AI_MODEL", AI_DEFAULT_MODEL)
    except Exception:
        return None, AI_DEFAULT_MODEL


def claude_writer(key, model):
    """AI 호출 함수를 만들어 돌려줘요. 부를 때마다 세션 호출 수를 세고, 상한을 넘으면 멈춰요."""
    def write(messages):
        used = st.session_state.get("ai_calls", 0)
        if used >= AI_CALL_LIMIT:
            raise RuntimeError(f"이 접속에서 AI 호출 {AI_CALL_LIMIT}번을 모두 썼어요. 새로 접속하면 다시 쓸 수 있어요.")
        st.session_state["ai_calls"] = used + 1
        body = json.dumps({"model": model, "max_tokens": 1500, "system": AI_SYSTEM, "messages": messages}).encode("utf-8")
        req = urllib.request.Request(AI_URL, data=body, headers={
            "x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"AI 호출 실패 (HTTP {e.code}): {e.read().decode('utf-8', 'ignore')[:200]}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"AI 서버에 연결하지 못했어요: {e.reason}") from e
        return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
    return write


def fake_writer(base_report, summary):
    """API 키 없이 흐름을 시험하는 가짜 AI: 첫 번째엔 숫자 하나를 틀리게 쓰고, 고쳐 달라고 하면 바르게 써요.
    틀린 숫자가 우연히 다른 계산값과 비슷해 검증을 통과하지 않도록, 검증기가 잡는 값으로 골라요."""
    summary_text = json.dumps(summary, ensure_ascii=False)

    def write(messages):
        if len(messages) > 1:   # 고쳐 달라는 요청이 붙어 있으면 바른 리포트
            return base_report
        for m in re.finditer(r"(?<![\w.])(\d+\.\d+|\d{2,})(?![\w.])", base_report):
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


def cached_report(summary, writer_name, writer):
    """같은 요약 + 같은 작성자면 저장해 둔 결과를 재사용해요(호출 수 절약)."""
    key = hashlib.sha256((writer_name + json.dumps(summary, ensure_ascii=False, sort_keys=True)).encode("utf-8")).hexdigest()
    cache = st.session_state.setdefault("ai_cache", {})
    if key not in cache:
        cache[key] = write_verified_report(summary, writer)
    return cache[key]


def robust_z(values, base_n):
    base = values[:base_n]
    m = np.median(base)
    s = 1.4826 * np.median(np.abs(base - m))
    return m, s, (values - m) / (s if s > 0 else 1)


def segments(mask, min_len):
    out, start = [], None
    for i, on in enumerate(list(mask) + [False]):
        if on and start is None:
            start = i
        if not on and start is not None:
            if i - start >= min_len:
                out.append((start, i - 1))
            start = None
    return out


def engine_segments(y, k):
    """공용 엔진을 시계열 모드로 돌려 (시작 행, 끝 행) 목록을 돌려줘요.
    시간 열이 없는 데이터라 mode를 지정하지 않으면 표 모드가 되어 이벤트가 0건이 돼요."""
    out = run_engine(pd.DataFrame({"값": y}), mode="timeseries", k=k)
    rn = out["prep"]["rowNums"]  # CSV 줄 번호(머리글이 1번) → 0부터 세는 행 번호는 -2
    return out, [(rn[e["start"]] - 2, rn[e["end"]] - 2) for e in out["result"]["events"]]


st.caption("엔진 실험실 · 담당: 수학과 B")
st.title("이 엔진을 어디까지 믿을 수 있나")
big = L["bench"]["big"]
a, b, c, d = st.columns(4)
a.metric("240회 재현율", f"{L['bench']['recall']:.1%}")
b.metric("정밀도", f"{L['bench']['precision']:.1%}")
c.metric("4σ 이상 F1", f"{big['f1']:.2f}")
d.metric("숫자 검증기 재현율", f"{L['verifier']['numRecall']:.1%}")

t_inject, t_csv, t_bench, t_k, t_ver, t_rob, t_ai = st.tabs(
    ["이상 심어 보기", "내 CSV로 돌려 보기", "240회 실험", "기준값 k", "숫자 검증기", "robust Z 손 재현", "AI 전송 설계"])

with t_inject:
    st.subheader("이상을 하나 심고 탐지기를 돌려요")
    st.caption("회색 = 심은 구간(정답), 빨강 = 잡은 이벤트 · 겹치면 정답, 안 겹치면 오탐")
    detector = st.radio("탐지기", ["robust Z 하나 (단순)", "공용 엔진 (시계열 모드)"], horizontal=True)
    use_engine = detector.startswith("공용")
    col1, col2, col3, col4 = st.columns(4)
    kind = col1.selectbox("이상 종류", ["튀는 값", "수준 이동", "변동성 증가", "서서히 흐름"])
    size = col2.slider("크기(σ)", 1.0, 8.0, 4.0, 0.5)
    if use_engine:
        k_val = col3.slider("민감도 k (크면 덜 민감)", 0.5, 3.0, 1.0, 0.1)
    else:
        k_val = col3.slider("기준 |z|", 2.0, 8.0, 4.0, 0.5)
    seed = col4.number_input("시드", 0, 9999, 42)
    r = np.random.default_rng(int(seed))
    n, start, length = 500, 320, 40
    y = 10 + r.standard_normal(n)
    if kind == "튀는 값":
        length = 1
        y[start] += size
    elif kind == "수준 이동":
        y[start: start + length] += size
    elif kind == "변동성 증가":
        y[start: start + length] = 10 + (y[start: start + length] - 10) * size
    else:
        y[start: start + length] += np.linspace(0, size, length)
    if use_engine:
        out, events = engine_segments(y, k_val)
        st.caption(f"엔진 모드: {out['mode']} · 분석 열: {', '.join(out['columns'])} · {out['notice']}")
    else:
        _, _, z = robust_z(y, 150)
        events = segments(np.abs(z) >= k_val, 1 if kind == "튀는 값" else 2)
    hit = any(s <= start + length - 1 and e >= start for s, e in events)
    fp = sum(1 for s, e in events if e < start or s > start + length - 1)
    fig = go.Figure(go.Scatter(y=y, name="값", line=dict(width=1)))
    fig.add_vrect(x0=start - 0.5, x1=start + length - 0.5, fillcolor="gray", opacity=0.3, line_width=0)
    for s, e in events:
        fig.add_vrect(x0=s - 0.5, x1=e + 0.5, fillcolor="crimson", opacity=0.3, line_width=0)
    fig.update_layout(height=320, xaxis_title="행", margin=dict(t=20))
    st.plotly_chart(fig, width="stretch")
    a, b = st.columns(2)
    a.metric("심은 이상", "잡음" if hit else "놓침")
    b.metric("오탐", f"{fp}건")
    if use_engine:
        with st.expander("모드를 지정하지 않으면? (auto → 표 모드)"):
            auto = run_engine(pd.DataFrame({"값": y}), k=k_val)
            st.write(f"자동 모드 결과: **{auto['mode']}** · 이벤트 {len(auto['result'].get('events', []))}건 · "
                     f"확인이 필요한 행 {len(auto['result'].get('outliers', []))}개")
            st.caption("이 데이터에는 시간 열이 없어서 자동이면 표 모드가 돼요. 행을 하나씩 따로 보므로 수준 이동·서서히 흐름 같은 '이어진 이상'을 이벤트로 묶지 못해요.")

with t_csv:
    st.subheader("내 CSV를 공용 엔진에 넣어 보기")
    st.caption("시간 열이 있으면 자동으로 시계열, 없으면 표 모드예요. 시간 열 없이 행 순서가 시간인 센서 데이터라면 '시계열'을 고르세요.")
    file = st.file_uploader("CSV 파일", type=["csv"])
    col1, col2 = st.columns(2)
    mode_ko = {"자동": "auto", "시계열 (행 순서 = 시간)": "timeseries", "표 (한 행 = 한 건)": "table"}
    mode_label = col1.radio("모드", list(mode_ko), horizontal=True)
    k_csv = col2.slider("민감도 k (크면 덜 민감)", 0.5, 3.0, 1.0, 0.1, key="k_csv")
    if file is not None:
        try:
            table = load_csv(file.getvalue())
            out = run_engine(table, mode=mode_ko[mode_label], k=k_csv, dataset_name=file.name)
        except ValueError as e:
            st.error(str(e))
        else:
            a, b, c = st.columns(3)
            a.metric("실제로 돈 모드", "시계열" if out["mode"] == "timeseries" else "표")
            b.metric("분석한 열", len(out["columns"]))
            c.metric("이벤트" if out["mode"] == "timeseries" else "확인이 필요한 행",
                     len(out["result"]["events"] if out["mode"] == "timeseries" else out["result"]["outliers"]))
            st.caption(out["notice"])
            if mode_ko[mode_label] == "auto" and out["mode"] == "table":
                st.info("시간 열을 찾지 못해 표 모드로 돌았어요. 센서처럼 행 순서가 시간인 데이터라면 위에서 '시계열'을 고르세요.")
            st.markdown("**데이터 점검**")
            st.dataframe(pd.DataFrame(check_table(out)), width="stretch", hide_index=True)
            st.markdown("**이벤트**" if out["mode"] == "timeseries" else "**확인이 필요한 행**")
            st.dataframe(pd.DataFrame(events_table(out)), width="stretch", hide_index=True)
            with st.expander("기본 리포트 (AI 없이 요약에서 바로 만든 글)"):
                st.markdown(report_text(out))

            st.markdown("**AI 리포트**")
            st.caption("AI에게는 아래 '보내는 내용'(집계 요약)만 보내요. 리포트 속 숫자는 숫자 검증기로 요약과 대조하고, 틀리면 최대 2번 다시 쓰게 해요.")
            api_key, model = ai_settings()
            writers = ["가짜 AI (시험용 · 숫자 하나를 일부러 틀림)"] + ([f"Claude ({model})"] if api_key else [])
            writer_label = st.radio("리포트 작성자", writers, horizontal=True)
            if not api_key:
                st.caption("Streamlit Secrets에 ANTHROPIC_API_KEY가 없어서 실제 AI는 꺼져 있어요. 가짜 AI로 검증·재작성 흐름만 시험할 수 있어요.")
            with st.expander("AI에게 보내는 내용 (원본 행 없음)"):
                st.json(out["summary"])
            if st.button("AI 리포트 만들기"):
                writer = claude_writer(api_key, model) if writer_label.startswith("Claude") else fake_writer(report_text(out), out["summary"])
                try:
                    st.session_state["ai_last"] = cached_report(out["summary"], writer_label, writer)
                except RuntimeError as e:
                    st.session_state.pop("ai_last", None)
                    st.error(f"{e} 기본 리포트를 대신 쓰세요.")
            res = st.session_state.get("ai_last")
            if res:
                a, b, c = st.columns(3)
                a.metric("숫자 검증", "통과" if res["passed"] else "실패")
                b.metric("요약과 맞는 숫자", f"{res['check']['ok']} / {res['check']['total']}")
                c.metric("쓴 횟수", f"{len(res['attempts'])}번", f"이 접속 AI 호출 {st.session_state.get('ai_calls', 0)}/{AI_CALL_LIMIT}", delta_color="off")
                if res["passed"]:
                    st.success("리포트의 숫자가 모두 계산값 요약에 있어요. (숫자가 맞는 자리에 쓰였는지는 사람이 확인해야 해요.)")
                else:
                    st.warning(f"다시 써도 요약에 없는 숫자가 남았어요: {', '.join(res['check']['bad'])}. 이 리포트는 쓰지 말고 기본 리포트를 쓰세요.")
                st.markdown(res["report"])
                with st.expander("시도별 기록"):
                    for i, t in enumerate(res["attempts"], 1):
                        bad = t["check"]["bad"]
                        st.markdown(f"**{i}번째** · " + (f"요약에 없는 숫자: {', '.join(bad)}" if bad else "통과"))

with t_bench:
    st.subheader("이상 종류 × 크기별 재현율")
    st.caption(f"총 {L['bench']['runs']}회 · 기준값 k = {L['bench']['k']} · HTML 시안에 저장된 결과")
    rows = []
    for g in L["bench"]["grid"]:
        for cell in g["cells"]:
            row = {"이상 종류": g["name"], "단위": g.get("unit", "")}
            row.update({key: val for key, val in cell.items() if not isinstance(val, (list, dict))})
            rows.append(row)
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

with t_k:
    st.subheader("기준값 k에 따른 정밀도·재현율·F1")
    pts = pd.DataFrame(L["sweep"]["pts"]).set_index("k")[["precision", "recall", "f1"]]
    st.line_chart(pts)
    st.metric("최고 F1을 낸 k", L["sweep"]["best"]["k"], f"F1 {L['sweep']['best']['f1']:.3f}", delta_color="off")
    st.dataframe(pts.round(3), width="stretch")

with t_ver:
    st.subheader("숫자 검증기 직접 써 보기")
    st.caption("리포트 속 숫자가 계산값 요약에 있는지 하나씩 대조해요. AI 리포트를 검사할 때와 같은 공용 엔진 함수(verify_numbers)를 써요.")
    st.caption("기본 예시는 보험계리 페이지의 계산값이에요(관측 빈도·검증 A/E·학습자료 평균 순보험료). 240.5는 일부러 틀리게 쓴 숫자예요.")
    col1, col2 = st.columns(2)
    summary = col1.text_area("계산값 요약", '{"관측_빈도": 0.1061, "검증_A_E_건수": 1.022, "순보험료": 235.9}', height=140)
    report = col2.text_area("리포트", "관측 빈도는 0.1061이고 검증 A/E는 1.022다. 순보험료는 240.5유로다.", height=140)
    chk = verify_numbers(report, summary)
    if not chk["total"]:
        st.info("검사할 숫자가 없어요. (10 이하 정수는 목록 번호일 수 있어 건너뛰어요.)")
    elif chk["bad"]:
        st.warning(f"숫자 {chk['total']}개 중 {len(chk['bad'])}개가 계산값에 없어요: {', '.join(chk['bad'])}")
    else:
        st.success(f"숫자 {chk['total']}개가 모두 계산값과 맞아요.")
    st.caption("반올림 차이(약 0.6%)와 두 계산값의 차이를 인용한 경우는 맞는 것으로 봐요. 숫자가 맞는 자리에 쓰였는지는 보지 않아요.")
    st.subheader("기존 시험 결과")
    V = L["verifier"]
    a, b, c = st.columns(3)
    a.metric("바꾼 숫자 중 잡은 수", f"{V['caught']} / {V['totalChanged']}")
    b.metric("변조 리포트 적발률", f"{V['reportRecall']:.0%}")
    c.metric("깨끗한 리포트 오경보", f"{V['falseAlarm']}건 / {V['nClean']}건")
    case_id = st.selectbox("시험 리포트", range(len(V["cases"])), format_func=lambda i: V["cases"][i]["id"] + (" · 변조" if V["cases"][i]["tampered"] else " · 원본"))
    st.markdown(V["cases"][case_id]["report"])

with t_rob:
    st.subheader("값 15개, 그중 하나가 튐")
    st.caption("엑셀로 같은 계산을 해서 맞춰 보는 예제예요.")
    text = st.text_input("값 (쉼표로 구분)", "10.2, 9.8, 10.1, 9.9, 10.0, 10.3, 9.7, 10.1, 9.9, 10.2, 10.0, 9.8, 18.5, 10.1, 9.9")
    try:
        vals = np.array([float(x) for x in text.split(",")])
        m, s, z = robust_z(vals, len(vals))
        a, b, c = st.columns(3)
        a.metric("중앙값", f"{m:.3f}")
        b.metric("MAD", f"{s / 1.4826:.3f}")
        c.metric("1.4826 × MAD", f"{s:.3f}")
        st.dataframe(pd.DataFrame({"값": vals, "robust z": np.round(z, 2)}), width="stretch")
        st.latex(r"z_i = \frac{x_i - \mathrm{median}(x)}{1.4826 \times \mathrm{MAD}}")
    except ValueError:
        st.error("숫자를 쉼표로 구분해 입력해 주세요.")

with t_ai:
    st.subheader("AI에게 보내는 것과 보내지 않는 것")
    col1, col2 = st.columns(2)
    col1.markdown("**보내지 않음**\n- 원본 행 전체\n- 이름·연락처·주민번호 열\n- 한 사람·한 계약 단위의 값")
    col2.markdown("**보냄**\n- 집계 통계와 이벤트 목록\n- 계산된 지표(A/E, AUC, RMSE)\n- 분야 기준 규칙과 출처")
    st.subheader("비용·안정성 장치 (코드에 반영됨)")
    st.markdown(f"- 틀린 숫자 다시 쓰기 최대 {AI_MAX_RETRIES}번 (`write_verified_report`)\n"
                "- 같은 요약은 저장해 재사용 (`cached_report`, 접속 안에서)\n"
                f"- 접속당 호출 {AI_CALL_LIMIT}번 상한 (`claude_writer`)\n"
                "- 다시 써도 틀리면 AI 리포트 대신 기본 리포트를 쓰라고 안내")
    api_key, model = ai_settings()
    if api_key:
        st.success(f"AI 연결됨 · 모델 {model} · 이 접속에서 쓴 호출 {st.session_state.get('ai_calls', 0)}/{AI_CALL_LIMIT}번")
    else:
        st.info("API 키가 아직 없어요. Streamlit Cloud의 앱 설정 → Secrets에 아래처럼 넣으면 '내 CSV로 돌려 보기' 탭에서 실제 AI를 고를 수 있어요. "
                "키는 코드나 저장소에 넣지 마세요(.streamlit/secrets.toml은 .gitignore에 들어 있어요).")
        st.code('ANTHROPIC_API_KEY = "sk-ant-..."\nAI_MODEL = "claude-haiku-5-5"   # 생략하면 이 값', language="toml")
