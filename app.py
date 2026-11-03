"""VibeData AI 홈 화면 (담당 미정: 11/4 회의에서 지정)"""
import json
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="VibeData AI", layout="wide")

ROOT = Path(__file__).parent


@st.cache_data
def load_snap():
    return json.loads((ROOT / "data" / "snap.json").read_text(encoding="utf-8"))


S = load_snap()
H, B, L = S["home"], S["bearing"], S["lab"]

st.caption("12/6 최종 시연 기준")
st.title("데이터를 올리면, 숫자 근거가 붙은 판단까지")
st.write(
    "VibeData AI는 표 데이터(CSV)를 올리면 통계 엔진이 이상을 찾고, 분야별 분석을 내놓는 웹 앱이에요. "
    "공통 엔진 하나에 제조 PHM·보험 계리·헬스케어 세 분야를 얹었어요."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("베어링 결함 분류 정확도 (부하 조건 분할)", f"{B['byLoad']['acc']:.1%}")
c1.caption(f"무작위로 나누면 {B['random']['acc']:.1%}로 부풀려져요 · 산공")
c2.metric("자동차보험 검증 A/E (건수 / 금액)", f"{H['actAEn']:.2f} / {H['actAEs']:.2f}")
c2.caption(f"순보험료 평균 €{H['actPP']:.0f}/계약년 · 수학과 A")
c3.metric("당뇨 고위험군 선별 AUC", f"{H['auc']:.2f}")
c3.caption(f"공복혈당 제외 · Brier {H['brier']:.3f} · 식품영양학과")
c4.metric("이상 주입 240회, 4σ 이상 F1", f"{L['bench']['big']['f1']:.2f}")
c4.caption(f"튀는 값은 6σ에서도 {L['bench']['grid'][0]['cells'][2]['recall']:.0%}만 잡아요 · 수학과 B")

st.caption("위 숫자는 HTML 시안에 저장된 시뮬레이션 값이에요. 실제 값은 각자 공개 데이터로 다시 계산해요.")

st.subheader("분야별 페이지")
PACKS = [
    ("pages/1_제조PHM.py", "제조 PHM", "산공", "실제 3D 프린터의 이상을 종류까지 가르고, 베어링 결함·남은 수명·가장 싼 정비 시점까지 계산해요."),
    ("pages/2_보험계리.py", "보험 계리", "수학과 A · 팀장", "빈도와 심도를 나눠 모형화하고, 순보험료가 검증자료에서 맞는지(A/E) 확인해요. 지급준비금도 계산해요."),
    ("pages/3_헬스케어.py", "헬스케어", "식품영양학과", "검진 데이터로 위험군을 나누고, 대사증후군 유형과 당뇨 고위험군을 찾아 영양 중재로 이어요."),
    ("pages/4_엔진실험실.py", "엔진 실험실", "수학과 B", "이상을 일부러 심어서 엔진이 무엇을 잡고 무엇을 놓치는지 숫자로 보여 줘요."),
]
for col, (path, name, who, line) in zip(st.columns(2) + st.columns(2), PACKS):
    with col.container(border=True):
        st.markdown(f"**{name}** · {who}")
        st.write(line)
        st.page_link(path, label=f"{name} 페이지 열기")

st.subheader("모든 페이지의 공통 순서")
FLOW = [
    ("업로드", "CSV를 올리거나 예시 데이터를 골라요."),
    ("데이터 점검", "행·열·결측을 확인해요."),
    ("이상 탐지 엔진", "기준에서 벗어난 구간을 이벤트로 잡아요."),
    ("분야별 분석", "그 분야에서만 의미 있는 계산을 해요."),
    ("AI 리포트", "요약만 보내고, 리포트 속 숫자를 계산값과 대조해요. (아직 연결 전)"),
]
for col, (i, (head, text)) in zip(st.columns(5), enumerate(FLOW, 1)):
    col.markdown(f"**{i}. {head}**")
    col.caption(text)

st.subheader("최종 시연 3분 (12/6)")
st.table(
    [
        {"시간": "20초", "주제": "문제", "내용": "현장 담당자는 데이터가 있어도 이상인지 판단하기 어렵고, 범용 AI는 숫자를 지어내요."},
        {"시간": "40초", "주제": "제조", "내용": "3D 프린터 이상 판별 → 베어링 진단 → 잔여수명 → 정비 시점."},
        {"시간": "40초", "주제": "AI 리포트와 토론", "내용": "AI 리포트와 숫자 검증 표시."},
        {"시간": "30초", "주제": "같은 엔진, 다른 분야", "내용": "자동차보험 순보험료와 A/E, 검진 위험군과 당뇨 고위험군."},
        {"시간": "30초", "주제": "신뢰", "내용": "이상 주입 실험: 무엇을 잡고 무엇을 놓치는지."},
        {"시간": "20초", "주제": "확장성", "내용": "새 분야는 설정 파일(팩) 하나로 붙여요."},
    ]
)
