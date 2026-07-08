Baseline 1: 기존 Fractal RL, unlimited funds, population reward
Baseline 2: Fractal RL + limited budget + tax, VLM 없음
Ours 1: RL + VLM auditor reward
Ours 2: RL + VLM high-level option
Ours 3: RL + VLM adversarial crisis curriculum

평가지표는 final population 하나만 보면 안 되고:
최종 인구
평균 인구
인구 변동성
파산 횟수
생존 개월 수
population per dollar
tax revenue stability
unpowered zone ratio
road accessibility
RCI demand balance
VLM urban quality score

재미 포인트를 하나 더 넣고 싶으면, **“시장 브리핑”**을 만들면 좋겠어. 매 1년마다 VLM이 도시 이미지를 보고 이런 보고서를 작성해.

현재 도시는 주거지가 빠르게 증가했지만, 상업/산업 구역이 부족해 장기 수요 균형이 불안정합니다. 
세율은 단기적으로 7%까지 올려도 되지만, 주거 수요가 하락하면 다시 낮춰야 합니다. 
다음 20 step 동안은 신규 확장보다 북쪽 주거지의 도로 접근성과 전력 연결을 우선해야 합니다.

RL은 도시를 짓는 실무자고, VLM 시장은 도시 운영 결과를 평가하는 사람이야.
VLM: 전력 안 이어졌으니까 power line 지어.
RL: power line 지음.

이러면 RL이 이미 배울 수 있는 걸 VLM이 또 말하는 거라 별로야.
도움 되는 구조는 이런 거야.
RL: 1년 동안 도시를 운영함.
VLM 시장: 이 도시는 인구는 높지만 세금 의존도가 크고, 발전소 하나에 너무 의존하며, 주거지-산업지 분리가 안 되어 장기적으로 불안정하다.
RL: 다음 episode에서 그런 도시가 낮은 평가를 받는다는 걸 학습함.

1. RL agent가 도시를 몇 달/몇 년 운영한다.
2. 도시 스크린샷 + 수치 정보가 저장된다.
   - population
   - funds
   - tax rate
   - monthly income
   - R/C/I demand
   - powered ratio
   - crime/pollution/traffic 가능하면 포함
3. VLM 시장이 도시를 평가한다.
   - 승인 / 조건부 승인 / 거부
   - 장점
   - 위험요인
   - 시장 점수
4. 그 평가를 이용해 auxiliary reward 또는 preference reward model을 만든다.
5. RL은 population reward + budget reward + mayor reward를 같이 보며 학습한다.



docker build -f Dockerfile.mayorl -t mayorl:latest .
컨테이너 실행 커맨드: 
docker run --gpus all \
    -v "C:\Users\hel08\Documents\PlayGround\ralph-city:/usr/src/app" \
    -v "C:\Users\hel08\.cache\huggingface:/root/.cache/huggingface" \
    mayorl:latest

세금 테스트
xvfb-run -a python3 -m mayorl.test_tax