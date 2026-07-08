"""세금 유입 결정적 검증 프로브.

test_tax.py는 전력이 연결되지 않은 도시를 지어(발전소↔주거지 Wire 없음) 인구가
0에 머물러 세수를 검증하지 못했다. 이 프로브는:

  1. 엔진 자체 생성기 ``micro.layGrid``로 도로+전선+RCI zone이 연결된 도시를 짓고,
  2. 전선망 위에 CoalPowerPlant 몇 개를 얹어 전력을 공급한 뒤,
  3. 충분히 긴 시뮬 시간(기본 400 step) 동안 Nil 액션으로 진행하며
     인구·누적 세수를 주기적으로 로깅한다.
  4. 세율 0 vs 20을 비교해 세수-세율 관계가 나타나는지 본다.

인구가 자라면 세수가 유입되어야 정상. 인구가 끝까지 0이면 (전력 미공급 또는
zone 성장에 필요한 시뮬 시간이 에피소드보다 훨씬 길다는 뜻) → 학습 신호 희소성
문제로 escalate.

Docker에서:
    xvfb-run -a python3 -u -m mayorl.verify_tax
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mayorl.config import MayorlConfig
from mayorl.env import BudgetCityEnv


NIL_TOOL_IDX = 19  # config.TOOLS 마지막 = 'Nil'


def build_powered_city(env, map_w):
    """인접을 보장한 클린 도시를 손으로 짓는다 (layGrid는 Rubble/틈투성이라 폐기).

    _diag_power.py 시나리오 C에서 검증됨 — zone 급전 100%, 인구 0→32 성장.
    행 레이아웃 (틈 없이 붙임):
        y=3      : 도로   (주거지 상단 접근)
        y=4..6   : 주거 zone 행 (센터 y=5, x센터 3/6/9/12)
        y=7      : 전선   (주거지 하단 급전)
        y=8..11  : 발전소 (전선에 상단 인접)
    각 주거 zone은 위로 도로·아래로 전선에 동시 인접 → 급전+접근 충족."""
    m = env.micro

    def place(tool, x, y):
        idx = env.micro.tools.index(tool)
        a = idx * (map_w * map_w) + x * map_w + y
        env.step(a)

    def build(x, y, tool):
        m.simTick()          # 배치 사이 틱 (acted 리셋)
        m.doTool(x, y, tool)

    for x in range(2, 14):
        build(x, 3, "Road")
    for cx in (3, 6, 9, 12):
        build(cx, 5, "Residential")
    for x in range(2, 14):
        build(x, 7, "Wire")
    for cx in (3, 10):
        if env.micro.getFunds() >= 3000:
            place("CoalPowerPlant", cx, 9)
    for _ in range(20):
        m.simTick()


def run_one(tax_rate, steps=2000, log_every=200):
    map_w = 16
    config = MayorlConfig(
        map_x=map_w,
        map_y=map_w,
        init_budget=2_000_000,   # 예산 제약이 아니라 '세수 유입'만 보려는 프로브
        max_steps=steps + 100,
        empty_start=True,
        render_gui=False,
        auto_bulldoze=True,      # layGrid가 지형 위에 깔끔히 짓도록
    )
    env = BudgetCityEnv(MAP_X=map_w, MAP_Y=map_w, config=config)
    env.setMapSize((map_w, map_w), rank=0, max_step=steps + 100, render_gui=False)
    env.reset()
    env.micro.setTaxRate(tax_rate)

    build_powered_city(env, map_w)

    nil_action = NIL_TOOL_IDX * (map_w * map_w)  # Nil @ (0,0) = 진짜 no-op
    funds_after_build = env.micro.getFunds()
    cum_rev = 0.0

    print(f"\n=== tax_rate={tax_rate}% | 건설 후 잔액=${funds_after_build:,.0f} ===")
    print(f"  {'step':>4} | {'res':>5} {'com':>4} {'ind':>4} | {'funds':>12} | {'tick_rev':>9} | {'cum_rev':>10}")

    for s in range(1, steps + 1):
        _, _, done, info = env.step(nil_action)
        cum_rev += max(info.get("tick_revenue", 0.0), 0.0)
        if s % log_every == 0 or done:
            print(
                f"  {s:>4} | {info['res_pop']:>5} {info['com_pop']:>4} "
                f"{info['ind_pop']:>4} | {info['funds']:>12,.0f} | "
                f"{info['tick_revenue']:>9,.0f} | {cum_rev:>10,.0f}"
            )
        if done:
            print(f"  (에피소드 종료 @ step {s})")
            break

    tot_pop = env.micro.getResPop() + env.micro.getComPop() + env.micro.getIndPop()
    print(f"  최종 인구={tot_pop}, 누적 세수(추정)=${cum_rev:,.0f}")
    return tot_pop, cum_rev


def main():
    print("=" * 70)
    print("세금 유입 결정적 검증 (연결·전력공급 도시)")
    print("=" * 70)
    results = {}
    for tax in [0, 20]:
        pop, rev = run_one(tax)
        results[tax] = (pop, rev)

    print("\n" + "=" * 70)
    print("요약")
    for tax, (pop, rev) in results.items():
        print(f"  세율 {tax:>2}% → 최종인구={pop:>5}, 누적세수=${rev:,.0f}")
    p0, r0 = results[0]
    p20, r20 = results[20]
    print("-" * 70)
    if p0 == 0 and p20 == 0:
        print("판정: 인구가 끝까지 0 → 전력 미공급 또는 zone 성장에 필요한 시뮬")
        print("      시간이 에피소드보다 훨씬 김. 세수 검증 불가 → 학습 신호 희소성 escalate.")
    elif r20 > r0:
        print("판정: PASS — 인구가 자라고 세율↑ 시 세수↑. 세금 메커니즘 정상 작동.")
    elif r0 > 0 or r20 > 0:
        print("판정: 세수는 유입되나 세율 효과 불명확 — 추가 관찰 필요.")
    else:
        print("판정: 인구는 자랐으나 세수 0 — 세금 징수 경로(연간 게이팅 등) 재확인 필요.")
    print("=" * 70)


if __name__ == "__main__":
    main()
