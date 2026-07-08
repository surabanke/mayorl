"""세금 동작 검증 스크립트.

Docker 컨테이너 안에서 실행:
    conda activate isaac
    python -m mayorl.test_tax

확인 사항:
    1. simTick() 후 자금이 변화하는가 (세수 유입)
    2. 인구가 없으면 세수가 0인가 (정상 동작)
    3. 세율 변경이 세수에 영향을 주는가
    4. setFunds()가 호출되지 않으면 잔액이 유지되는가
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mayorl.config import MayorlConfig
from mayorl.env import BudgetCityEnv


def run_test():
    print("=" * 60)
    print("BudgetCityEnv 세금 동작 테스트")
    print("=" * 60)

    config = MayorlConfig(
        map_x=16,
        map_y=16,
        init_budget=50_000,
        max_steps=300,
        empty_start=True,
        render_gui=False,
    )

    env = BudgetCityEnv(MAP_X=config.map_x, MAP_Y=config.map_y, config=config)
    env.setMapSize(
        (config.map_x, config.map_y),
        rank=0,
        max_step=config.max_steps,
        render_gui=False,
    )

    # -------------------------------------------------------
    # 테스트 1: 빈 맵에서 세수가 0인지 확인
    # -------------------------------------------------------
    print("\n[테스트 1] 빈 맵 — 세수가 0이어야 함")
    obs = env.reset()
    start_funds = env.micro.getFunds()
    tax_rate = env.micro.getTaxRate()
    print(f"  초기 자금: ${start_funds:,}")
    print(f"  세율: {tax_rate}%")

    revenues = []
    for step in range(10):
        before = env.micro.getFunds()
        obs, reward, done, info = env.step(0)  # Nil 액션 (아무것도 안 함)
        after = env.micro.getFunds()
        rev = info.get("tick_revenue", after - before)
        revenues.append(rev)

    avg_revenue = sum(revenues) / len(revenues)
    print(f"  10 step 평균 세수: ${avg_revenue:.1f}")
    if abs(avg_revenue) < 10:
        print("  PASS: 인구 없음 → 세수 ≈ 0")
    else:
        print(f"  INFO: 세수 = {avg_revenue:.1f} (엔진 내부 초기화 비용 등 포함 가능)")

    # -------------------------------------------------------
    # 테스트 2: 건물을 몇 개 짓고 세수 변화 관찰
    # -------------------------------------------------------
    print("\n[테스트 2] 주거지 + 전력 건설 후 세수 관찰")
    obs = env.reset()
    start_funds = env.micro.getFunds()

    # 전력소 (CoalPowerPlant) 인덱스 찾기
    tools = env.micro.tools
    print(f"  사용 가능한 도구: {tools}")

    power_idx = tools.index("CoalPowerPlant") if "CoalPowerPlant" in tools else None
    res_idx = tools.index("Residential") if "Residential" in tools else 0
    road_idx = tools.index("Road") if "Road" in tools else None

    map_w = config.map_x

    def make_action(tool_idx, x, y):
        return tool_idx * (map_w * map_w) + x * map_w + y

    built = []

    # 전력소 건설
    if power_idx is not None and env.micro.getFunds() >= 3000:
        a = make_action(power_idx, 2, 2)
        obs, r, done, info = env.step(a)
        built.append(f"CoalPowerPlant @(2,2), 잔액=${info['funds']:,}")

    # 도로 + 전선
    if road_idx is not None:
        for i in range(5):
            if env.micro.getFunds() >= 10:
                a = make_action(road_idx, 5, i + 2)
                obs, r, done, info = env.step(a)

    # 주거지 5칸
    for i in range(5):
        if env.micro.getFunds() >= 100:
            a = make_action(res_idx, 6 + i, 3)
            obs, r, done, info = env.step(a)
            built.append(f"Residential @({6+i},3), 잔액=${info['funds']:,}")

    print(f"  건설 내역:")
    for b in built:
        print(f"    {b}")

    # 50 step 진행하며 세수 추적
    print("\n  50 step 진행 (세수 추적):")
    total_revenue = 0
    pop_start = env.micro.getResPop() + env.micro.getComPop() + env.micro.getIndPop()

    revenues_50 = []
    for step in range(50):
        obs, reward, done, info = env.step(0)  # Nil 액션
        rev = info.get("tick_revenue", 0)
        revenues_50.append(rev)
        total_revenue += max(rev, 0)
        if done:
            print(f"  에피소드 종료 (step {step})")
            break

    pop_end = env.micro.getResPop() + env.micro.getComPop() + env.micro.getIndPop()
    funds_end = env.micro.getFunds()

    print(f"  인구 변화: {pop_start} → {pop_end}")
    print(f"  50 step 총 세수 유입: ${total_revenue:,}")
    print(f"  최종 잔액: ${funds_end:,} (초기 ${start_funds:,})")
    print(f"  세수 발생 step 수: {sum(1 for r in revenues_50 if r > 0)} / 50")

    if total_revenue > 0:
        print("  PASS: 세수가 정상적으로 유입됨")
    else:
        print("  WARN: 세수 0 — cityEvaluation/changeCensus 비활성화 여부 또는 인구 부족 확인 필요")

    # -------------------------------------------------------
    # 테스트 3: 세율 변경 효과
    # -------------------------------------------------------
    print("\n[테스트 3] 세율 0% vs 15% 비교 (각 20 step)")

    for tax in [0, 15]:
        obs = env.reset()
        env.micro.setTaxRate(tax)
        # 동일한 초기 건설
        if power_idx is not None and env.micro.getFunds() >= 3000:
            env.step(make_action(power_idx, 2, 2))
        for i in range(3):
            if env.micro.getFunds() >= 100:
                env.step(make_action(res_idx, 6 + i, 3))

        rev_total = 0
        for _ in range(30):
            _, _, done, info = env.step(0)
            rev_total += max(info.get("tick_revenue", 0), 0)
            if done:
                break
        pop = env.micro.getResPop()
        print(f"  세율 {tax:2d}% → 30 step 세수: ${rev_total:,}  인구: {pop}")

    print("\n세율이 높을수록 세수가 많고, 낮을수록 인구가 유입되는 트레이드오프가 나타나면 정상.")

    # -------------------------------------------------------
    # 테스트 4: setFunds 미호출 확인 (잔액이 유지돼야 함)
    # -------------------------------------------------------
    print("\n[테스트 4] setFunds 미호출 — 건설 비용이 실제로 차감되는지")
    obs = env.reset()
    initial = env.micro.getFunds()
    # 비싼 건물 (CoalPowerPlant = 3000)
    if power_idx is not None:
        a = make_action(power_idx, 8, 8)
        obs, r, done, info = env.step(a)
        after = info["funds"]
        expected_max = initial - 3000  # 적어도 3000은 줄어야 함
        print(f"  발전소 건설 전: ${initial:,} → 후: ${after:,}")
        if after <= expected_max + 50:  # 세수 한 틱 허용
            print("  PASS: 건설 비용 차감 확인")
        else:
            print(f"  FAIL: 잔액이 줄지 않음 (setFunds 리셋이 남아있을 가능성)")
    else:
        print("  SKIP: CoalPowerPlant 없음")

    print("\n" + "=" * 60)
    print("테스트 완료")
    print("=" * 60)


if __name__ == "__main__":
    run_test()
