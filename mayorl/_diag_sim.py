"""엔진 시뮬레이션 진행/성장 진단.

질문: mayorl에서 인구가 0인 것이 (A) 스텝당 시뮬 시간이 너무 짧아 게임연도(세금
주기)가 에피소드에 안 들어오는 문제인가, 아니면 (B) zone이 근본적으로 발전 안 하는
(전력/수요) 문제인가?

방법: 연결·전력공급 도시를 지은 뒤 engine.simTick()을 대량(6000회) 직접 호출하며
cityTime, RCI 수요, 인구, 자금을 주기적으로 로깅한다. 충분한 틱에서 인구가 오르면
(A), 끝까지 0이면 (B).
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mayorl.config import MayorlConfig
from mayorl.env import BudgetCityEnv


def main():
    map_w = 16
    config = MayorlConfig(
        map_x=map_w, map_y=map_w, init_budget=2_000_000,
        max_steps=100000, empty_start=True, render_gui=False,
        auto_bulldoze=True,
    )
    env = BudgetCityEnv(MAP_X=map_w, MAP_Y=map_w, config=config)
    env.setMapSize((map_w, map_w), rank=0, max_step=100000, render_gui=False)
    env.reset()
    env.micro.setTaxRate(7)

    # 연결 도시 + 발전소
    env.micro.layGrid(5, 5)
    for (x, y) in [(3, 3), (3, 11), (11, 3), (11, 11)]:
        idx = env.micro.tools.index("CoalPowerPlant")
        a = idx * (map_w * map_w) + x * map_w + y
        env.step(a)

    eng = env.micro.engine

    def snapshot(tag):
        try:
            dem = eng.getDemands()
        except Exception as e:
            dem = ("err", e)
        ct = getattr(eng, "cityTime", "n/a")
        r, c, i = env.micro.getResPop(), env.micro.getComPop(), env.micro.getIndPop()
        funds = env.micro.getFunds()
        print(f"  [{tag:>7}] cityTime={ct} | pop r/c/i={r}/{c}/{i} "
              f"| demands={dem} | funds={funds:,.0f}", flush=True)

    print("=" * 70)
    print("엔진 simTick 대량 진단 (직접 호출)")
    print("=" * 70)
    snapshot("build")
    total = 6000
    for n in range(1, total + 1):
        env.micro.simTick()
        if n % 500 == 0:
            snapshot(str(n))
    print("=" * 70)
    print("해석: pop가 어느 시점부터 오르면 (A) 시뮬시간/에피소드길이 문제.")
    print("      cityTime이 안 변하면 updateDate 무효, 끝까지 pop 0+demands<=0이면")
    print("      (B) 전력/수요 미충족(zone 발전 조건 불만족).")
    print("=" * 70)


if __name__ == "__main__":
    main()
