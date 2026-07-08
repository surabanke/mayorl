"""전력 전달 진단 프로브 — "zone이 실제로 전력을 받는가?"

배경: verify_tax / _diag_sim 에서 layGrid + 발전소 4개 도시가 인구 0에 머문다.
corecontrol.py 건설 경로는 gym-city-origin과 바이트 단위로 동일하고, layGrid는
학습에 안 쓰이는 검증 안 된 헬퍼다. 유력 가설:

  H1: 엔진은 정상. layGrid 전선망과 발전소 4개 위치가 연결 안 돼 zone이 무전력.

이 프로브는 추측 대신 **측정**한다:
  1. layGrid + 발전소 배치 후 전체 맵의 타일 구성을 분류·집계
  2. 각 타일이 전력을 받는지 getPowerGrid로 직접 질의 (getTile과 동일 좌표 규약)
  3. zone 타일 중 전력 받는 비율을 출력
  4. 틱을 진행하며 인구 + 전력 zone 수 변화를 관찰

판정:
  - zone의 전력공급률 ~0%  → H1 확정 (발전소가 zone과 연결 안 됨). 근본원인 = 배치.
  - zone은 전력 받는데 인구 0 → H1 기각. census/도로접근/콜백 등 더 깊은 문제.

Docker에서:
    xvfb-run -a python3 -u -m mayorl._diag_power
"""

import sys
import os

print("[_diag_power] 스크립트 진입 — import 시작", flush=True)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mayorl.config import MayorlConfig
from mayorl.env import BudgetCityEnv
from gym_city.envs.tilemap import zoneFromInt

print("[_diag_power] import 완료", flush=True)


ZONE_KINDS = {"Residential", "Commercial", "Industrial"}
POWER_SRC = {"CoalPowerPlant", "NuclearPowerPlant"}

SYM = {
    "Land": ".", "Water": "~", "Forest": "f",
    "Road": "#", "RoadWire": "%", "Wire": "=",
    "Residential": "R", "Commercial": "C", "Industrial": "I",
    "CoalPowerPlant": "P", "NuclearPowerPlant": "N",
    "Rubble": "x", "Park": "p", "Net": "o",
}


def classify_map(env):
    """logical (i,j) -> (tile_class, powered_bool). getTile과 getPowerGrid를
    동일한 (i+MAP_XS, j+MAP_YS) 좌표 규약으로 질의해 정합성을 보장한다."""
    micro = env.micro
    eng = micro.engine
    MX, MY = micro.MAP_X, micro.MAP_Y
    xs, ys = micro.MAP_XS, micro.MAP_YS
    tiles, power = {}, {}
    for i in range(MX):
        for j in range(MY):
            t = eng.getTile(i + xs, j + ys) & 1023
            tiles[(i, j)] = zoneFromInt(t)
            try:
                power[(i, j)] = int(eng.getPowerGrid(i + xs, j + ys))
            except Exception:
                power[(i, j)] = -1  # query 실패 표시
    return tiles, power, MX, MY


def print_maps(tiles, power, MX, MY):
    print("\n  [타일 맵]                         [전력 맵: X=전력, .=무전력]")
    for j in range(MY):
        trow = "".join(SYM.get(tiles[(i, j)], "?") for i in range(MX))
        prow = "".join(
            ("X" if power[(i, j)] > 0 else ("." if power[(i, j)] == 0 else " "))
            for i in range(MX)
        )
        print(f"  {trow}   {prow}")


def summarize(tiles, power, MX, MY):
    from collections import Counter
    cnt = Counter(tiles.values())
    zone_tiles = [xy for xy, k in tiles.items() if k in ZONE_KINDS]
    zone_powered = [xy for xy in zone_tiles if power[xy] > 0]
    src_tiles = [xy for xy, k in tiles.items() if k in POWER_SRC]
    src_powered = [xy for xy in src_tiles if power[xy] > 0]
    print("\n  [타일 집계]", dict(cnt))
    print(f"  zone 타일 수={len(zone_tiles)}  그중 전력공급={len(zone_powered)} "
          f"({(100*len(zone_powered)/max(1,len(zone_tiles))):.0f}%)")
    print(f"  발전소 타일 수={len(src_tiles)}  그중 전력표시={len(src_powered)}")
    return len(zone_tiles), len(zone_powered)


def place(env, tool, x, y, map_w):
    idx = env.micro.tools.index(tool)
    a = idx * (map_w * map_w) + x * map_w + y
    env.step(a)


def build_repro_city(env, map_w):
    """A: verify_tax / _diag_sim 과 동일한 도시 — layGrid + 발전소 4개(고정 좌표)."""
    env.micro.layGrid(5, 5)
    for (x, y) in [(3, 3), (3, 11), (11, 3), (11, 11)]:
        place(env, "CoalPowerPlant", x, y, map_w)


def build_wired_city(env, map_w):
    """B: layGrid가 실제로 깐 Wire 타일 '위에' 발전소를 얹는다. (layGrid의 Rubble/틈
    때문에 zone 급전 실패로 판명됨 — 기록용으로 보존)."""
    env.micro.layGrid(5, 5)
    tiles, _, MX, MY = classify_map(env)
    wires = [xy for xy, k in tiles.items() if k in ("Wire", "RoadWire")]
    picks = wires[:: max(1, len(wires) // 4)][:4] if wires else []
    for (x, y) in picks:
        if env.micro.getFunds() >= 3000:
            place(env, "CoalPowerPlant", x, y, map_w)
    print(f"  (전선 타일 {len(wires)}개 발견, 발전소 얹은 위치: {picks})")


def build_handcrafted_city(env, map_w):
    """C: layGrid를 버리고 인접을 보장한 최소 정상 도시를 손으로 짓는다.
    행 레이아웃 (틈/Rubble 없이 붙임):
        y=3      : 도로   (주거지 상단에 접근 제공)
        y=4..6   : 주거 zone 행 (센터 y=5, x센터 3/6/9/12 → 틈없이 x=2..13 채움)
        y=7      : 전선   (주거지 하단에 급전)
        y=8..11  : 발전소 (전선에 상단이 인접 → 전선망 에너지화)
    각 주거 zone은 위로 도로, 아래로 전선에 동시 인접 → 급전+접근 모두 충족."""
    m = env.micro

    def build(x, y, tool):
        m.simTick()          # layGrid와 동일하게 배치 사이 틱 (acted 리셋)
        m.doTool(x, y, tool)

    # 접근 도로 (y=3)
    for x in range(2, 14):
        build(x, 3, "Road")
    # 주거 zone 행 (센터 y=5)
    for cx in (3, 6, 9, 12):
        build(cx, 5, "Residential")
    # 급전 전선 (y=7)
    for x in range(2, 14):
        build(x, 7, "Wire")
    # 발전소: 전선 바로 아래(클릭 y=9 → 풋프린트 y=8..11, 상단 y=8 이 전선 y=7 에 인접)
    for cx in (3, 10):
        place(env, "CoalPowerPlant", cx, 9, map_w)
    for _ in range(20):
        m.simTick()


def run_scenario(env, label, build_fn, map_w, n_blocks=12):
    print("\n" + "=" * 70)
    print(f"[{label}]")
    print("=" * 70)
    env.reset()
    env.micro.setTaxRate(7)
    build_fn(env, map_w)

    tiles, power, MX, MY = classify_map(env)
    print_maps(tiles, power, MX, MY)
    summarize(tiles, power, MX, MY)

    eng = env.micro.engine
    print("\n  [틱 진행 중 인구/전력 zone 변화]")
    for blk in range(1, n_blocks + 1):
        for _ in range(100):
            env.micro.simTick()
        tiles, power, MX, MY = classify_map(env)
        zpow = sum(1 for xy, k in tiles.items()
                   if k in ZONE_KINDS and power[xy] > 0)
        r, c, i = env.micro.getResPop(), env.micro.getComPop(), env.micro.getIndPop()
        try:
            dem = eng.getDemands()
        except Exception:
            dem = "n/a"
        print(f"  tick={blk*100:>5} | pop r/c/i={r}/{c}/{i} | "
              f"전력 zone={zpow} | demands={dem}")
    tot = env.micro.getResPop() + env.micro.getComPop() + env.micro.getIndPop()
    print(f"  → 최종 인구={tot}")
    return tot


def main():
    map_w = 16
    config = MayorlConfig(
        map_x=map_w, map_y=map_w, init_budget=2_000_000,
        max_steps=100000, empty_start=True, render_gui=False,
        auto_bulldoze=True,
    )
    env = BudgetCityEnv(MAP_X=map_w, MAP_Y=map_w, config=config)
    env.setMapSize((map_w, map_w), rank=0, max_step=100000, render_gui=False)

    print("#" * 70)
    print("# 전력 전달 진단 프로브")
    print("#" * 70)

    popA = run_scenario(env, "A: 재현 도시 (layGrid + 발전소 4개, 고정 좌표)",
                        build_repro_city, map_w)
    popC = run_scenario(env, "C: 손으로 짠 정상 도시 (도로/주거/전선/발전소 인접 보장)",
                        build_handcrafted_city, map_w)

    print("\n" + "#" * 70)
    print("# 종합 판정")
    print(f"#   A(재현/layGrid) 최종인구={popA}   C(손으로 짠 정상도시) 최종인구={popC}")
    print("#" * 70)
    if popC > 0:
        print("판정: 엔진 정상 — 제대로 연결·급전된 도시는 성장함. 원인은 layGrid가")
        print("      Rubble/틈투성이 도시를 만들어 zone이 전력·도로에 연결 안 된 것.")
        print("      → 재현/검증 스크립트의 도시 빌더를 C 방식으로 교체 후 세수 검증.")
    else:
        print("판정: 인접·급전을 보장한 도시조차 zone 전력=0 또는 인구=0 →")
        print("      전력맵을 보고 어디서 끊기는지 확인. 배치가 아닌 더 깊은 문제")
        print("      (엔진 빌드/도로접근 조건/census) 가능성. origin 대조로 escalate.")
    print("#" * 70)


if __name__ == "__main__":
    main()
