"""기능적 zone 탐지 검증 (Docker). 클린 클러스터는 functional>0, 흩뿌린 layGrid는 ~0.

Docker:
    docker exec -e PYTHONPATH=/usr/src/app/mayorl/_shim mayorl-diag \
      sh -c 'cd /usr/src/app && xvfb-run -a python3 -u -m mayorl._diag_functional'
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from mayorl.config import MayorlConfig
from mayorl.env import BudgetCityEnv


def _make_env(map_w=16):
    cfg = MayorlConfig(map_x=map_w, map_y=map_w, init_budget=2_000_000,
                       max_steps=100000, empty_start=True, render_gui=False,
                       auto_bulldoze=True)
    env = BudgetCityEnv(MAP_X=map_w, MAP_Y=map_w, config=cfg)
    env.setMapSize((map_w, map_w), rank=0, max_step=100000, render_gui=False)
    env.reset()
    return env, map_w


def _place(env, tool, x, y, map_w):
    idx = env.micro.tools.index(tool)
    env.step(idx * (map_w * map_w) + x * map_w + y)


def build_clean(env, map_w):
    m = env.micro
    def b(x, y, t):
        m.simTick(); m.doTool(x, y, t)
    for x in range(2, 14): b(x, 3, "Road")
    for cx in (3, 6, 9, 12): b(cx, 5, "Residential")
    for x in range(2, 14): b(x, 7, "Wire")
    for cx in (3, 10): _place(env, "CoalPowerPlant", cx, 9, map_w)
    for _ in range(20): m.simTick()


def build_clean_no_road(env, map_w):
    """도로 없는 클러스터: wire+plant+zone만. 전력은 공급되나 도로 접근 없음
    → 부분점수(functional_partial_credit) 검증용."""
    m = env.micro
    def b(x, y, t):
        m.simTick(); m.doTool(x, y, t)
    for cx in (3, 6, 9, 12): b(cx, 5, "Residential")
    for x in range(2, 14): b(x, 7, "Wire")
    for cx in (3, 10): _place(env, "CoalPowerPlant", cx, 9, map_w)
    for _ in range(20): m.simTick()


def build_laygrid(env, map_w):
    env.micro.layGrid(5, 5)
    for (x, y) in [(3, 3), (3, 11), (11, 3), (11, 11)]:
        _place(env, "CoalPowerPlant", x, y, map_w)


def main():
    print("=" * 60)
    env, map_w = _make_env()
    build_clean(env, map_w)
    f_clean = env._calc_functional_zone_fraction()
    print(f"클린 클러스터(도로+전력) functional_fraction = {f_clean:.4f}  (기대: 최대)")

    env, map_w = _make_env()
    build_clean_no_road(env, map_w)
    f_noroad = env._calc_functional_zone_fraction()
    print(f"무도로 클러스터(전력만)  functional_fraction = {f_noroad:.4f}  (기대: ≈절반 가중)")

    env, map_w = _make_env()
    build_laygrid(env, map_w)
    f_lay = env._calc_functional_zone_fraction()
    print(f"layGrid(흩뿌림)          functional_fraction = {f_lay:.4f}  (기대: ~0)")
    print("=" * 60)
    ok = (f_clean > f_noroad > f_lay) and (f_noroad > 0.0) and (f_lay < 0.01)
    print(f"순서 검증: full({f_clean:.4f}) > no-road({f_noroad:.4f}) > layGrid({f_lay:.4f})")
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
