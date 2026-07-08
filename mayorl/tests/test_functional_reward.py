"""functional shaping 리워드 유닛테스트 (엔진 불필요, conda isaac에서 실행)."""
from mayorl.config import RewardWeights, MayorlConfig
from mayorl.reward import RewardCalculator, StepInfo, functional_potential

_ZERO_METRICS = {'res_pop': 0, 'com_pop': 0, 'ind_pop': 0, 'mayor_rating': 0}


def _base_kwargs():
    return dict(
        city_metrics=dict(_ZERO_METRICS),
        last_city_metrics=dict(_ZERO_METRICS),
        current_funds=100.0, previous_funds=100.0, init_budget=100.0,
    )


def test_weight_renamed():
    w = RewardWeights()
    assert w.functional_shaping == 2.0
    assert not hasattr(w, 'infrastructure_shaping')


def test_road_access_radius_default():
    assert MayorlConfig().road_access_radius == 2


def test_functional_potential_identity():
    assert functional_potential(0.0) == 0.0
    assert functional_potential(0.25) == 0.25


def test_compute_rewards_functional_progress():
    cfg = MayorlConfig()
    calc = RewardCalculator(cfg)
    info = StepInfo(**_base_kwargs(),
                    functional_fraction=0.5, last_functional_fraction=0.0)
    r = calc.compute(info)
    assert abs(r - cfg.reward_weights.functional_shaping * 0.5) < 1e-9


def test_compute_no_shaping_when_flat():
    cfg = MayorlConfig()
    calc = RewardCalculator(cfg)
    info = StepInfo(**_base_kwargs(),
                    functional_fraction=0.3, last_functional_fraction=0.3)
    assert calc.compute(info) == 0.0


def test_stepinfo_has_no_coverage_fields():
    info = StepInfo(**_base_kwargs())
    for dead in ('power_coverage', 'road_coverage',
                 'last_power_coverage', 'last_road_coverage'):
        assert not hasattr(info, dead), f'{dead} should be removed'


def main():
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print(f'PASS {name}')
            passed += 1
    print(f'ALL PASS ({passed})')


if __name__ == '__main__':
    main()
