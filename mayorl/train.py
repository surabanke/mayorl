"""mayorl/train.py -- Training script for budget-constrained city RL agent.

Uses the original model/algo infrastructure from ralph-city, but with:
  - BudgetCityEnv (funds not reset each step)
  - BudgetCurriculum (progressive budget reduction)
  - RewardCalculator (fiscal-health aware reward)

Usage:
    python -m mayorl.train [OPTIONS]
    python -m mayorl.train --algo ppo --map-width 16 --num-processes 4
"""

import argparse
import copy
import logging
import os
import sys
import time
from collections import deque

import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from model import Policy
from storage import RolloutStorage
from utils import get_vec_normalize
import algo

from .config import MayorlConfig
from .curriculum import BudgetCurriculum, EpisodeStats
from .wrapper import make_mayorl_env
from .utils import (
    EpisodeTracker,
    format_episode_log,
    format_curriculum_log,
    seed_everything,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Storage fix
# ---------------------------------------------------------------------------

class MayorlRolloutStorage(RolloutStorage):
    """원본 RolloutStorage의 value_preds 크기 버그를 바로잡는 서브클래스.

    원본 storage.py는 ``value_preds``를 ``num_steps`` 크기로 할당하지만,
    ``algo/ppo.py``는 ``returns[:-1] - value_preds[:-1]`` (둘 다 num_steps)을
    기대하고 GAE 분기도 ``value_preds[-1]``/``value_preds[step+1]``을 참조한다.
    표준 구현처럼 ``num_steps+1``로 재할당하여 PPO와 GAE 모두 정상 동작시킨다.
    원본 파일은 건드리지 않는다.
    """

    def __init__(self, num_steps, num_processes, obs_shape, action_space,
                 recurrent_hidden_state_size, args=None):
        super().__init__(num_steps, num_processes, obs_shape, action_space,
                         recurrent_hidden_state_size, args=args)
        # value_preds를 num_steps+1로 재할당 (원본은 num_steps).
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)


# ---------------------------------------------------------------------------
# Argument parsing (mayorl-specific, simpler than original)
# ---------------------------------------------------------------------------

def get_mayorl_args():
    parser = argparse.ArgumentParser(description='MayoRL: Budget-constrained city RL')

    # Algorithm
    parser.add_argument('--algo', default='ppo', choices=['a2c', 'ppo'],
                        help='RL algorithm (default: ppo)')
    parser.add_argument('--lr', type=float, default=7e-4)
    parser.add_argument('--eps', type=float, default=1e-5)
    parser.add_argument('--alpha', type=float, default=0.99)
    parser.add_argument('--gamma', type=float, default=0.99)
    # GAE 사용. 원본 storage.py는 value_preds를 num_steps로 잡아 ppo.py/GAE와
    # 어긋나는 버그가 있는데, mayorl은 MayorlRolloutStorage(아래)에서
    # value_preds를 num_steps+1로 바로잡아 이를 해소한다.
    parser.add_argument('--use-gae', action='store_true', default=True)
    parser.add_argument('--tau', type=float, default=0.95)
    parser.add_argument('--entropy-coef', type=float, default=0.01)
    parser.add_argument('--value-loss-coef', type=float, default=0.5)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)

    # PPO specific
    parser.add_argument('--clip-param', type=float, default=0.2)
    parser.add_argument('--ppo-epoch', type=int, default=4)
    parser.add_argument('--num-mini-batch', type=int, default=4)

    # Training
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--num-processes', type=int, default=4,
                        help='parallel environments (default: 4)')
    parser.add_argument('--num-steps', type=int, default=5,
                        help='rollout length per update')
    parser.add_argument('--num-frames', type=int, default=5_000_000,
                        help='total training frames')
    parser.add_argument('--no-cuda', action='store_true', default=False)

    # Environment
    parser.add_argument('--map-width', type=int, default=16)
    parser.add_argument('--max-step', type=int, default=200)
    parser.add_argument('--init-budget', type=int, default=None,
                        help='override initial budget (default: from curriculum Phase 0)')
    # 탐색 부트스트랩 (pop 가속 계획 Part A)
    parser.add_argument('--random-builds', action='store_true', default=False,
                        help='reset마다 랜덤 구조물 프리빌드 (원본 gym-city 기본값 복원, '
                             '시작상태 다양화로 탐색 보조)')
    parser.add_argument('--auto-bulldoze', action='store_true', default=False,
                        help='autoBulldoze=True로 A/B (건설 마찰 제거; 기본 False 유지)')
    parser.add_argument('--prebuild-backbone', action='store_true', default=False,
                        help='에피소드 시작 시 도로+전선+발전소 백본을 static 프리빌드 '
                             '(에이전트 과제를 "전선 옆 zone 배치"로 축소)')

    # Model
    parser.add_argument('--model', default='FullyConv',
                        help='model architecture (default: FullyConv)')
    parser.add_argument('--n-recs', type=int, default=3)
    parser.add_argument('--n-chan', type=int, default=64)
    parser.add_argument('--recurrent-policy', action='store_true', default=False)

    # Logging / saving
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--save-interval', type=int, default=100)
    parser.add_argument('--save-dir', default='trained_models/mayorl')
    parser.add_argument('--load-dir', default=None)
    parser.add_argument('--experiment-name', default='')

    # Curriculum
    parser.add_argument('--no-curriculum', action='store_true', default=False,
                        help='disable budget curriculum (use fixed budget)')

    # LLM/VLM Mayor — Qwen2.5-VL-7B-Instruct 4bit (Phase 2/3)
    parser.add_argument('--use-vlm', action='store_true', default=False,
                        help='Qwen2.5-VL 시장 에이전트 활성화 (보조 리워드 + 전략 목표)')
    parser.add_argument('--vlm-eval-every', type=int, default=50,
                        help='몇 update마다 Qwen 시장을 호출할지 (기본: 50)')
    parser.add_argument('--vlm-reward-weight', type=float, default=0.5,
                        help='VLM 보조 리워드 가중치 (기본: 0.5)')
    parser.add_argument('--vlm-with-screenshot', action='store_true', default=False,
                        help='도시 스크린샷을 VLM에 전달 (vision 모드, 기본: text-only)')
    parser.add_argument('--vlm-apply-tax', action='store_true', default=False,
                        help='Qwen 권장 세율을 env에 자동 적용')

    # FractalNet (pass-through for compatibility)
    parser.add_argument('--rule', default='extend')
    parser.add_argument('--intra-shr', action='store_true', default=False)
    parser.add_argument('--inter-shr', action='store_true', default=False)
    parser.add_argument('--drop-path', action='store_true', default=False)
    parser.add_argument('--auto-expand', action='store_true', default=False)

    # 원본 model.py / storage.py / algo 가 참조하는 필드 (arguments.py 기본값과 동일)
    parser.add_argument('--env-name', default='MicropolisEnv-v0',
                        help='원본 모듈 호환용 (RolloutStorage/Policy가 참조)')
    parser.add_argument('--power-puzzle', action='store_true', default=False)
    parser.add_argument('--val-kern', type=int, default=3)
    parser.add_argument('--prebuild', action='store_true', default=False)
    parser.add_argument('--beta', type=float, default=0.2,
                        help='ICM inverse/forward balance (원본 호환)')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args


# ---------------------------------------------------------------------------
# Vectorized environment creation
# ---------------------------------------------------------------------------

def make_mayorl_vec_envs(config, curriculum, num_processes, device):
    """Create multiple BudgetCityEnv instances for parallel training.

    Since the original make_vec_envs is tightly coupled to gym registration,
    we create environments manually and wrap them in a simple vectorized
    interface using SubprocVecEnv pattern.
    """
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

    def make_env(rank):
        def _init():
            env = make_mayorl_env(
                config=config,
                curriculum=curriculum,
                action_mask=True,
                # 사후 치환 OFF: 지불불가 액션을 Nil로 몰래 바꾸지 않고 env로 흘려보내
                # BudgetCityEnv.step이 no-op+action_failed=True로 처리 → invalid penalty가
                # 실제로 발동한다(문제 #2). action_masks()는 여전히 노출됨(로짓 마스킹 도입 대비).
                replace_invalid=False,
                # 관측 채널 스케일 정규화(문제 #3): 원본 RCI/밀도 채널(원값)과 mayorl
                # budget 채널([-1,1])의 스케일 혼재를 running mean/std로 통일.
                obs_norm=config.obs_norm,
                rank=rank,
            )
            return env
        return _init

    if num_processes == 1:
        envs = DummyVecEnv([make_env(0)])
    else:
        envs = SubprocVecEnv([make_env(i) for i in range(num_processes)])

    return envs


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class MayorlTrainer:
    """Training loop for budget-constrained city RL."""

    def __init__(self, args=None):
        if args is None:
            args = get_mayorl_args()
        self.args = args

        # Seed
        seed_everything(args.seed)
        torch.set_num_threads(1)

        # Device
        self.device = torch.device('cuda:0' if args.cuda else 'cpu')

        # Config
        config = MayorlConfig(
            map_x=args.map_width,
            map_y=args.map_width,
            max_steps=args.max_step,
        )
        if args.init_budget is not None:
            config.init_budget = args.init_budget
        # VLM 사용 여부 → 관측 goal 채널 포함 여부를 결정 (비-VLM이면 goal 채널 제외)
        config.use_vlm = args.use_vlm
        # 탐색 부트스트랩 플래그 (pop 가속 계획 Part A)
        if args.random_builds:
            config.random_builds = True
        if args.auto_bulldoze:
            config.auto_bulldoze = True
        if args.prebuild_backbone:
            config.prebuild_backbone = True
        self.config = config

        # Curriculum
        if args.no_curriculum:
            self.curriculum = None
        else:
            self.curriculum = BudgetCurriculum(config)

        # Save dir
        if not args.experiment_name:
            import datetime
            args.experiment_name = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

        args.save_dir = os.path.join(
            args.save_dir,
            f'{args.algo}_{args.model}_w{args.map_width}_{args.experiment_name}'
        )
        os.makedirs(args.save_dir, exist_ok=True)

        # Environment (single env for setup, then vectorized)
        logger.info('Creating %d parallel environments (map=%dx%d)...',
                     args.num_processes, args.map_width, args.map_width)
        self.envs = make_mayorl_vec_envs(
            config, self.curriculum, args.num_processes, self.device
        )

        # Model
        obs_shape = self.envs.observation_space.shape
        action_space = self.envs.action_space
        num_actions = len(config.curriculum_phases)  # placeholder

        # For Micropolis-style envs, num_actions = num_tools
        from .config import TOOLS
        num_actions = len(TOOLS)

        in_w = obs_shape[1] if len(obs_shape) == 3 else 1
        in_h = obs_shape[2] if len(obs_shape) == 3 else 1
        num_inputs = obs_shape[0]

        self.actor_critic = Policy(
            obs_shape, action_space,
            base_kwargs={
                'map_width': args.map_width,
                'num_actions': num_actions,
                'recurrent': args.recurrent_policy,
                'prebuild': False,
                'in_w': in_w, 'in_h': in_h,
                'num_inputs': num_inputs,
                'out_w': args.map_width, 'out_h': args.map_width,
            },
            curiosity=False, algo=args.algo,
            model=args.model, args=args,
        )
        self.actor_critic.to(self.device)

        # Agent (optimizer)
        self.agent = self._init_agent()

        # Rollout storage
        recurrent_hidden_state_size = self.actor_critic.recurrent_hidden_state_size
        self.rollouts = MayorlRolloutStorage(
            args.num_steps, args.num_processes,
            obs_shape, action_space,
            recurrent_hidden_state_size, args=args,
        )

        # Load checkpoint if available
        self.n_frames = 0
        self._load_checkpoint()

        # Initialize rollouts
        obs = self.envs.reset()
        obs_tensor = torch.FloatTensor(obs)
        self.rollouts.obs[0].copy_(obs_tensor)
        self.rollouts.to(self.device)

        # 진단(1회): 정책 분포의 실제 로짓 수 K 실측.
        # 학습 로그의 entropy가 ln(action_space.n)을 초과하면 K 불일치(액션 인덱싱
        # 정합 문제) 신호이므로, 시작 시점에 실측해 둔다.
        with torch.no_grad():
            _v, _feats, _h = self.actor_critic.base(
                self.rollouts.obs[0],
                self.rollouts.recurrent_hidden_states[0],
                self.rollouts.masks[0],
            )
            _dist = self.actor_critic.dist(_feats)
            _k = int(_dist.probs.shape[-1])
            _ent0 = float(_dist.entropy().mean())
        logger.info(
            '정책 분포 진단: K=%d, env action_space.n=%d, ln(K)=%.4f, 초기 entropy=%.4f%s',
            _k, self.envs.action_space.n, np.log(_k), _ent0,
            '' if _k == self.envs.action_space.n else '  ⚠ K != action_space.n — 인덱싱 정합 문제!',
        )

        # Tracking
        self.episode_tracker = EpisodeTracker(window_size=100)
        self.episode_rewards = deque(maxlen=100)
        self._functional_recent = deque(maxlen=100)
        self.start_time = time.time()

        # VLM Mayor (optional)
        self.vlm_mayor = None
        self._vlm_bonus = 0.0  # 현재 VLM auxiliary reward 값
        self._last_vlm_infos: list = []  # 마지막 rollout의 infos
        if args.use_vlm:
            from .vlm_mayor import VLMMayor
            self.vlm_mayor = VLMMayor(
                eval_every=args.vlm_eval_every,
                text_only=not args.vlm_with_screenshot,
            )
            logger.info('Qwen2.5-VL 시장 활성화: eval_every=%d updates, vision=%s',
                        args.vlm_eval_every, args.vlm_with_screenshot)

    def _init_agent(self):
        args = self.args
        if args.algo == 'a2c':
            return algo.A2C(
                self.actor_critic, args.value_loss_coef,
                args.entropy_coef, lr=args.lr, eps=args.eps,
                alpha=args.alpha, max_grad_norm=args.max_grad_norm,
                curiosity=False, args=args,
            )
        elif args.algo == 'ppo':
            return algo.PPO(
                self.actor_critic, args.clip_param,
                args.ppo_epoch, args.num_mini_batch,
                args.value_loss_coef, args.entropy_coef,
                lr=args.lr, eps=args.eps,
                max_grad_norm=args.max_grad_norm,
            )

    def _load_checkpoint(self):
        args = self.args
        load_path = args.load_dir or args.save_dir
        checkpoint_path = os.path.join(load_path, 'mayorl_checkpoint.tar')
        if os.path.exists(checkpoint_path):
            logger.info('Loading checkpoint from %s', checkpoint_path)
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.actor_critic.load_state_dict(checkpoint['model_state_dict'])
            self.agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.n_frames = checkpoint.get('n_frames', 0)
            if self.curriculum and 'curriculum_phase' in checkpoint:
                self.curriculum.force_phase(checkpoint['curriculum_phase'])
            logger.info('Resumed from frame %d', self.n_frames)

    def train(self):
        """Main training loop."""
        args = self.args
        total_updates = int(args.num_frames) // (args.num_steps * args.num_processes)

        logger.info('Starting training: %d updates, %d total frames', total_updates, args.num_frames)
        if self.curriculum:
            logger.info('Curriculum enabled: %d phases', self.curriculum.num_phases)
            logger.info('Phase 0: %s (budget=%d)',
                        self.curriculum.current_phase.name,
                        self.curriculum.current_budget)

        for update_i in range(total_updates):
            # Collect rollouts
            self._last_vlm_infos = []
            for step in range(args.num_steps):
                self._collect_step(step)

            # VLM Mayor 평가 (text-only, 집계된 stats 기반)
            if self.vlm_mayor and update_i % args.vlm_eval_every == 0:
                self._run_vlm_evaluation(update_i)

            # Compute returns
            with torch.no_grad():
                next_value = self.actor_critic.get_value(
                    self.rollouts.obs[-1],
                    self.rollouts.recurrent_hidden_states[-1],
                    self.rollouts.masks[-1],
                ).detach()

            self.rollouts.compute_returns(
                next_value, args.use_gae, args.gamma, args.tau
            )

            # Policy update
            value_loss, action_loss, dist_entropy = self.agent.update(self.rollouts)
            self.rollouts.after_update()

            self.n_frames += args.num_steps * args.num_processes

            # Logging
            if update_i % args.log_interval == 0:
                self._log_progress(update_i, value_loss, action_loss, dist_entropy)

            # Save
            if update_i % args.save_interval == 0 and update_i > 0:
                self._save_checkpoint()

        # Final save
        self._save_checkpoint()
        logger.info('Training complete. Total frames: %d', self.n_frames)

    def _collect_step(self, step):
        """Collect one step of experience from all parallel environments."""
        with torch.no_grad():
            value, action, action_log_probs, recurrent_hidden_states = \
                self.actor_critic.act(
                    self.rollouts.obs[step],
                    self.rollouts.recurrent_hidden_states[step],
                    self.rollouts.masks[step],
                    deterministic=False,
                )

        # Environment step
        actions_np = action.squeeze(1).cpu().numpy()
        obs, reward, done, infos = self.envs.step(actions_np)

        # 관측성: 스텝별 functional_fraction 평균을 최근 버퍼에 기록
        fr = [info.get('functional_fraction', 0.0) for info in infos]
        if fr:
            self._functional_recent.append(sum(fr) / len(fr))

        # Track episode completions
        for i, info in enumerate(infos):
            if done[i]:
                budget_mon = info.get('budget_monitor', {})
                if budget_mon:
                    self.episode_tracker.record(budget_mon)
                    self.episode_rewards.append(budget_mon.get('total_reward', 0))

        # VLM bonus 적용 및 infos 누적
        if self.vlm_mayor:
            self._last_vlm_infos.extend(infos)

        # Store experience
        obs_tensor = torch.FloatTensor(obs)
        reward_np = np.array(reward, dtype=np.float32) + self._vlm_bonus
        reward_tensor = torch.FloatTensor(reward_np).unsqueeze(1)
        masks = torch.FloatTensor([[0.0] if d else [1.0] for d in done])

        self.rollouts.insert(
            obs_tensor, recurrent_hidden_states, action,
            action_log_probs, value, reward_tensor, masks,
        )

    def _run_vlm_evaluation(self, update_i: int) -> None:
        """VLM 시장 평가를 실행하고 bonus reward와 strategic goal을 갱신한다.

        vectorized env에 직접 접근할 수 없으므로, 마지막 rollout의
        infos 집계치로 CityStats를 구성하고 text_only 모드로 호출한다.
        """
        from .vlm_mayor import CityStats, MayorEvaluation

        args = self.args
        infos = self._last_vlm_infos
        if not infos:
            return

        # 마지막 rollout infos에서 평균 수치 집계.
        # env.py가 매 step info에 전체 도시 수치를 기록하므로, 벡터 env에서도
        # 실제 R/C/I 인구·수요·커버리지·지지율을 집계할 수 있다.
        def _mean(key, default=0.0):
            vals = [i.get(key) for i in infos if key in i]
            return float(np.mean(vals)) if vals else float(default)

        def _sum(key, default=0.0):
            vals = [i.get(key) for i in infos if key in i]
            return float(np.sum(vals)) if vals else float(default)

        # 마지막 step의 RCI 수요(가능하면 직접 조회, 아니면 0)
        try:
            demands = self.envs.env_method('_get_demands')[0]
        except Exception:
            demands = (0.0, 0.0, 0.0)

        stats = CityStats(
            res_pop=int(_mean('res_pop')),
            com_pop=int(_mean('com_pop')),
            ind_pop=int(_mean('ind_pop')),
            funds=int(_mean('funds')),
            init_budget=self.config.init_budget,
            tax_rate=int(_mean('tax_rate', 7)),
            mayor_rating=int(_mean('mayor_rating')),
            res_demand=float(demands[0]),
            com_demand=float(demands[1]),
            ind_demand=float(demands[2]),
            power_coverage=_mean('power_coverage'),
            road_coverage=_mean('road_coverage'),
            tax_revenue=int(_sum('tick_revenue')),
            failed_builds=int(_sum('failed_build')),
        )

        # VLM 호출 (text_only — 학습 중 screenshot은 생략)
        evaluation = self.vlm_mayor._call_model(stats, image_path=None)
        self.vlm_mayor._last_eval = evaluation
        self.vlm_mayor._eval_count += 1

        # bonus reward 갱신
        self._vlm_bonus = (evaluation.score - 5.0) / 10.0 * args.vlm_reward_weight

        # strategic goal을 모든 env에 주입
        goal_vec = evaluation.goal_vector
        try:
            self.envs.env_method('set_strategic_goal', goal_vec)
        except Exception:
            pass  # DummyVecEnv는 env_method 미지원 가능

        # VLM 권장 세율 적용 (옵션)
        if args.vlm_apply_tax:
            try:
                self.envs.env_method(
                    'set_tax_rate_via_vlm', evaluation.tax_recommendation
                )
            except Exception:
                pass

        logger.info(
            '[VLM Mayor] update=%d score=%.1f bonus=%.3f goal=%s tax_rec=%d | %s',
            update_i, evaluation.score, self._vlm_bonus,
            evaluation.strategic_goal, evaluation.tax_recommendation,
            evaluation.briefing[:60],
        )

    def _log_progress(self, update_i, value_loss, action_loss, dist_entropy):
        """Log training progress."""
        elapsed = time.time() - self.start_time
        fps = int(self.n_frames / max(elapsed, 1))

        msg_parts = [
            f'Update {update_i}',
            f'frames={self.n_frames}',
            f'FPS={fps}',
        ]

        if self.episode_rewards:
            msg_parts.append(
                f'reward={np.mean(self.episode_rewards):.2f}'
                f'({np.min(self.episode_rewards):.2f}~{np.max(self.episode_rewards):.2f})'
            )

        summary = self.episode_tracker.summary()
        if summary['total_episodes'] > 0:
            msg_parts.extend([
                f'pop={summary["avg_population"]:.0f}',
                f'bankrupt={summary["bankruptcy_rate"]:.1%}',
                f'invalid={summary["avg_invalid_actions"]:.1f}',
            ])

        if self._functional_recent:
            msg_parts.append(
                f'func={np.mean(self._functional_recent):.4f}'
            )

        msg_parts.extend([
            f'v_loss={value_loss:.4f}',
            f'a_loss={action_loss:.4f}',
            f'entropy={dist_entropy:.4f}',
        ])

        logger.info(' | '.join(msg_parts))

        # Curriculum status
        if self.curriculum and update_i % (self.args.log_interval * 5) == 0:
            logger.info('\n%s', format_curriculum_log(self.curriculum.get_stats_summary()))

    def _save_checkpoint(self):
        """Save model, optimizer, and curriculum state."""
        save_path = os.path.join(self.args.save_dir, 'mayorl_checkpoint.tar')

        save_model = copy.deepcopy(self.actor_critic)
        if self.args.cuda:
            save_model.cpu()

        checkpoint = {
            'n_frames': self.n_frames,
            'model_state_dict': save_model.state_dict(),
            'optimizer_state_dict': self.agent.optimizer.state_dict(),
            'args': self.args,
            'config': self.config,
        }
        if self.curriculum:
            checkpoint['curriculum_phase'] = self.curriculum.current_phase_index

        torch.save(checkpoint, save_path)
        logger.info('Checkpoint saved: %s (frames=%d)', save_path, self.n_frames)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    trainer = MayorlTrainer()
    trainer.train()


if __name__ == '__main__':
    main()
