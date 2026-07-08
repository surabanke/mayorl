"""mayorl/evaluate.py -- Evaluation script for trained budget-constrained city agents.

Loads a trained checkpoint and runs deterministic episodes, reporting:
  - Final population, funds, mayor rating
  - Build efficiency (population per dollar spent)
  - Bankruptcy rate
  - Action distribution (tool usage)

Usage:
    python -m mayorl.evaluate --load-dir trained_models/mayorl/ppo_FullyConv_w16_...
    python -m mayorl.evaluate --load-dir ... --num-episodes 50 --render
"""

import argparse
import logging
import os
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from model import Policy

from .config import MayorlConfig, TOOLS
from .curriculum import BudgetCurriculum
from .wrapper import make_mayorl_env
from .utils import (
    EpisodeTracker,
    compute_build_efficiency,
    format_episode_log,
    seed_everything,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)


def get_eval_args():
    parser = argparse.ArgumentParser(description='MayoRL Evaluation')
    parser.add_argument('--load-dir', required=True, help='checkpoint directory')
    parser.add_argument('--num-episodes', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--deterministic', action='store_true', default=True)
    parser.add_argument('--render', action='store_true', default=False)
    parser.add_argument('--budget', type=int, default=None,
                        help='override budget for evaluation')
    parser.add_argument('--phase', type=int, default=None,
                        help='evaluate at a specific curriculum phase')
    return parser.parse_args()


def evaluate(args=None):
    if args is None:
        args = get_eval_args()

    seed_everything(args.seed)
    device = torch.device('cpu')

    # Load checkpoint
    checkpoint_path = os.path.join(args.load_dir, 'mayorl_checkpoint.tar')
    if not os.path.exists(checkpoint_path):
        logger.error('Checkpoint not found: %s', checkpoint_path)
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    train_args = checkpoint['args']
    config = checkpoint.get('config', MayorlConfig())

    # Override budget if specified
    if args.budget is not None:
        config.init_budget = args.budget
    elif args.phase is not None:
        if args.phase < len(config.curriculum_phases):
            config.init_budget = config.curriculum_phases[args.phase].init_budget
            logger.info('Evaluating at phase %d budget: %d', args.phase, config.init_budget)

    config.render_gui = args.render

    # Create environment
    env = make_mayorl_env(
        config=config,
        curriculum=None,  # no curriculum during eval
        action_mask=True,
        # 학습과 동일하게 관측 정규화 적용(문제 #3). 단 running 통계는 체크포인트에
        # 저장되지 않으므로 eval 중 새로 수렴한다(빠른 평가엔 근사적으로 충분).
        obs_norm=config.obs_norm,
        rank=0,
    )

    # Load model
    obs_shape = env.observation_space.shape
    action_space = env.action_space
    num_actions = len(TOOLS)
    in_w = obs_shape[1] if len(obs_shape) == 3 else 1
    in_h = obs_shape[2] if len(obs_shape) == 3 else 1

    actor_critic = Policy(
        obs_shape, action_space,
        base_kwargs={
            'map_width': config.map_x,
            'num_actions': num_actions,
            'recurrent': getattr(train_args, 'recurrent_policy', False),
            'prebuild': False,
            'in_w': in_w, 'in_h': in_h,
            'num_inputs': obs_shape[0],
            'out_w': config.map_x, 'out_h': config.map_y,
        },
        curiosity=False,
        algo=getattr(train_args, 'algo', 'ppo'),
        model=getattr(train_args, 'model', 'FullyConv'),
        args=train_args,
    )
    actor_critic.load_state_dict(checkpoint['model_state_dict'])
    actor_critic.to(device)
    actor_critic.eval()

    # Run evaluation
    tracker = EpisodeTracker(window_size=args.num_episodes)
    recurrent_hidden_state_size = actor_critic.recurrent_hidden_state_size
    all_tool_usage = {t: 0 for t in TOOLS}

    logger.info('Evaluating %d episodes (budget=%d, deterministic=%s)',
                args.num_episodes, config.init_budget, args.deterministic)

    for ep in range(args.num_episodes):
        obs = env.reset()
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)

        if isinstance(recurrent_hidden_state_size, tuple):
            rnn_hxs = torch.zeros(1, 2, *recurrent_hidden_state_size)
        else:
            rnn_hxs = torch.zeros(1, recurrent_hidden_state_size)

        masks = torch.ones(1, 1)
        done = False
        ep_reward = 0.0
        ep_actions = []

        while not done:
            with torch.no_grad():
                value, action, _, rnn_hxs = actor_critic.act(
                    obs_tensor, rnn_hxs, masks,
                    deterministic=args.deterministic,
                )

            action_int = action.item()
            ep_actions.append(action_int)

            obs, reward, done, info = env.step(action_int)
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            masks = torch.zeros(1, 1) if done else torch.ones(1, 1)
            ep_reward += reward

        # Record stats
        budget_mon = info.get('budget_monitor', {})
        if budget_mon:
            tracker.record(budget_mon)

        # Tool usage
        cells = config.map_x * config.map_y
        for a in ep_actions:
            tool_idx = a // cells
            if tool_idx < len(TOOLS):
                all_tool_usage[TOOLS[tool_idx]] += 1

        # Per-episode log
        ep_stats = info.get('budget_monitor', {
            'final_population': info.get('episode_stats', {}).get('total_builds', 0),
            'total_reward': ep_reward,
            'final_funds': info.get('funds', 0),
            'num_steps': len(ep_actions),
            'invalid_action_count': 0,
            'went_bankrupt': info.get('bankrupt', False),
        })
        logger.info(format_episode_log(ep, ep_stats))

    # Summary
    summary = tracker.summary()
    logger.info('\n=== Evaluation Summary (%d episodes) ===', args.num_episodes)
    logger.info('  Avg Population:      %.1f', summary['avg_population'])
    logger.info('  Avg Reward:          %.2f', summary['avg_reward'])
    logger.info('  Avg Final Funds:     %.0f', summary['avg_funds'])
    logger.info('  Bankruptcy Rate:     %.1f%%', summary['bankruptcy_rate'] * 100)
    logger.info('  Avg Invalid Actions: %.1f', summary['avg_invalid_actions'])
    logger.info('  Avg Episode Length:  %.0f', summary['avg_episode_length'])

    # Tool usage
    total_actions = sum(all_tool_usage.values())
    if total_actions > 0:
        logger.info('\n=== Tool Usage ===')
        sorted_tools = sorted(all_tool_usage.items(), key=lambda x: -x[1])
        for tool, count in sorted_tools:
            if count > 0:
                pct = count / total_actions * 100
                logger.info('  %-20s %5d (%5.1f%%)', tool, count, pct)

    env.close()
    return summary


def main():
    evaluate()


if __name__ == '__main__':
    main()
