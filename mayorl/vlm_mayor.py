"""mayorl/vlm_mayor.py -- Qwen2.5-VL 기반 시장 에이전트.

모델: Qwen/Qwen2.5-VL-7B-Instruct (4-bit 양자화)
- text-only 기본 (vision은 --vlm-with-screenshot 옵션 시 활성화)
- 모델은 처음 호출 시 한 번만 로드 (lazy init)

필수 패키지:
    pip install transformers accelerate bitsandbytes qwen-vl-utils
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

# ---------------------------------------------------------------------------
# 전략 목표
# ---------------------------------------------------------------------------

STRATEGIC_GOALS = [
    "expand",       # 새 구역 건설 확장
    "consolidate",  # 기존 구역 강화, 확장 자제
    "raise_tax",    # 세율 올려 세수 확보
    "lower_tax",    # 세율 낮춰 인구 유입
    "power_first",  # 전력망 연결 우선
    "road_first",   # 도로 접근성 우선
]
NUM_GOALS = len(STRATEGIC_GOALS)

# ---------------------------------------------------------------------------
# 프롬프트
# ---------------------------------------------------------------------------

MAYOR_BRIEFING_PROMPT = """\
당신은 Micropolis 도시의 시장 AI입니다. 도시 운영 현황을 보고 평가해주세요.

## 현재 도시 현황
- 인구: 주거 {res_pop}, 상업 {com_pop}, 산업 {ind_pop} (합계 {total_pop})
- 잔액: ${funds:,} (초기 예산 ${init_budget:,})
- 세율: {tax_rate}% (범위 0~20)
- 시장 지지율: {mayor_rating}/100
- 수요: 주거 {res_demand:.1f}, 상업 {com_demand:.1f}, 산업 {ind_demand:.1f}
- 전력 연결률: {power_coverage:.1%}
- 도로 커버리지: {road_coverage:.1%}
- 이번 에피소드 세수 합계: ${tax_revenue:,}
- 건설 실패 횟수 (자금 부족): {failed_builds}

## 응답 형식 (JSON만 출력, 다른 텍스트 없음)
{{
  "score": <0~10 정수>,
  "financial_health": <0~10 정수>,
  "infrastructure": <0~10 정수>,
  "demand_balance": <0~10 정수>,
  "issues": ["문제점1", "문제점2"],
  "strategic_goal": "<expand|consolidate|raise_tax|lower_tax|power_first|road_first>",
  "tax_recommendation": <0~20 정수>,
  "briefing": "<한 문장 시장 브리핑>"
}}"""

# ---------------------------------------------------------------------------
# 데이터 클래스
# ---------------------------------------------------------------------------

@dataclass
class CityStats:
    res_pop: int = 0
    com_pop: int = 0
    ind_pop: int = 0
    funds: int = 0
    init_budget: int = 0
    tax_rate: int = 7
    mayor_rating: int = 0
    res_demand: float = 0.0
    com_demand: float = 0.0
    ind_demand: float = 0.0
    power_coverage: float = 0.0
    road_coverage: float = 0.0
    tax_revenue: int = 0
    failed_builds: int = 0

    @property
    def total_pop(self) -> int:
        return self.res_pop + self.com_pop + self.ind_pop


@dataclass
class MayorEvaluation:
    score: float = 0.0
    financial_health: float = 0.0
    infrastructure: float = 0.0
    demand_balance: float = 0.0
    issues: List[str] = field(default_factory=list)
    strategic_goal: str = "expand"
    tax_recommendation: int = 7
    briefing: str = ""
    raw_response: str = ""

    @property
    def normalized_score(self) -> float:
        return self.score / 10.0

    @property
    def goal_vector(self) -> np.ndarray:
        vec = np.zeros(NUM_GOALS, dtype=np.float32)
        if self.strategic_goal in STRATEGIC_GOALS:
            vec[STRATEGIC_GOALS.index(self.strategic_goal)] = 1.0
        else:
            vec[0] = 1.0
        return vec


# ---------------------------------------------------------------------------
# 모델 싱글톤 (프로세스 당 한 번만 로드)
# ---------------------------------------------------------------------------

_model = None
_processor = None


def _load_model():
    global _model, _processor
    if _model is not None:
        return _model, _processor

    logger.info("Qwen2.5-VL-7B-Instruct 4-bit 로드 중... (처음 한 번만)")
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    _model.eval()

    _processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    logger.info("Qwen2.5-VL 로드 완료")
    return _model, _processor


# ---------------------------------------------------------------------------
# VLMMayor
# ---------------------------------------------------------------------------

class VLMMayor:
    """Qwen2.5-VL-7B 기반 시장 에이전트.

    Parameters
    ----------
    eval_every : int
        몇 update마다 호출할지.
    image_dir : str or None
        스크린샷 저장 경로. None이면 임시 디렉터리.
    text_only : bool
        True면 이미지 없이 텍스트만 (기본값).
    max_new_tokens : int
        생성 최대 토큰 수.
    """

    def __init__(
        self,
        eval_every: int = 50,
        image_dir: Optional[str] = None,
        text_only: bool = True,
        max_new_tokens: int = 512,
    ):
        self.eval_every = eval_every
        self.image_dir = image_dir or tempfile.mkdtemp(prefix="mayorl_screenshots_")
        self.text_only = text_only
        self.max_new_tokens = max_new_tokens

        self._last_eval: Optional[MayorEvaluation] = None
        self._eval_count: int = 0

    # ------------------------------------------------------------------
    # 도시 수치 수집
    # ------------------------------------------------------------------

    def collect_stats(self, env) -> CityStats:
        try:
            demands = env.micro.engine.getDemands()
        except Exception:
            demands = (0.0, 0.0, 0.0)

        ep_stats = getattr(env, '_episode_stats', {})
        return CityStats(
            res_pop=int(env.micro.getResPop()),
            com_pop=int(env.micro.getComPop()),
            ind_pop=int(env.micro.getIndPop()),
            funds=int(env.micro.getFunds()),
            init_budget=int(getattr(env.config, 'init_budget', 0)),
            tax_rate=int(env.micro.getTaxRate()),
            mayor_rating=int(getattr(env.micro.engine, 'cityYes', 0)),
            res_demand=float(demands[0]),
            com_demand=float(demands[1]),
            ind_demand=float(demands[2]),
            power_coverage=float(getattr(env, '_last_power_coverage', 0.0)),
            road_coverage=float(getattr(env, '_last_road_coverage', 0.0)),
            tax_revenue=int(ep_stats.get('tax_revenue', 0)),
            failed_builds=int(ep_stats.get('failed_builds', 0)),
        )

    # ------------------------------------------------------------------
    # 스크린샷
    # ------------------------------------------------------------------

    def _render_screenshot(self, env, step: int) -> Optional[str]:
        if self.text_only:
            return None
        try:
            path = os.path.join(self.image_dir, f"city_step_{step:07d}.png")
            env.micro.render_to_file(path)
            return path
        except Exception as e:
            logger.warning("스크린샷 저장 실패: %s — text-only로 진행", e)
            return None

    # ------------------------------------------------------------------
    # 추론
    # ------------------------------------------------------------------

    def _build_messages(self, stats: CityStats, image_path: Optional[str]) -> list:
        prompt = MAYOR_BRIEFING_PROMPT.format(
            res_pop=stats.res_pop,
            com_pop=stats.com_pop,
            ind_pop=stats.ind_pop,
            total_pop=stats.total_pop,
            funds=stats.funds,
            init_budget=stats.init_budget,
            tax_rate=stats.tax_rate,
            mayor_rating=stats.mayor_rating,
            res_demand=stats.res_demand,
            com_demand=stats.com_demand,
            ind_demand=stats.ind_demand,
            power_coverage=stats.power_coverage,
            road_coverage=stats.road_coverage,
            tax_revenue=stats.tax_revenue,
            failed_builds=stats.failed_builds,
        )

        content = []
        if image_path and os.path.exists(image_path):
            content.append({"type": "image", "image": f"file://{image_path}"})
        content.append({"type": "text", "text": prompt})

        return [{"role": "user", "content": content}]

    def _call_model(self, stats: CityStats, image_path: Optional[str]) -> MayorEvaluation:
        import torch

        model, processor = _load_model()
        messages = self._build_messages(stats, image_path)

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        if image_path and os.path.exists(image_path):
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt",
            ).to(model.device)
        else:
            inputs = processor(text=[text], return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        # 입력 토큰 제외하고 생성 부분만 디코딩
        generated = output_ids[0][inputs.input_ids.shape[1]:]
        raw = processor.decode(generated, skip_special_tokens=True)

        self._eval_count += 1
        return self._parse_response(raw)

    def _parse_response(self, raw: str) -> MayorEvaluation:
        result = MayorEvaluation(raw_response=raw)
        try:
            match = re.search(r"```json\s*([\s\S]*?)\s*```", raw)
            json_str = match.group(1) if match else raw
            brace = re.search(r"\{[\s\S]*\}", json_str)
            if brace:
                json_str = brace.group(0)
            data = json.loads(json_str)

            result.score = float(data.get("score", 0))
            result.financial_health = float(data.get("financial_health", 0))
            result.infrastructure = float(data.get("infrastructure", 0))
            result.demand_balance = float(data.get("demand_balance", 0))
            result.issues = data.get("issues", [])
            result.strategic_goal = data.get("strategic_goal", "expand")
            result.tax_recommendation = int(data.get("tax_recommendation", 7))
            result.briefing = data.get("briefing", "")
        except Exception as e:
            logger.warning("응답 파싱 실패: %s\n원문: %s", e, raw[:300])
        return result

    # ------------------------------------------------------------------
    # 외부 인터페이스
    # ------------------------------------------------------------------

    def should_evaluate(self, update_i: int) -> bool:
        return update_i > 0 and update_i % self.eval_every == 0

    def evaluate(self, env, step: int) -> MayorEvaluation:
        stats = self.collect_stats(env)
        image_path = self._render_screenshot(env, step)
        evaluation = self._call_model(stats, image_path)
        self._last_eval = evaluation

        logger.info(
            "시장 평가 (step=%d): score=%.1f goal=%s tax_rec=%d | %s",
            step, evaluation.score, evaluation.strategic_goal,
            evaluation.tax_recommendation, evaluation.briefing,
        )
        return evaluation

    def apply_tax_recommendation(self, env, evaluation: MayorEvaluation) -> None:
        env.micro.setTaxRate(evaluation.tax_recommendation)

    @property
    def last_evaluation(self) -> Optional[MayorEvaluation]:
        return self._last_eval

    @property
    def total_eval_count(self) -> int:
        return self._eval_count
