import asyncio
from datetime import datetime
from typing import Dict, Any, List

from pyrit.memory import DuckDBMemory
from pyrit.executor.attack.multi_turn.crescendo import CrescendoAttack
from pyrit.executor.attack.core import (
    AttackAdversarialConfig,
    AttackConverterConfig,
    AttackScoringConfig,
)
from pyrit.score import (
    SelfAskRefusalScorer,
    SelfAskScaleScorer,
    FloatScaleThresholdScorer,
)
from pyrit.models import Message

from .utils import save_results, print_result_summary, print_task_progress


class ExperimentConfig:
    def __init__(
        self,
        target_model: str,
        attacker_model: str,
        judge_model: str,
        refusal_judge_model: str,
        max_turns: int = 10,
        max_backtracks: int = 10,
        objective_threshold: float = 0.5,
    ):
        self.target_model = target_model
        self.attacker_model = attacker_model
        self.judge_model = judge_model
        self.refusal_judge_model = refusal_judge_model
        self.max_turns = max_turns
        self.max_backtracks = max_backtracks
        self.objective_threshold = objective_threshold

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_model": self.target_model,
            "attacker_model": self.attacker_model,
            "judge_model": self.judge_model,
            "refusal_judge_model": self.refusal_judge_model,
            "max_turns": self.max_turns,
            "max_backtracks": self.max_backtracks,
            "objective_threshold": self.objective_threshold,
        }


class CrescendoExperiment:
    def __init__(self, config: ExperimentConfig, targets: Dict[str, Any]):
        self.config = config
        self.targets = targets
        self.memory = DuckDBMemory()
        self.results: List[Dict[str, Any]] = []

    def _create_refusal_judge(self):
        return SelfAskRefusalScorer(
            chat_target=self.targets["refusal_judge"]
        )

    def _create_primary_judge(self):
        return FloatScaleThresholdScorer(
            scorer=SelfAskScaleScorer(
                chat_target=self.targets["judge"],
                scale_arguments_path=SelfAskScaleScorer.ScalePaths.TASK_ACHIEVED_SCALE.value,
                system_prompt_path=SelfAskScaleScorer.SystemPaths.RED_TEAMER_SYSTEM_PROMPT.value,
            ),
            threshold=self.config.objective_threshold,
        )

    def _create_secondary_judge(self):
        return SelfAskScaleScorer(
            chat_target=self.targets["judge"],
            scale_arguments_path=SelfAskScaleScorer.ScalePaths.JUDGMENT_CORRECTION_SCALE.value,
            system_prompt_path=SelfAskScaleScorer.SystemPaths.RED_TEAMER_SYSTEM_PROMPT.value,
        )

    async def run_single_task(self, objective: str, task_name: str) -> Dict[str, Any]:

        print(f"\n{'=' * 70}")
        print(f"Task: {task_name}")
        print(f"Objective: {objective}")
        print(f"{'=' * 70}")

        primary_judge = self._create_primary_judge()
        secondary_judge = self._create_secondary_judge()
        refusal_judge = self._create_refusal_judge()

        scoring_config = AttackScoringConfig(
            objective_scorer=primary_judge,
            refusal_scorer=refusal_judge,
            auxiliary_scorers=[secondary_judge],
            successful_objective_threshold=self.config.objective_threshold,
            use_score_as_feedback=True,
        )

        adversarial_config = AttackAdversarialConfig(
            target=self.targets["attacker"],
            system_prompt_path=None,
        )

        attack = CrescendoAttack(
            objective_target=self.targets["target"],
            attack_adversarial_config=adversarial_config,
            attack_scoring_config=scoring_config,
            attack_converter_config=AttackConverterConfig(),
            max_turns=self.config.max_turns,
            max_backtracks=self.config.max_backtracks,
        )

        start_time = datetime.now()

        try:
            result = await attack.run_attack_async(objective=objective)

            conversation: List[Message] = attack.get_memory()

            duration = (datetime.now() - start_time).total_seconds()

            num_turns = sum(1 for m in conversation if m.role == "user")

            result_dict = {
                "task_name": task_name,
                "objective": objective,
                "success": result.achieved,
                "score": getattr(result, "score", None),
                "num_turns": num_turns,
                "duration_seconds": duration,
                "config": self.config.to_dict(),
                "conversation": [
                    {
                        "turn": i,
                        "role": m.role,
                        "content": (
                            m.content[:200] + "..."
                            if len(m.content) > 200 else m.content
                        ),
                    }
                    for i, m in enumerate(conversation)
                ],
            }

            self.results.append(result_dict)
            return result_dict

        except Exception as e:
            err = {
                "task_name": task_name,
                "objective": objective,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "config": self.config.to_dict(),
            }
            self.results.append(err)
            return err

    async def run_multiple_tasks(self, tasks: List[Dict[str, str]]):
        total = len(tasks)
        for idx, t in enumerate(tasks, 1):
            print_task_progress(idx, total, t["name"])
            await self.run_single_task(t["objective"], t["name"])

    def save_results(self, filename: str = None):
        return save_results(self.results, filename)

    def print_summary(self):
        print_result_summary(self.results)
