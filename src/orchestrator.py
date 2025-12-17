import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional

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
        return self.__dict__


class CrescendoExperiment:
    def __init__(self, config: ExperimentConfig, targets: Dict[str, Any]):
        self.config = config
        self.targets = targets
        self.results: List[Dict[str, Any]] = []


    # 판단 모듈
    def _create_refusal_judge(self):
        return SelfAskRefusalScorer(chat_target=self.targets["refusal_judge"])
    
    def _create_primary_judge(self):
        return FloatScaleThresholdScorer(
            scorer=SelfAskScaleScorer(
                chat_target=self.targets["judge"],
                scale_arguments_path=SelfAskScaleScorer.ScalePaths.TASK_ACHIEVED_SCALE.value,
                system_prompt_path=SelfAskScaleScorer.SystemPaths.RED_TEAMER_SYSTEM_PROMPT.value,
            ),
            threshold=self.config.objective_threshold,
        )


    # 단일 테스크 실행
    async def run_single_task(self, objective: str, task_name: str) -> Dict[str, Any]:

        print("\n" + "=" * 70)
        print(f"[Task Started] {task_name}")
        print("Objective:")
        print(objective)
        print("=" * 70)

        # Judges
        primary_judge = self._create_primary_judge()
        refusal_judge = self._create_refusal_judge()

        # PyRIT scoring config
        scoring_config = AttackScoringConfig(
            objective_scorer=primary_judge,
            refusal_scorer=refusal_judge,
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
            # Crescendo Attack 실행
            result = await attack.execute_async(objective=objective)
            conversation: List[Message] = attack.get_memory()

            duration = (datetime.now() - start_time).total_seconds()
            num_turns = sum(1 for m in conversation if m.role == "user")

            # Final Turn Summary 추출
            last_attacker = conversation[-2] if len(conversation) >= 2 else None
            last_target = conversation[-1] if len(conversation) >= 1 else None

            final_turn_summary = None
            if last_attacker and last_target:
                final_turn_summary = {
                    "turn": num_turns,
                    "attacker": last_attacker.content,
                    "target": last_target.content,
                }

            final_score = None
            if hasattr(result, "last_score") and result.last_score:
                final_score = result.last_score.get_value()

            # 결과 저장 및 출력
            result_dict = {
                "task_name": task_name,
                "objective": objective,
                "success": result.achieved,
                "score": final_score,
                "num_turns": num_turns,
                "duration_seconds": duration,
                "final_turn": final_turn_summary,
                "config": self.config.to_dict(),
            }

            print("\n" + "=" * 70)
            print(f"[Task Completed] {task_name}")
            print("=" * 70)
            print(f"Success: {result.achieved}")
            print(f"Score: {final_score}")
            print(f"Turns Used: {num_turns}")
            print(f"Duration: {duration:.1f} seconds")

            if final_turn_summary:
                print("\nFinal Turn Summary:")
                print(f"  Turn {final_turn_summary['turn']}")
                print("  Attacker Prompt:")
                print(f"      {final_turn_summary['attacker'][:200]}")
                print("  Target Response:")
                print(f"      {final_turn_summary['target'][:200]}")

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


    # 다중 테스크 실행
    async def run_multiple_tasks(self, tasks: List[Dict[str, str]]):
        total = len(tasks)
        for idx, t in enumerate(tasks, 1):
            print_task_progress(idx, total, t["name"])
            await self.run_single_task(t["objective"], t["name"])


    # 유틸 함수
    def save_results(self, filename: Optional[str] = None) -> str:
        return save_results(self.results, filename)

    def print_summary(self):
        print_result_summary(self.results)
