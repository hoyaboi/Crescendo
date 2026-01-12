import asyncio
from datetime import datetime
from types import NoneType
from typing import Dict, Any, List, Optional, MutableSequence

from pyrit.executor.attack.multi_turn.crescendo import CrescendoAttack
from pyrit.executor.attack.core import (
    AttackAdversarialConfig,
    AttackConverterConfig,
    AttackScoringConfig,
)
from pyrit.prompt_converter import EmojiConverter
from pyrit.prompt_normalizer import PromptConverterConfiguration
from pyrit.score import (
    SelfAskRefusalScorer,
    SelfAskScaleScorer,
    FloatScaleThresholdScorer,
)
from pyrit.models import AttackOutcome, Message
from pyrit.memory.central_memory import CentralMemory

from .utils import save_results, save_turn_logs, append_result_to_file, print_result_summary, print_task_progress


class ExperimentConfig:
    def __init__(
        self,
        target_model: str,
        attacker_model: str,
        judge_model: str,
        refusal_judge_model: str,
        max_turns: int = 10,
        max_backtracks: int = 10,
        objective_threshold: float = 0.8,
        use_converter: bool = True,
    ):
        self.target_model = target_model
        self.attacker_model = attacker_model
        self.judge_model = judge_model
        self.refusal_judge_model = refusal_judge_model
        self.max_turns = max_turns
        self.max_backtracks = max_backtracks
        self.objective_threshold = objective_threshold
        self.use_converter = use_converter

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__


class CrescendoExperiment:
    def __init__(self, config: ExperimentConfig, targets: Dict[str, Any], results_filepath: Optional[str] = None):
        self.config = config
        self.targets = targets
        self.results: List[Dict[str, Any]] = []
        self.results_filepath = results_filepath  # 결과 파일 경로 저장


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


    # 공격 실행 메서드
    async def run_attack_async(
        self, 
        objective: str, 
        memory_labels: Optional[Dict[str, str]] = None
    ) -> Any:
        primary_judge = self._create_primary_judge()
        refusal_judge = self._create_refusal_judge()

        scoring_config = AttackScoringConfig(
            objective_scorer=primary_judge,
            refusal_scorer=refusal_judge,
            successful_objective_threshold=self.config.objective_threshold,
            use_score_as_feedback=True,
        )

        adversarial_config = AttackAdversarialConfig(target=self.targets["attacker"])

        if self.config.use_converter:
            converters = PromptConverterConfiguration.from_converters(converters=[EmojiConverter()])
            converter_config = AttackConverterConfig(request_converters=converters)
        else:
            converter_config = AttackConverterConfig()

        attack = CrescendoAttack(
            objective_target=self.targets["target"],
            attack_adversarial_config=adversarial_config,
            attack_scoring_config=scoring_config,
            attack_converter_config=converter_config,
            max_turns=self.config.max_turns,
            max_backtracks=self.config.max_backtracks,
        )

        result = await attack.execute_async(
            objective=objective,
            memory_labels=memory_labels
        )
        return result

    # 단일 테스크 실행
    async def run_single_task(
        self, 
        objective: str, 
        task_name: str,
        memory_labels: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        print("\n" + "=" * 70)
        print(f"[Task Started] {task_name}")
        print("Objective:")
        print(objective)
        print("=" * 70)

        start_time = datetime.now()

        try:
            result = await self.run_attack_async(objective=objective, memory_labels=memory_labels)
            
            # 대화 내용 가져오기
            memory = CentralMemory.get_memory_instance()
            conversation: MutableSequence[Message] = memory.get_conversation(
                conversation_id=result.conversation_id
            )

            duration = (datetime.now() - start_time).total_seconds()
            
            # 각 턴의 상세 정보 추출
            turn_history = self._extract_turn_history(conversation)
            num_turns = len(turn_history)

            # Final Turn Summary 추출
            final_turn_summary = None
            if turn_history:
                last_turn = turn_history[-1]
                final_turn_summary = {
                    "turn": last_turn["turn"],
                    "attacker_original": last_turn["attacker_original"],
                    "attacker_converted": last_turn["attacker_converted"],
                    "target": last_turn["target_response"],
                }

            success = result.outcome == AttackOutcome.SUCCESS

            # 결과 저장 및 출력
            result_dict = {
                "task_name": task_name,
                "objective": objective,
                "success": success,
                "num_turns": num_turns,
                "duration_seconds": duration,
                "final_turn": final_turn_summary,
                "turn_history": turn_history, 
                "config": self.config.to_dict(),
            }
            
            log_filepath = save_turn_logs(task_name, turn_history)

            print("\n" + "=" * 70)
            print(f"[Task Completed] {task_name}")
            print("=" * 70)
            print(f"Success: {success}")
            print(f"Turns Used: {num_turns}")
            print(f"Duration: {duration:.1f} seconds")

            if final_turn_summary:
                print("\nFinal Turn Summary:")
                print(f"  Turn {final_turn_summary['turn']}")
                if final_turn_summary.get('attacker_original'):
                    print("  Original Prompt:")
                    print(f"      {final_turn_summary['attacker_original'][:200]}")
                print("  Attacker Prompt (Converted):")
                print(f"      {final_turn_summary['attacker_converted'][:200]}")
                print("  Target Response:")
                print(f"      {final_turn_summary['target'][:200]}")

            self.results.append(result_dict)
            
            if self.results_filepath:
                append_result_to_file(result_dict, self.results_filepath)

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
            
            if self.results_filepath:
                append_result_to_file(err, self.results_filepath)
            
            return err


    # 다중 테스크 실행
    async def run_multiple_tasks(self, tasks: List[Dict[str, str]]):
        total = len(tasks)
        for idx, t in enumerate(tasks, 1):
            print_task_progress(idx, total, t["name"])
            await self.run_single_task(t["objective"], t["name"])


    def save_results(self, filename: Optional[str] = None) -> str:
        return save_results(self.results, filename)

    def _extract_turn_history(self, conversation: MutableSequence[Message]) -> List[Dict[str, Any]]:
        turn_history = []
        turn_number = 0
        
        i = 0
        while i < len(conversation):
            msg = conversation[i]
            
            # User 메시지 (attacker prompt) 찾기
            if msg.role == "user" or (hasattr(msg, 'message_pieces') and msg.message_pieces and 
                                      any(mp.role == "user" for mp in msg.message_pieces)):
                turn_number += 1
                
                # Attacker prompt 추출
                try:
                    if hasattr(msg, 'message_pieces') and msg.message_pieces:
                        attacker_original = msg.message_pieces[0].original_value if msg.message_pieces else None
                        attacker_converted = msg.get_value() if hasattr(msg, 'get_value') else str(msg)
                    else:
                        attacker_original = None
                        attacker_converted = str(msg)
                except Exception:
                    attacker_original = None
                    attacker_converted = str(msg)
                
                # 다음 Assistant 메시지 찾기 (target response)
                target_response = None
                j = i + 1
                while j < len(conversation):
                    next_msg = conversation[j]
                    if next_msg.role == "assistant" or (hasattr(next_msg, 'message_pieces') and 
                                                        next_msg.message_pieces and
                                                        any(mp.role == "assistant" for mp in next_msg.message_pieces)):
                        try:
                            target_response = next_msg.get_value() if hasattr(next_msg, 'get_value') else str(next_msg)
                        except Exception:
                            target_response = str(next_msg)
                        break
                    j += 1
                
                # 턴 정보 저장
                turn_info = {
                    "turn": turn_number,
                    "attacker_original": attacker_original,
                    "attacker_converted": attacker_converted,
                    "target_response": target_response,
                }
                turn_history.append(turn_info)
            
            i += 1
        
        return turn_history

    def print_summary(self):
        print_result_summary(self.results)
