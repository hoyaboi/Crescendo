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
from pyrit.models import AttackOutcome, Message, ConversationType
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
            
            # 각 턴의 상세 정보 추출 (백트래킹 정보 포함)
            turn_history = self._extract_turn_history(conversation, result, memory)
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

    def _extract_turn_history(self, conversation: MutableSequence[Message], result: Any = None, memory: Any = None) -> List[Dict[str, Any]]:
        # 먼저 현재 conversation에서 턴 정보 추출
        turn_history = []
        turn_number = 0
        
        i = 0
        while i < len(conversation):
            msg = conversation[i]
            
            # User 메시지 (attacker prompt) 찾기
            if msg.role == "user" or (hasattr(msg, 'message_pieces') and msg.message_pieces and 
                                      any(mp.role == "user" for mp in msg.message_pieces)):
                # 이전에 assistant 메시지가 있었는지 확인하여 새로운 턴인지 판단
                is_new_turn = True
                if i > 0:
                    # 이전 메시지들을 역순으로 확인
                    for k in range(i - 1, max(-1, i - 20), -1):
                        prev_msg = conversation[k]
                        if prev_msg.role == "assistant" or (hasattr(prev_msg, 'message_pieces') and 
                                                             prev_msg.message_pieces and
                                                             any(mp.role == "assistant" for mp in prev_msg.message_pieces)):
                            is_new_turn = True
                            break
                        elif prev_msg.role == "user" or (hasattr(prev_msg, 'message_pieces') and prev_msg.message_pieces and
                                                          any(mp.role == "user" for mp in prev_msg.message_pieces)):
                            is_new_turn = False
                            break
                
                if is_new_turn:
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
                
                # 턴 정보 저장 (같은 턴의 마지막 메시지만 저장)
                if is_new_turn:
                    turn_info = {
                        "turn": turn_number,
                        "backtrack_count": 0,
                        "attacker_original": attacker_original,
                        "attacker_converted": attacker_converted,
                        "target_response": target_response,
                    }
                    turn_history.append(turn_info)
                else:
                    # 같은 턴의 업데이트된 정보로 교체
                    if turn_history and turn_history[-1]["turn"] == turn_number:
                        turn_history[-1] = {
                            "turn": turn_number,
                            "backtrack_count": 0,
                            "attacker_original": attacker_original,
                            "attacker_converted": attacker_converted,
                            "target_response": target_response,
                        }
            
            i += 1
        
        # related_conversations에서 백트래킹된 conversation들을 분석하여 각 턴의 백트래킹 횟수 계산
        if result and memory and hasattr(result, 'related_conversations'):
            # PRUNED 타입의 conversation들 (백트래킹된 conversation)
            pruned_conversations = [
                conv_ref for conv_ref in result.related_conversations
                if conv_ref.conversation_type == ConversationType.PRUNED
            ]
            
            # 각 PRUNED conversation에서 마지막 턴을 확인하여 어떤 턴에서 백트래킹이 발생했는지 추적
            turn_backtrack_counts = {}
            for pruned_conv in pruned_conversations:
                try:
                    pruned_messages = memory.get_conversation(conversation_id=pruned_conv.conversation_id)
                    if pruned_messages:
                        # PRUNED conversation의 마지막 턴 번호 계산
                        pruned_turn_count = self._count_turns_in_conversation(pruned_messages)
                        # 해당 턴에서 백트래킹 발생
                        if pruned_turn_count > 0:
                            turn_backtrack_counts[pruned_turn_count] = turn_backtrack_counts.get(pruned_turn_count, 0) + 1
                except Exception:
                    # conversation을 가져올 수 없는 경우 무시
                    continue
            
            # 각 턴의 백트래킹 횟수 업데이트
            for turn_info in turn_history:
                turn_num = turn_info["turn"]
                turn_info["backtrack_count"] = turn_backtrack_counts.get(turn_num, 0)
        
        return turn_history
    
    def _count_turns_in_conversation(self, messages: MutableSequence[Message]) -> int:
        """conversation에서 턴의 개수를 계산합니다."""
        turn_count = 0
        i = 0
        while i < len(messages):
            msg = messages[i]
            if msg.role == "user" or (hasattr(msg, 'message_pieces') and msg.message_pieces and 
                                      any(mp.role == "user" for mp in msg.message_pieces)):
                # 이전에 assistant 메시지가 있었는지 확인
                is_new_turn = True
                if i > 0:
                    for k in range(i - 1, max(-1, i - 20), -1):
                        prev_msg = messages[k]
                        if prev_msg.role == "assistant" or (hasattr(prev_msg, 'message_pieces') and 
                                                             prev_msg.message_pieces and
                                                             any(mp.role == "assistant" for mp in prev_msg.message_pieces)):
                            is_new_turn = True
                            break
                        elif prev_msg.role == "user" or (hasattr(prev_msg, 'message_pieces') and prev_msg.message_pieces and
                                                          any(mp.role == "user" for mp in prev_msg.message_pieces)):
                            is_new_turn = False
                            break
                if is_new_turn:
                    turn_count += 1
            i += 1
        return turn_count

    def print_summary(self):
        print_result_summary(self.results)
