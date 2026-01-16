import asyncio
import re
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
from pyrit.models import AttackOutcome, Message, ConversationType, Score
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
            
            # 각 턴의 상세 정보 추출 (Judge 점수 및 백트래킹 정보 포함)
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
                judge_score = None
                j = i + 1
                while j < len(conversation):
                    next_msg = conversation[j]
                    if next_msg.role == "assistant" or (hasattr(next_msg, 'message_pieces') and 
                                                        next_msg.message_pieces and
                                                        any(mp.role == "assistant" for mp in next_msg.message_pieces)):
                        try:
                            target_response = next_msg.get_value() if hasattr(next_msg, 'get_value') else str(next_msg)
                            # Judge 점수 추출 (assistant 메시지의 message_pieces에서 scores 찾기)
                            judge_score = self._extract_judge_score(next_msg)
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
                    "judge_score": judge_score,
                    "backtracked_pairs": [],
                    "backtrack_count": 0,
                }
                turn_history.append(turn_info)
            
            i += 1
        
        # 백트래킹된 프롬프트-응답 쌍 추출
        if result and memory and hasattr(result, 'related_conversations'):
            self._extract_backtracked_pairs(turn_history, result, memory)
        
        # 각 턴의 backtrack_count 업데이트
        for turn_info in turn_history:
            if "backtracked_pairs" in turn_info:
                turn_info["backtrack_count"] = len(turn_info["backtracked_pairs"])
        
        return turn_history
    
    def _extract_judge_score(self, assistant_msg: Message) -> Optional[float]:
        """Assistant 메시지에서 Judge 점수를 추출합니다.
        
        FloatScaleThresholdScorer의 score_rationale에서 원래 float scale 점수를 파싱합니다.
        예: "Normalized scale score: 0.75 < threshold 0.8" -> 0.75
        """
        try:
            if hasattr(assistant_msg, 'message_pieces') and assistant_msg.message_pieces:
                for piece in assistant_msg.message_pieces:
                    if hasattr(piece, 'scores') and piece.scores:
                        # FloatScaleThresholdScorer의 경우, score_rationale에서 원래 점수 파싱
                        for score in piece.scores:
                            scorer_type = score.scorer_class_identifier.get("__type__", "")
                            
                            if "FloatScaleThresholdScorer" in scorer_type:
                                # score_rationale에서 "Normalized scale score: 0.75 < threshold 0.8" 형식 파싱
                                score_value = self._parse_scale_score_from_rationale(score.score_rationale)
                                if score_value is not None:
                                    return score_value
                        
                        # FloatScaleThresholdScorer를 찾지 못한 경우, SelfAskScaleScorer의 점수 직접 찾기
                        for score in piece.scores:
                            scorer_type = score.scorer_class_identifier.get("__type__", "")
                            # SelfAskScaleScorer의 점수 직접 찾기 (float_scale 타입)
                            if "SelfAskScaleScorer" in scorer_type and score.score_type == "float_scale":
                                return float(score.get_value())
        except Exception:
            pass
        return None
    
    def _parse_scale_score_from_rationale(self, rationale: str) -> Optional[float]:
        """score_rationale에서 원래 float scale 점수를 파싱합니다.
        
        예: "Normalized scale score: 0.75 < threshold 0.8" -> 0.75
        """
        try:
            # "Normalized scale score: 0.75 < threshold 0.8" 패턴 매칭
            pattern = r"Normalized scale score:\s*([0-9]+\.?[0-9]*)\s*[<>=]"
            match = re.search(pattern, rationale)
            if match:
                return float(match.group(1))
        except Exception:
            pass
        return None
    
    def _extract_backtracked_pairs(self, turn_history: List[Dict[str, Any]], result: Any, memory: Any) -> None:
        """related_conversations에서 백트래킹된 프롬프트-응답 쌍을 추출하여 각 턴에 추가합니다."""
        # PRUNED 타입의 conversation들 (백트래킹된 conversation)
        pruned_conversations = [
            conv_ref for conv_ref in result.related_conversations
            if conv_ref.conversation_type == ConversationType.PRUNED
        ]
        
        # 각 PRUNED conversation에서 마지막 턴의 프롬프트-응답 쌍 추출
        for pruned_conv in pruned_conversations:
            try:
                pruned_messages = memory.get_conversation(conversation_id=pruned_conv.conversation_id)
                if pruned_messages:
                    # PRUNED conversation의 마지막 턴 추출
                    last_turn_pair = self._extract_last_turn_from_conversation(pruned_messages)
                    if last_turn_pair:
                        # 해당 턴 번호 계산
                        turn_num = self._count_turns_in_conversation(pruned_messages)
                        # 해당 턴의 backtracked_pairs에 추가
                        if turn_num > 0 and turn_num <= len(turn_history):
                            turn_info = turn_history[turn_num - 1]
                            if "backtracked_pairs" not in turn_info:
                                turn_info["backtracked_pairs"] = []
                            turn_info["backtracked_pairs"].append(last_turn_pair)
            except Exception:
                continue
    
    def _extract_last_turn_from_conversation(self, messages: MutableSequence[Message]) -> Optional[Dict[str, Any]]:
        """conversation에서 마지막 턴의 프롬프트-응답 쌍과 refusal_judge 여부를 추출합니다."""
        last_user_msg = None
        last_assistant_msg = None
        
        # 마지막 user와 assistant 메시지 찾기
        for msg in reversed(messages):
            if msg.role == "assistant" and last_assistant_msg is None:
                last_assistant_msg = msg
            elif msg.role == "user" and last_user_msg is None:
                last_user_msg = msg
                break  # user 메시지를 찾으면 중단 (역순이므로)
        
        if last_user_msg and last_assistant_msg:
            try:
                attacker_original = None
                attacker_converted = None
                if hasattr(last_user_msg, 'message_pieces') and last_user_msg.message_pieces:
                    attacker_original = last_user_msg.message_pieces[0].original_value if last_user_msg.message_pieces else None
                    attacker_converted = last_user_msg.get_value() if hasattr(last_user_msg, 'get_value') else str(last_user_msg)
                
                target_response = None
                if hasattr(last_assistant_msg, 'get_value'):
                    target_response = last_assistant_msg.get_value()
                else:
                    target_response = str(last_assistant_msg)
                
                # refusal_judge 여부 추출
                refusal_judge = self._extract_refusal_judge(last_assistant_msg)
                
                return {
                    "attacker_original": attacker_original,
                    "attacker_converted": attacker_converted,
                    "target_response": target_response,
                    "refusal_judge": refusal_judge,
                }
            except Exception:
                pass
        
        return None
    
    def _extract_refusal_judge(self, assistant_msg: Message) -> bool:
        """Assistant 메시지에서 refusal_judge 여부를 추출합니다."""
        try:
            if hasattr(assistant_msg, 'message_pieces') and assistant_msg.message_pieces:
                for piece in assistant_msg.message_pieces:
                    if hasattr(piece, 'scores') and piece.scores:
                        # SelfAskRefusalScorer의 점수 찾기
                        for score in piece.scores:
                            scorer_type = score.scorer_class_identifier.get("__type__", "")
                            if "SelfAskRefusalScorer" in scorer_type:
                                if score.score_type == "true_false":
                                    return bool(score.get_value())
        except Exception:
            pass
        return False
    
    def _count_turns_in_conversation(self, messages: MutableSequence[Message]) -> int:
        """conversation에서 턴의 개수를 계산합니다."""
        turn_count = 0
        for msg in messages:
            if msg.role == "user" or (hasattr(msg, 'message_pieces') and msg.message_pieces and 
                                      any(mp.role == "user" for mp in msg.message_pieces)):
                turn_count += 1
        return turn_count

    def print_summary(self):
        print_result_summary(self.results)
