from typing import Dict, List
from .model_configs import AVAILABLE_MODELS, ModelConfig
import os


class ModelFactory:
    @staticmethod
    def get_model_config(model_key: str) -> ModelConfig:
        # ModelConfig 불러오기
        if model_key not in AVAILABLE_MODELS:
            available = list(AVAILABLE_MODELS.keys())
            raise ValueError(
                f"Unknown model: {model_key}\n"
                f"Available models: {', '.join(available)}\n"
                f"Use ModelFactory.list_available_models() to see details."
            )
        return AVAILABLE_MODELS[model_key]
    

    @staticmethod
    def create_targets(target_model: str, attack_model: str, judge_model: str, refusal_judge_model: str) -> Dict:
        # 모델 객체 생성
        print(f"\nCreating model targets...")
        print(f"   Target:         {target_model}")
        print(f"   Attacker:       {attack_model}")
        print(f"   Judge:          {judge_model}")
        print(f"   Refusal Judge:  {refusal_judge_model}")
        
        try:
            targets = {
                "target": ModelFactory.get_model_config(target_model).to_target(),
                "attacker": ModelFactory.get_model_config(attack_model).to_target(),
                "judge": ModelFactory.get_model_config(judge_model).to_target(),
                "refusal_judge": ModelFactory.get_model_config(refusal_judge_model).to_target(),
            }
            print(f"All targets created successfully\n")
            return targets
        except Exception as e:
            print(f"Error creating targets: {str(e)}\n")
            raise
    

    @staticmethod
    def list_available_models():
        # 사용 가능한 모델 목록 출력
        print("\n" + "="*30 + " AVAILABLE MODELS " + "="*30)
        
        by_type = {}
        for key, config in AVAILABLE_MODELS.items():
            by_type.setdefault(config.model_type, []).append((key, config))
        
        for model_type in sorted(by_type.keys()):
            print(f"\n{model_type.upper()} Models:")
            for key, config in by_type[model_type]:
                print(f"  {key:20s} -> {config.name}")
        
        print("\n" + "="*78 + "\n")
    

    @staticmethod
    def check_api_keys():
        # 설정된 API 키 확인
        print("\n" + "="*31 + " API KEY STATUS " + "="*31)
        
        required_keys = set()
        for config in AVAILABLE_MODELS.values():
            required_keys.add(config.api_key_env)
        
        for key_name in sorted(required_keys):
            key_value = os.getenv(key_name)
            if key_value:
                print(f"\n{key_name:20s} SET")
            else:
                print(f"\n{key_name:20s} NOT SET")
        
        print("\n" + "="*78 + "\n")
    
    
    @staticmethod
    def get_models_by_type(model_type: str) -> List[str]:
        return [
            key for key, config in AVAILABLE_MODELS.items()
            if config.model_type == model_type
        ]
