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
    async def create_targets(target_model: str, attack_model: str, judge_model: str, refusal_judge_model: str) -> Dict:
        # 모델 객체 생성
        print(f"\nCreating model targets...")
        print(f"   Target:         {target_model}")
        print(f"   Attacker:       {attack_model}")
        print(f"   Judge:          {judge_model}")
        print(f"   Refusal Judge:  {refusal_judge_model}")
        
        try:
            import asyncio
            targets = {}
            for name, model_key in [("target", target_model), ("attacker", attack_model), 
                                   ("judge", judge_model), ("refusal_judge", refusal_judge_model)]:
                try:
                    print(f"   Creating {name} ({model_key})...")
                    target = ModelFactory.get_model_config(model_key).to_target()
                    targets[name] = target
                    
                    # HuggingFace 모델인 경우 모델 로딩 완료 대기
                    if hasattr(target, "load_model_and_tokenizer_task"):
                        print(f"   Waiting for {name} model to load (this may take several minutes)...")
                        try:
                            # 타임아웃 설정 (30분)
                            await asyncio.wait_for(target.load_model_and_tokenizer_task, timeout=1800)
                            print(f"   ✓ {name} model loaded successfully")
                            
                            # GPU 사용 확인
                            if hasattr(target, "model") and hasattr(target, "device"):
                                if target.device == "cuda":
                                    import torch
                                    if torch.cuda.is_available():
                                        memory_gb = torch.cuda.memory_allocated(0) / 1024**3
                                        print(f"   ✓ {name} model is on GPU (memory: {memory_gb:.2f} GB)")
                                else:
                                    print(f"   ⚠ Warning: {name} model is on {target.device}, not GPU")
                        except asyncio.TimeoutError:
                            print(f"   ✗ Error: {name} model loading timed out after 30 minutes")
                            raise
                        except Exception as e:
                            print(f"   ✗ Error loading {name} model: {e}")
                            import traceback
                            traceback.print_exc()
                            raise
                    else:
                        print(f"   ✓ {name} created")
                except Exception as e:
                    print(f"   ✗ Error creating {name} ({model_key}): {e}")
                    import traceback
                    traceback.print_exc()
                    raise
            
            print(f"All targets created successfully\n")
            return targets
        except Exception as e:
            print(f"Error creating targets: {str(e)}\n")
            import traceback
            traceback.print_exc()
            raise
    

    @staticmethod
    def list_available_models():
        # 사용 가능한 모델 목록 출력
        print("\n" + "="*70)
        print("AVAILABLE MODELS")
        print("="*70)
        
        by_type = {}
        for key, config in AVAILABLE_MODELS.items():
            by_type.setdefault(config.model_type, []).append((key, config))
        
        for model_type in sorted(by_type.keys()):
            print(f"\n{model_type.upper()} Models:")
            for key, config in by_type[model_type]:
                print(f"  {key:20s} -> {config.name}")
        
        print("\n" + "="*70 + "\n")
    

    @staticmethod
    def check_api_keys():
        # 설정된 API 키 확인
        print("\n" + "="*70)
        print("API KEY STATUS")
        print("="*70)
        
        required_keys = set()
        for config in AVAILABLE_MODELS.values():
            required_keys.add(config.api_key_env)
        
        for key_name in sorted(required_keys):
            key_value = os.getenv(key_name)
            if key_value:
                print(f"\n{key_name:20s} SET")
            else:
                print(f"\n{key_name:20s} NOT SET")
        
        print("\n" + "="*70 + "\n")
    

    @staticmethod
    def get_models_by_type(model_type: str) -> List[str]:
        return [
            key for key, config in AVAILABLE_MODELS.items()
            if config.model_type == model_type
        ]
