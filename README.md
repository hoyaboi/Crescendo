# Crescendo Attack Evaluation

Crescendo 공격 평가를 위한 실험 프레임워크입니다. PyRIT의 CrescendoAttack을 사용하여 멀티턴 공격을 실행하고 결과를 분석합니다.

## 개요

Crescendo 공격은 점진적으로 모델을 유도하여 유해한 콘텐츠를 생성하도록 하는 전략입니다. 이 프로젝트는 Crescendo 공격을 체계적으로 평가하고 결과를 저장하는 도구를 제공합니다.

## 주요 기능

- Crescendo 공격 실행 및 평가
- OpenAI, HuggingFace, Anthropic Claude 모델 지원
- 원격 서버 모델 지원 (vLLM 등)
- 다중 테스크 일괄 처리
- 각 턴의 상세 로깅 (원본/변환 프롬프트, 응답)
- 결과 자동 저장 및 요약 (각 테스크마다 incremental save)
- .env 파일을 통한 API 키 관리

## 설치

### 1. 저장소 클론

```bash
git clone <repository-url>
cd crescendo
```

### 2. 의존성 설치

```bash
pip install -r requirements.txt
```

### 3. 환경 변수 설정

`.env` 파일을 생성하고 API 키를 설정합니다:

```bash
# OpenAI API (OpenAI 모델 사용 시)
OPENAI_API_KEY=your-openai-api-key

# HuggingFace (HuggingFace 모델 사용 시)
HUGGINGFACE_TOKEN=your-huggingface-token

# Anthropic API (Claude 모델 사용 시)
ANTHROPIC_API_KEY=your-anthropic-api-key

# Remote Server (원격 서버 모델 사용 시, 선택사항)
REMOTE_API_KEY=your-remote-api-key-if-needed
```

## 사용법

### 기본 실행

```bash
python crescendo_eval.py
```

### 모델 선택

```bash
# 모든 모델을 gpt-4o로 설정
python crescendo_eval.py \
  --target-model gpt-4o \
  --attacker-model gpt-4o \
  --judge-model gpt-4o \
  --refusal-judge-model gpt-4o
```

### 실험 파라미터 조정

```bash
python crescendo_eval.py \
  --max-turns 5 \
  --max-backtracks 3 \
  --objective-threshold 0.8
```

### 커스텀 테스크 파일 사용

```bash
python crescendo_eval.py --tasks-file data/your_tasks.json
```

### 유틸리티 명령어

```bash
# 사용 가능한 모델 목록 확인
python crescendo_eval.py --list-models

# API 키 상태 확인
python crescendo_eval.py --check-keys
```

## 명령줄 옵션

### 모델 설정

- `--target-model`: 공격 대상 모델 (기본값: gpt-4o-mini)
- `--attacker-model`: 공격 프롬프트 생성 모델 (기본값: gpt-4o-mini)
- `--judge-model`: 목표 달성 판단 모델 (기본값: gpt-4o-mini)
- `--refusal-judge-model`: 거부 감지 모델 (기본값: gpt-4o-mini)

### 실험 파라미터

- `--max-turns`: 최대 턴 수 (기본값: 10)
- `--max-backtracks`: 최대 백트래킹 횟수 (기본값: 10)
- `--objective-threshold`: 목표 달성 임계값 (기본값: 0.8)

### 파일 설정

- `--tasks-file`: 테스크 JSON 파일 경로 (기본값: data/custom_tasks.json)
- `--output-file`: 결과 파일명 (기본값: 자동 생성)

## 테스크 파일 형식

테스크는 JSON 파일로 정의합니다:

```json
[
  {
    "name": "Task Name",
    "objective": "Attack objective description"
  }
]
```

## 지원 모델

### OpenAI 모델

- `gpt-4`: GPT-4
- `gpt-4o`: GPT-4o
- `gpt-4o-mini`: GPT-4o-mini
- `gpt-3.5-turbo`: GPT-3.5 Turbo

### HuggingFace 모델 (로컬 실행)

- `llama-2-7b`: LLaMA-2-7B
- `llama-2-70b`: LLaMA-2-70B
- `llama-3-8b`: LLaMA-3-8B
- `llama-3-70b`: LLaMA-3-70B

### 원격 서버 모델 (vLLM 등)

- `llama-3-8b-remote`: LLaMA-3-8B (원격 서버, OpenAI 호환 API)

### Anthropic Claude 모델

- `claude-3-5-sonnet`: Claude 3.5 Sonnet
- `claude-3-opus`: Claude 3 Opus
- `claude-3-haiku`: Claude 3 Haiku
- `claude-sonnet-4`: Claude Sonnet 4

사용 가능한 모든 모델은 `--list-models` 옵션으로 확인할 수 있습니다.

## 결과

### 결과 파일

실험 결과는 `outputs/results/` 디렉토리에 JSON 형식으로 저장됩니다. **각 테스크가 완료될 때마다 자동으로 저장**되므로, 중간에 프로그램이 중단되어도 지금까지의 결과는 보존됩니다.

각 결과 파일에는 다음 정보가 포함됩니다:

- 테스크 이름 및 목표
- 성공 여부 (Success)
- 사용된 턴 수
- 실행 시간
- 최종 턴 요약 (Attacker Prompt, Target Response)
- **턴 히스토리** (`turn_history`): 모든 턴의 상세 정보
  - 각 턴의 원본 프롬프트 (`attacker_original`)
  - 각 턴의 변환된 프롬프트 (`attacker_converted`, 이모지 변환 등)
  - 각 턴의 타겟 응답 (`target_response`)
- 실험 설정

### 로깅 파일

각 테스크의 상세 로깅은 `outputs/logs/` 디렉토리에 별도로 저장됩니다:

- 파일명 형식: `turn_logs_{task_name}_{timestamp}.json`
- 각 턴의 상세 정보 (원본/변환 프롬프트, 응답)
- 백트래킹은 제외하고 실제 턴만 기록

## 프로젝트 구조

```
crescendo/
├── crescendo_eval.py           # 메인 실행 스크립트
├── src/
│   ├── orchestrator.py        # CrescendoExperiment 클래스
│   └── utils.py               # 유틸리티 함수 (결과 저장, 로깅)
├── models/
│   ├── model_configs.py       # 모델 설정
│   ├── model_factory.py       # 모델 팩토리
│   └── anthropic_chat_target.py  # Anthropic Claude 지원
├── data/
│   └── custom_tasks.json      # 테스크 정의
├── outputs/
│   ├── results/               # 결과 저장 디렉토리
│   └── logs/                 # 턴 로깅 저장 디렉토리
├── requirements.txt           # 의존성 목록
└── README.md                  # 이 파일
```

## 요구사항

- Python 3.11+
- PyRIT 프레임워크
- OpenAI API 키 (OpenAI 모델 사용 시)
- HuggingFace 토큰 (HuggingFace 모델 사용 시)
- Anthropic API 키 (Claude 모델 사용 시)