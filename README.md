# gan-tutorials

PyTorch 기반 딥러닝 실습을 통해  
Git 버전 관리와 TDD(Test-Driven Development)를 함께 학습하기 위한 교육용 프로젝트입니다.

본 저장소는 **단계별(Part 기반) 실습 문서**, **소스 코드**, **테스트 코드**가  
서로 1:1로 대응되도록 설계되어 있습니다.

---

## 프로젝트 목표

- Git을 활용한 체계적인 코드 이력 관리
- pytest 기반 TDD 개발 사이클 학습
- 실무에 바로 적용 가능한 프로젝트 구조 습득
- 단계별 확장 가능한 딥러닝 코드베이스 구성

---

## 프로젝트 구조

```

gan-tutorials/
├── docs/                   # 실습 문서 (Part 단위)
│   ├── part00_environment_setup_desktop.md
│   ├── part01_dataset_tdd.md
│   ├── part02_generator_tdd.md
│   ├── part03_discriminator_tdd.md
│   ├── part04_trainer_tdd.md
│   └── ...
│
├── src/                    # 실제 소스 코드
│   └── gan-tutorials/
│       ├── **init**.py
│       ├── dataset.py
│       ├── generator.py
│       ├── discriminator.py
│       ├── trainer.py
│       └── ...
│
├── tests/                  # pytest 기반 테스트 코드
│   ├── test_00_environment.py
│   ├── test_01_dataset.py
│   ├── test_02_generator.py
│   ├── test_03_discriminator.py
│   ├── test_04_trainer.py
│   └── conftest.py
│
├── notebooks/              # 실험 및 참고용 노트북
│
├── outputs/                # 학습 결과물 (git 추적 제외)
│
├── .gitignore
├── pytest.ini
└── README.md

````

---

## 디렉토리 설명

### docs/
- Part 단위 실습 문서 저장
- 학습 순서를 기준으로 `partXX_` 접두어 사용
- 문서 흐름 = 전체 커리큘럼 흐름

예시:
- `part00_environment_setup_desktop.md`
- `part01_dataset_tdd.md`

---

### src/
- 실제 Python 소스 코드
- 배포 표준을 따르는 `src/` 구조 사용
- 구현 대상은 Part 문서 및 테스트와 직접 대응

---

### tests/
- pytest 기반 테스트 코드
- 파일명은 반드시 `test_XX_*.py` 형식
- 각 테스트 파일은 해당 Part의 구현 내용을 검증

예시:
- `test_01_dataset.py` ↔ `part01_dataset_tdd.md`

---

### notebooks/
- 코드 실험, 결과 확인, 설명 보조용
- 정식 구현 및 테스트와는 분리
- 학습 참고용

---

### outputs/
- 모델 체크포인트, 로그, 결과물 저장
- `.gitignore`로 Git 추적 제외

---

## 개발 및 실행 환경

- OS: Windows 11
- Linux: WSL2 (Ubuntu)
- Python: Anaconda (pytorch_env)
- Framework: PyTorch
- Test: pytest (+ pytest-cov, pytest-mock, pytest-benchmark)
- VCS: Git + GitHub (SSH)

---

## 기본 워크플로우

1. 문서 확인 (docs/partXX_*.md)
2. 테스트 작성 (tests/test_XX_*.py)
3. 테스트 실패 확인 (Red)
4. 최소 구현 (Green)
5. 리팩토링 (Refactor)
6. 커밋 기록으로 개발 단계 관리

---

## 테스트 실행

```bash
pytest
````

커버리지 확인:

```bash
pytest --cov=src
```

---

## Git 브랜치 전략 (요약)

* `main`: 안정된 기준 브랜치
* `develop`: 개발 통합 브랜치
* `feature/*`: Part 또는 기능 단위 작업 브랜치

---

## 문서 진행 순서

1. Part 00: Environment Setup
2. Part 01: Dataset (TDD)
3. Part 02: Generator (TDD)
4. Part 03: Discriminator (TDD)
5. Part 04: Trainer (TDD)
6. 이후 확장 Part

---

## 참고

* 본 저장소는 **교육 목적**을 우선으로 설계되었습니다.
* 실무 프로젝트 구조와 최대한 유사하게 구성하되,
  학습 가독성과 단계적 이해를 중시합니다.

---


## README 작성 규칙 요약 (앞으로의 기준)

- README는 **“진입 문서”** 역할만 수행
- 상세 설명은 docs/Part 문서로 이동
- 폴더 구조는 반드시 README에 포함
- 실행 예시는 최소한만 포함
- 장황한 설명, 튜토리얼은 README에 넣지 않음

---

다음으로 진행 가능:
- README에 **Part 링크 자동 목록** 추가
- docs/README.md (문서 인덱스 전용)
- 교육용 레포에서 자주 쓰는 README 템플릿 분리

