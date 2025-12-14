# gan-tutorials 문서 로드맵 (Part Index)

## Part 00 — Desktop Environment Setup

**파일명**

```
docs/part00_environment_setup_desktop.md
```

**핵심 내용**

* Windows 11 + WSL2 개발 환경 구성
* Git 글로벌 설정 및 SSH 인증
* GitHub 레포지터리 생성 및 연결
* src / tests / docs 표준 프로젝트 구조
* pytest 및 확장 플러그인 설정
* PyTorch 정상 동작 테스트
* README.md 생성
* Commit Prefix Convention 정의

---

## Part 01 — Dataset 구현 (TDD 기초)

**파일명**

```
docs/part01_dataset_tdd.md
```

**핵심 내용**

* TDD 개념 실습 (Red → Green → Refactor)
* Dataset 요구사항 정의
* pytest 기반 Dataset 테스트 작성
* `__len__`, `__getitem__` 구현
* 테스트 실패 → 통과 과정 기록
* Dataset 코드 리팩토링

---

## Part 02 — Generator 구현 (모델 구조 설계)

**파일명**

```
docs/part02_generator_tdd.md
```

**핵심 내용**

* Generator 역할 정의
* 입력 / 출력 Shape 테스트 작성
* PyTorch `nn.Module` 구조 설계
* Forward 동작 검증
* 출력 범위 테스트
* Generator 코드 리팩토링

---

## Part 03 — Discriminator 구현 (판별기 구조)

**파일명**

```
docs/part03_discriminator_tdd.md
```

**핵심 내용**

* Discriminator 역할 및 입력 조건 정의
* Binary 출력 테스트 작성
* 네트워크 구조 구현
* Forward 결과 검증
* 모델 안정성 관련 리팩토링

---

## Part 04 — Trainer 구현 (학습 루프)

**파일명**

```
docs/part04_trainer_tdd.md
```

**핵심 내용**

* Trainer 책임 범위 정의
* 학습 루프 단위 테스트
* Optimizer / Loss 연결
* 1-step 학습 테스트
* 학습 상태 관리
* Trainer 코드 리팩토링

---

## Part 05 — 학습 실행 및 결과 확인

**파일명**

```
docs/part05_training_and_evaluation.md
```

**핵심 내용**

* 전체 파이프라인 연결
* 학습 실행 스크립트 구성
* Loss 변화 확인
* 간단한 결과 시각화
* 실험 재현성 점검

---

## Part 06 — 실험 관리와 코드 확장

**파일명**

```
docs/part06_experiment_management.md
```

**핵심 내용**

* 설정 값 분리 (config 개념)
* 실험 폴더 구조 설계
* 결과 관리 전략
* 반복 실험 패턴 정리
* 코드 확장 포인트 설명

---

## Part 07 — 조건부 모델 확장 (선택)

**파일명**

```
docs/part07_conditional_models.md
```

**핵심 내용**

* 조건 정보 개념 정리
* Dataset 확장 전략
* 조건 입력 처리 방식
* 구조 변경 포인트 설명
* 기존 코드 재사용 전략

---

## Part 08 — 테스트 전략 고도화 (선택)

**파일명**

```
docs/part08_testing_strategy.md
```

**핵심 내용**

* 단위 테스트 vs 통합 테스트
* 테스트 범위 설정
* 실패 테스트 설계 전략
* 테스트 유지보수 기준
* pytest 활용 심화

---

## Part 09 — Git 히스토리로 보는 개발 과정

**파일명**

```
docs/part09_git_history_review.md
```

**핵심 내용**

* 커밋 히스토리 해석 방법
* TDD 흐름이 남는 이유
* feature 브랜치 사용 이력
* 커밋 메시지 품질 분석
* 실무로 이어지는 Git 사용법

---

## Part 10 — 프로젝트 정리 및 다음 단계

**파일명**

```
docs/part10_project_wrapup.md
```

**핵심 내용**

* 전체 구조 재점검
* 핵심 학습 포인트 요약
* 실무 적용 가이드
* 추가 확장 아이디어
* 이후 학습 로드맵 제안
