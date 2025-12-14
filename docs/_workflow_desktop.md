아래에 **파일명 제안**과 **축약형(숙련자용) 개발 루틴 문서**를 함께 제공합니다.
둘 다 `docs/` 폴더에 그대로 저장해 사용할 수 있도록 설계했습니다.

---

## 1. 문서 파일명 제안

이 문서는 **“하루 개발 표준 절차(SOP)”** 성격이므로, Part 문서와는 분리하는 것이 좋습니다.

### 최종 추천 파일명

```text
docs/daily_feature_development_workflow.md
```

### 대안 후보 (참고)

```text
docs/feature_development_routine.md
docs/daily_git_workflow.md
```

하지만 **교육 + 실무 혼용** 기준에서는
`daily_feature_development_workflow.md`가 가장 명확합니다.

---

## 2. 숙련자용 요약 문서 (Checklist / Bullet Style)

아래 내용은 **위에서 작성한 전체 문서를 100% 포함하되**,
반복 사용을 전제로 **최소한의 확인 포인트만 남긴 버전**입니다.

이 문서는 같은 파일 하단에 `Appendix`로 두거나,
별도 파일로 저장해도 됩니다.

---

# Daily Feature Development Checklist (Condensed)

gan-tutorials · Desktop Workflow

---

## 0. 작업 시작

* [ ] Desktop 부팅
* [ ] 터미널 실행
* [ ] 프로젝트 이동

```bash
cd ~/github/gan-tutorials
```

---

## 1. main 최신화

* [ ] 현재 브랜치 확인
* [ ] main 최신 상태 반영

```bash
git checkout main
git pull origin main
```

---

## 2. Feature 브랜치 생성

* [ ] 작업 목적에 맞는 브랜치명 결정
* [ ] feature 브랜치 생성

```bash
git checkout -b feature/<topic>
```

---

## 3. 개발 사이클 (TDD)

### 3.1 Red — 테스트 작성

* [ ] tests/ 에 실패하는 테스트 작성
* [ ] pytest 실패 확인

```bash
pytest
```

```bash
git commit -m "test: add <topic> tests"
```

---

### 3.2 Green — 기능 구현

* [ ] src/ 코드 구현
* [ ] 테스트 통과 확인

```bash
pytest
```

```bash
git commit -m "feat: implement <topic>"
```

---

### 3.3 Refactor — 구조 개선

* [ ] 코드 정리
* [ ] 테스트 재확인

```bash
pytest
```

```bash
git commit -m "refactor: improve <topic> structure"
```

---

## 4. 병합 전 점검

* [ ] feature 브랜치 커밋 목록 확인
* [ ] 전체 테스트 통과

```bash
git log --oneline main..HEAD
pytest
```

---

## 5. main 병합

```bash
git checkout main
git pull origin main
git merge feature/<topic>
```

* [ ] 충돌 시 수동 해결
* [ ] 병합 후 테스트 실행

```bash
pytest
```

---

## 6. 마무리

* [ ] main 푸시

```bash
git push origin main
```

* [ ] feature 브랜치 삭제

```bash
git branch -d feature/<topic>
git push origin --delete feature/<topic>   # 원격 사용 시
```

---

## 7. 종료 전 상태 확인

```bash
git status
```

기대 상태:

```text
On branch main
nothing to commit, working tree clean
```

---

## 핵심 원칙 (암기용)

* main은 항상 테스트 통과 상태
* feature 브랜치는 작업 단위
* 병합 후 feature 삭제
* 커밋 히스토리가 개발 문서

---

## 추천 문서 구성 예시

```text
docs/
├── part_index.md
├── part00_environment_setup_desktop.md
├── daily_feature_development_workflow.md
└── part01_dataset_tdd.md
```
