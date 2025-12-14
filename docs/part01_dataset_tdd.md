아래는 **`gan-tutorials` 기준의 Part 01 정식 문서 완성본**입니다.
Part 00에서 구축한 환경을 **전제로만 사용**하며, **반복 설명은 제거**하고
**TDD 사고 흐름 + Git 히스토리 관리**에 초점을 맞추었습니다.

문서는 그대로 다음 파일로 저장하면 됩니다.

```
docs/part01_dataset_tdd.md
```

---

# Part 01: Dataset 구현 (TDD)

Test-Driven Development 기반 Dataset 설계와 구현

---

## 0. 목적

이 파트의 목적은 다음과 같다.

* PyTorch Dataset을 **TDD 방식**으로 구현한다
* 테스트 → 구현 → 리팩토링의 흐름을 **Git 히스토리로 남긴다**
* 이후 Generator / Trainer에서 **재사용 가능한 Dataset 설계 기준**을 확립한다

이 파트에서 구현한 Dataset은
**학습 대상이 무엇이든 공통으로 사용 가능한 구조**를 목표로 한다.

---

## 1. 전제 조건

다음 조건이 이미 충족되어 있어야 한다.

* Part 00 완료
* pytest 정상 동작
* main 브랜치가 최신 상태
* SSH 기반 GitHub 연결 완료

---

## 2. 작업 브랜치 생성

Dataset 구현은 독립된 기능이므로 feature 브랜치에서 시작한다.

```bash
git checkout main
git pull origin main
git checkout -b feature/dataset
```

---

## 3. Dataset 요구사항 정의

구현 전에 **명확한 요구사항을 먼저 정의**한다.

### 3.1 Dataset 책임 범위

Dataset은 다음 책임만 가진다.

* 데이터 샘플 개수 제공
* index 기반 데이터 반환
* PyTorch `Dataset` 인터페이스 준수

Dataset은 다음을 **하지 않는다**.

* 모델 로직
* 학습 로직
* 전처리 파이프라인 관리

---

### 3.2 최소 인터페이스

```text
__len__()        → int
__getitem__(i)  → torch.Tensor
```

---

## 4. Red 단계 — 실패하는 테스트 작성

### 4.1 테스트 파일 생성

```bash
cat > tests/test_dataset.py << 'EOF'
import torch
import pytest

from src.dataset import SimpleDataset
EOF
```

이 시점에서 테스트는 **실패가 정상**이다.

---

### 4.2 Dataset 기본 동작 테스트 작성

```bash
cat >> tests/test_dataset.py << 'EOF'

def test_dataset_length():
    dataset = SimpleDataset(size=10)
    assert len(dataset) == 10


def test_dataset_item_type():
    dataset = SimpleDataset(size=5)
    sample = dataset[0]
    assert isinstance(sample, torch.Tensor)


def test_dataset_item_shape():
    dataset = SimpleDataset(size=3, dim=4)
    sample = dataset[0]
    assert sample.shape == (4,)
EOF
```

---

### 4.3 테스트 실행 (실패 확인)

```bash
pytest
```

기대 결과:

```text
ModuleNotFoundError: No module named 'src.dataset'
```

---

### 4.4 Red 단계 커밋

```bash
git add tests/test_dataset.py
git commit -m "test: define dataset interface and behavior"
```

---

## 5. Green 단계 — 최소 구현

### 5.1 Dataset 파일 생성

```bash
cat > src/dataset.py << 'EOF'
import torch
from torch.utils.data import Dataset


class SimpleDataset(Dataset):
    def __init__(self, size, dim=1):
        self.size = size
        self.dim = dim

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return torch.randn(self.dim)
EOF
```

---

### 5.2 테스트 실행 (통과 확인)

```bash
pytest
```

기대 결과:

```text
3 passed in 0.10s
```

---

### 5.3 Green 단계 커밋

```bash
git add src/dataset.py
git commit -m "feat: implement minimal dataset"
```

---

## 6. Refactor 단계 — 구조 개선

기능은 동작하지만, **명확성과 안정성은 부족**하다.

### 개선 목표

* 입력 값 검증
* index 범위 보호
* 코드 가독성 향상

---

### 6.1 Refactor 구현

```bash
cat > src/dataset.py << 'EOF'
import torch
from torch.utils.data import Dataset


class SimpleDataset(Dataset):
    def __init__(self, size, dim=1):
        if size <= 0:
            raise ValueError("size must be positive")
        if dim <= 0:
            raise ValueError("dim must be positive")

        self.size = size
        self.dim = dim

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        if index < 0 or index >= self.size:
            raise IndexError("index out of range")
        return torch.randn(self.dim)
EOF
```

---

### 6.2 테스트 재실행

```bash
pytest
```

모든 테스트는 여전히 통과해야 한다.

---

### 6.3 Refactor 커밋

```bash
git add src/dataset.py
git commit -m "refactor: add validation and safety checks to dataset"
```

---

## 7. Dataset 테스트 확장 (선택)

Dataset의 안정성을 더 높이기 위해 테스트를 확장할 수 있다.

### 7.1 예외 테스트 추가

```bash
cat >> tests/test_dataset.py << 'EOF'

def test_invalid_size():
    with pytest.raises(ValueError):
        SimpleDataset(size=0)


def test_invalid_dim():
    with pytest.raises(ValueError):
        SimpleDataset(size=5, dim=0)
EOF
```

---

### 7.2 테스트 커밋

```bash
git add tests/test_dataset.py
git commit -m "test: add dataset validation tests"
```

---

## 8. main 브랜치 병합

### 8.1 병합 전 확인

```bash
pytest
git log --oneline main..HEAD
```

---

### 8.2 병합 수행

```bash
git checkout main
git pull origin main
git merge feature/dataset
```

---

### 8.3 병합 후 테스트 및 푸시

```bash
pytest
git push origin main
```

---

### 8.4 feature 브랜치 삭제

```bash
git branch -d feature/dataset
git push origin --delete feature/dataset
```

---

## 9. Git 히스토리 해석

이 파트의 히스토리는 다음을 명확히 보여준다.

```text
test: define dataset interface and behavior
feat: implement minimal dataset
refactor: add validation and safety checks to dataset
test: add dataset validation tests
```

이 히스토리 자체가 **Dataset 설계 문서**이다.

---

## 10. 핵심 요약

* 테스트가 설계를 주도한다
* 구현은 테스트를 만족시키는 최소 수준에서 시작한다
* 리팩토링은 기능 변경 없이 구조만 개선한다
* Git 히스토리는 개발 기록이 아니라 **사고 기록**이다

---

## 다음 단계

* Part 02: Generator 구현 (TDD)
* Dataset을 사용하는 모델 입력 구조 설계

다음으로 진행할 파트를 요청해 주세요.
