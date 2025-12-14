아래는 **기존 `part00_environment_setup_desktop.md`를 기준으로**,
요청하신 **3가지 항목을 자연스럽게 확장·통합한 최종 Part 0 문서**입니다.

추가·보강된 내용은 다음입니다.

* docs 문서 구조 및 작성 규칙 추가
* README.md 생성 및 역할 명시
* Commit Prefix Convention (7개 접두어 표 포함)

문서는 **교육용 기준 + 실무 재사용 가능**하도록 구성되어 있으며,
그대로 `docs/part00_environment_setup_desktop.md`에 저장해도 되는 **완성본**입니다.

---

# Part 0: Desktop 환경 설정

Git · SSH · pytest 기반 개발 환경 구축 (Windows 11)

---

## 0. 목적

이 문서는 **Desktop 환경(Windows 11)**에서 다음을 한 번에 구축하는 것을 목표로 합니다.

* Git 기반 버전 관리 환경
* GitHub SSH 인증 설정
* pytest 기반 테스트 환경
* PyTorch 정상 동작 확인
* 문서(docs) + 코드(src) + 테스트(tests)가 분리된 표준 프로젝트 구조
* 이후 모든 프로젝트에 재사용 가능한 **개발 베이스라인**

본 Part 0은 **모든 실습의 공통 기반**이며, 이후 파트에서는 반복하지 않습니다.

---

## 1. 작업 환경 명세

| 항목        | 값                      |
| --------- | ---------------------- |
| OS        | Windows 11             |
| Linux     | WSL2 (Ubuntu)          |
| Python    | Anaconda (pytorch_env) |
| Framework | PyTorch                |
| Editor    | VS Code                |
| VCS       | Git + GitHub (SSH)     |
| Test      | pytest + 확장 플러그인       |

---

## 2. Git 기본 설정 (최초 1회)

### 2.1 Git 설치 확인

```bash
git --version
```

정상 출력 예시:

```text
git version 2.43.0
```

---

### 2.2 사용자 정보 설정

```bash
git config --global user.name "홍길동"
git config --global user.email "hong@example.com"

git config --global core.editor "code --wait"
git config --global core.quotepath false
```

확인:

```bash
git config --list
```

---

## 3. SSH 설정 (강력 권장)

### 3.1 SSH가 필요한 이유

* 매 `push / pull` 시 비밀번호 입력 제거
* GitHub 공식 권장 인증 방식
* 여러 레포지터리에서 **하나의 키로 재사용 가능**
* Desktop / Laptop **기기 단위로 1회만 설정**

---

### 3.2 SSH 키 생성 (Desktop 1회)

```bash
ssh-keygen -t ed25519 -C "hong@example.com"
```

모든 질문은 **Enter**로 진행.

생성 파일:

* 개인키: `~/.ssh/id_ed25519`
* 공개키: `~/.ssh/id_ed25519.pub`

---

### 3.3 SSH Agent 등록

```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```

---

### 3.4 공개 키 확인 (중요)

```bash
cat ~/.ssh/id_ed25519.pub
```

출력 예시:

```text
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAI... hong@example.com
```

이 **한 줄 전체**를 복사한다.

---

### 3.5 GitHub에 SSH 키 등록

GitHub →
Settings → SSH and GPG keys → New SSH key

* Title: `Desktop-WSL2`
* Key: `cat`으로 출력한 전체 문자열

---

### 3.6 연결 테스트

```bash
ssh -T git@github.com
```

성공 메시지:

```text
Hi username! You've successfully authenticated.
```

---

## 4. GitHub 레포지터리 생성

### 4.1 GitHub 웹에서 생성

* Repository name: `gan-tutorials`
* Description: `PyTorch + Git + TDD tutorials`
* Public / Private 선택
* README / .gitignore **생성하지 않음**

---

## 5. 프로젝트 초기화 (Desktop)

### 5.1 로컬 디렉토리 생성

```bash
cd ~/github
mkdir gan-tutorials
cd gan-tutorials
code .
```

---

### 5.2 표준 프로젝트 구조 생성

```bash
mkdir -p docs
mkdir -p src/gan-tutorials
mkdir -p tests
mkdir -p notebooks
mkdir -p outputs

touch src/gan-tutorials/__init__.py
touch tests/__init__.py
```

구조:

```text
gan-tutorials/
├── docs/              # 실습 문서 (Part 단위)
├── src/
│   └── gan-tutorials/ # 실제 소스 코드
│       └── __init__.py
├── tests/             # pytest 테스트
│   └── __init__.py
├── notebooks/         # 참고용 노트북
├── outputs/           # 실행 결과 (git 추적 제외)
```

---

### 5.3 .gitignore 생성

```bash
cat > .gitignore << 'EOF'
__pycache__/
*.py[cod]
.ipynb_checkpoints/

*.pth
*.pt
*.ckpt

outputs/
logs/
runs/
data/

.coverage
htmlcov/
.pytest_cache/

.vscode/
.idea/
.DS_Store
EOF
```

---

## 6. pytest 설치 및 설정

### 6.1 pytest 관련 패키지 설치

```bash
pip install pytest pytest-cov pytest-mock pytest-benchmark
```

확인:

```bash
pytest --version
```

---

### 6.2 pytest 설정 파일

```bash
cat > pytest.ini << 'EOF'
[pytest]
testpaths = tests
python_files = test_*.py
addopts = -v --tb=short
EOF
```

---

## 7. PyTorch 정상 동작 확인

### 7.1 테스트 코드 작성

```bash
cat > tests/test_environment.py << 'EOF'
import torch

def test_torch_installed():
    assert torch.__version__ is not None

def test_tensor_creation():
    x = torch.randn(2, 3)
    assert x.shape == (2, 3)

def test_cuda_info():
    print("CUDA available:", torch.cuda.is_available())
EOF
```

---

### 7.2 테스트 실행

```bash
pytest
```

정상 출력 예시:

```text
3 passed in 0.20s
```

---

## 8. docs 문서 구조 및 작성 규칙

### 8.1 docs 디렉토리 역할

* `docs/`는 **실습 문서 전용 디렉토리**
* Part 단위로 하나의 Markdown 파일 사용
* 코드와 테스트가 아닌 **설명·절차·이론 중심**

---

### 8.2 파일명 규칙

```text
docs/
├── part00_environment_setup_desktop.md
├── part01_dataset_tdd.md
├── part02_generator_tdd.md
├── part03_discriminator_tdd.md
```

규칙:

* `part` + 두 자리 숫자
* 소문자 + underscore
* Part 번호 = 학습 단계

---

## 9. README.md 생성 및 역할

### 9.1 README.md 역할

* 레포지터리 **진입 문서**
* 프로젝트 목적, 구조, 워크플로우 요약
* 상세 실습 내용은 `docs/partXX`에서 다룸

---

### 9.2 README.md 생성

```bash
cat > README.md << 'EOF'
# gan-tutorials

PyTorch 기반 실습을 통해 Git 버전 관리와 TDD(Test-Driven Development)를 함께 학습하기 위한 교육용 프로젝트입니다.

## Project Structure

gan-tutorials/
├── docs/        # Part 단위 실습 문서
├── src/         # 실제 소스 코드
├── tests/       # pytest 기반 테스트 코드
├── notebooks/   # 참고용 노트북
└── outputs/     # 실행 결과 (git 추적 제외)

## Branch Strategy

- main: 안정 상태 기준 브랜치
- feature/*: Part 단위 작업 브랜치

## Workflow

1. docs/partXX 문서 확인
2. tests 작성 (Red)
3. src 구현 (Green)
4. 리팩토링 (Refactor)
5. feature 브랜치 → main 병합
EOF
```

---

## 10. Commit Prefix Convention (공통 규칙)

모든 커밋은 **아래 접두어 중 하나만 사용**한다.

### 10.1 허용되는 접두어 요약

| 접두어         | 용도     | 사용 시점                   |
| ----------- | ------ | ----------------------- |
| `feat:`     | 기능 구현  | 새로운 로직·클래스 추가           |
| `test:`     | 테스트 코드 | pytest 테스트 추가·수정        |
| `refactor:` | 리팩토링   | 기능 변경 없는 구조 개선          |
| `fix:`      | 버그 수정  | 잘못된 동작 수정               |
| `docs:`     | 문서     | README, docs 수정         |
| `chore:`    | 설정·기타  | 환경, 구조, 설정 작업           |
| `debug:`    | 디버깅    | 임시 확인용 (feature 브랜치 한정) |

### 10.2 커밋 메시지 형식

```text
<prefix>: <한 줄 설명>
```

예시:

```bash
git commit -m "feat: dataset 클래스 생성"
git commit -m "test: dataset length 테스트 추가"
git commit -m "refactor: dataset 코드 정리"
```

---

## 11. Git 저장소 초기화 및 원격 연결

### 11.1 Git 초기화

```bash
git init
git branch -M main
```

---

### 11.2 원격 저장소 연결 (SSH)

```bash
git remote add origin git@github.com:사용자명/gan-tutorials.git
git remote -v
```

---

### 11.3 첫 커밋

```bash
git add .
git commit -m "chore: initial project setup

- src-based project layout
- pytest configuration
- SSH-based Git workflow
- docs and README structure
"
```

---

### 11.4 첫 Push

```bash
git push -u origin main
```

---

## 12. Part 0 완료 체크리스트

```text
[ ] Git global 설정 완료
[ ] SSH 키 생성 및 GitHub 등록
[ ] gan-tutorials 레포지터리 생성
[ ] src 기반 프로젝트 구조 생성
[ ] docs / README 역할 분리
[ ] pytest 및 확장 패키지 설치
[ ] PyTorch 테스트 통과
[ ] SSH 기반 push 성공
```

---

## 다음 단계

* Part 1: Dataset부터 시작하는 **TDD 기반 개발 사이클**
* Red → Green → Refactor 커밋 히스토리 관리

---

이제 Part 0는 **환경·문서·커밋 규칙까지 포함한 완전한 기준 문서**가 되었습니다.
다음으로 원하시면 **Part 1 템플릿(테스트부터 시작)**을 바로 작성해 드릴 수 있습니다.
