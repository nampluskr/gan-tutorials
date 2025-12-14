# Part 0: Desktop 환경 설정

Git · SSH · pytest 기반 개발 환경 구축 (Windows 11)

---

## 0. 목적

이 문서는 **Desktop 환경(Windows 11)**에서 다음을 한 번에 구축하는 것을 목표로 합니다.

* Git 기반 버전 관리 환경
* GitHub SSH 인증 설정
* pytest 기반 테스트 환경
* PyTorch 정상 동작 확인
* 이후 모든 프로젝트에 재사용 가능한 표준 개발 베이스

본 파트는 **모든 실습의 공통 기반**이며, 이후 파트에서는 반복하지 않습니다.

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
* GitHub 권장 인증 방식
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
├── src/
│   └── gan-tutorials/
│       └── __init__.py
├── tests/
│   └── __init__.py
├── notebooks/
├── outputs/
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

## 8. Git 저장소 초기화 및 원격 연결

### 8.1 Git 초기화

```bash
git init
git branch -M main
```

---

### 8.2 원격 저장소 연결 (SSH)

```bash
git remote add origin git@github.com:사용자명/gan-tutorials.git
git remote -v
```

---

### 8.3 첫 커밋

```bash
git add .
git commit -m "chore: initial project setup

- src-based project layout
- pytest configuration
- SSH-based Git workflow
"
```

---

### 8.4 첫 Push

```bash
git push -u origin main
```

---

## 9. Part 0 완료 체크리스트

```text
[ ] Git global 설정 완료
[ ] SSH 키 생성 및 GitHub 등록
[ ] gan-tutorials 레포지터리 생성
[ ] src 기반 프로젝트 구조 생성
[ ] pytest 및 확장 패키지 설치
[ ] PyTorch 테스트 통과
[ ] SSH 기반 push 성공
```

---

## 다음 단계

* Part 1: Dataset부터 시작하는 **TDD 기반 개발 사이클**
* Red → Green → Refactor 커밋 히스토리 관리

원하시면 다음으로:

* Part 1-1 (Dataset + TDD)
* 또는 **Laptop 환경용 Part 0**를 별도로 작성해 드릴 수 있습니다.
