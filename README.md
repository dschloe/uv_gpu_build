# Github 설정

## GitHub 저장소 연결 방법

### 새 저장소를 처음 만들 때

```bash
echo "# uv_gpu_build" >> README.md   # README 파일 생성
git init                              # 로컬 git 저장소 초기화
git add README.md                     # 파일 스테이징
git commit -m "first commit"         # 첫 커밋
git branch -M main                   # 브랜치 이름을 main으로 변경
git remote add origin https://github.com/dschloe/uv_gpu_build.git  # 원격 저장소 연결
git push -u origin main              # 원격으로 푸시 (-u: 이후 git push만으로 가능)
```

### 기존 로컬 저장소를 GitHub에 연결할 때

```bash
git remote add origin https://github.com/dschloe/uv_gpu_build.git  # 원격 저장소 연결
git branch -M main                   # 브랜치 이름을 main으로 변경
git push -u origin main              # 원격으로 푸시
```

> 현재 디렉토리가 이미 존재하는 경우 두 번째 방법을 사용하세요.
