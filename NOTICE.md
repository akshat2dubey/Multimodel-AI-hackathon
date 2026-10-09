# 주의: 실행 환경
이 프로젝트는 **bash** 환경에서 실행하도록 설계되었습니다.
PowerShell(cmd, powershell.exe)이 아닌 **Git Bash** 또는 WSL bash에서 실행하세요.

## 올바른 실행 방법
```bash
cd "F:/Akshat/edc ideas/LadderLens/MultiModel/hackathon-trackb"
python scripts/run_pipeline.py ...
```

## 잘못된 실행 (이 프로젝트에서 사용하지 않음)
```powershell
# PowerShell — 이 프로젝트에서는 사용하지 않음
python scripts/run_pipeline.py ...
```

## 이유
- 스크립트의 shebang(`#!/usr/bin/env python3`)과 경로 처리가 bash 기준으로 작성됨
- 일부 스크립트의 stdout/stderr 인코딩 처리(`sys.stdout.reconfigure`)가 bash에서 정상 동작
- 터미널 도구(run_terminal_command)도 bash로 명령을 실행함
