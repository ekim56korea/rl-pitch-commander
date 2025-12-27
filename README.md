# ⚾ RL Pitch Commander: Deep Reinforcement Learning for Optimal Pitch Sequencing

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Research%20Preview-orange)]()

> **"Prediction is just the beginning. Prescription is the goal."**

## 1. Project Overview (프로젝트 개요)
**RL Pitch Commander**는 야구 데이터 분석의 패러다임을 '예측(What will happen?)'에서 **'최적 제어(What should happen?)'**로 전환하는 차세대 투구 의사결정 시스템입니다.

기존의 투구 모델들이 단순히 "과거에 투수가 무엇을 던졌는가"를 학습하여 모방하는 데 그쳤다면, 본 프로젝트는 **심층 강화학습(Deep Reinforcement Learning)**을 통해 기대 득점(Expected Run Value)을 최소화하고, 타자의 타이밍을 뺏으며(Deception), 투수의 생체역학적 효율성(Biomechanics)을 극대화하는 **Nash Equilibrium(내쉬 균형)** 전략을 도출합니다.

## 2. Key Differentiators (핵심 차별점)
1.  **Prediction vs Prescription:** 투수의 습관을 맞추는 것이 아니라, 승리를 위한 **최적의 해(Optimal Solution)**를 제안합니다.
2.  **Physics-Informed RL:** 단순 통계가 아닌, **VAA(수직 입사각)**, **SSW(Seam-Shifted Wake)** 등 최신 트래킹 데이터를 반영한 물리적 환경을 구축했습니다.
3.  **Counterfactual Simulation:** "만약 그때 커브 대신 직구를 던졌다면?"에 대한 인과 추론(Causal Inference)이 가능한 타자 시뮬레이터를 내장하고 있습니다.
4.  **Game Theory:** 타자가 투구 패턴에 적응하는 것을 방지하기 위해 혼합 전략(Mixed Strategy)을 학습합니다.

## 3. Tech Stack
- **Core Engine:** Python 3.9+, PyTorch
- **RL Framework:** Gymnasium (Environment), Stable-Baselines3 / Ray RLLib (Agent)
- **Data Pipeline:** DuckDB (OLAP), dbt (Transformation)
- **Modeling:** XGBoost (Batter Response Model), Transformer (Sequential Modeling)
- **Deployment:** Docker, FastAPI, Streamlit (Dashboard)

## 4. Getting Started
### Prerequisites
- Python 3.9+
- Docker (Optional for deployment)

### Installation
```bash
git clone [https://github.com/ekim56korea/rl-pitch-commander.git](https://github.com/ekim56korea/rl-pitch-commander.git)
cd rl-pitch-commander
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

Usage

Bash
# 데이터베이스 초기화 및 전처리
python -m src.database.initialize

# 강화학습 에이전트 학습 시작 (PPO)
python -m src.train --algo ppo --timesteps 1000000

# 대시보드 실행
streamlit run src/frontend/app.py