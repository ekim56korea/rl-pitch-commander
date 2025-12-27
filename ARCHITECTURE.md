
### 2. `ARCHITECTURE.md`
**역할:** 시스템 설계도. 데이터의 흐름과 각 모듈(환경, 에이전트, 모델)의 상호작용을 기술적으로 설명합니다.

# 🏗️ System Architecture

본 문서는 `rl-pitch-commander`의 기술적 아키텍처와 데이터 파이프라인, 그리고 모듈 간의 상호작용을 설명합니다.

## 1. High-Level Architecture Diagram
시스템은 크게 **Data Layer**, **Simulation Environment (Digital Twin)**, **Agent Core**, **Application Layer**로 구성됩니다.

```mermaid
graph TD
    subgraph "Data Layer (DuckDB)"
        Raw[Savant Raw Data] --> |dbt| Feat[Feature Store<br/>(VAA, RunValue, History)]
    end

    subgraph "Simulation Environment (Gymnasium)"
        Feat --> Physics[Physics Engine]
        Feat --> Batter[Batter Behavior Model<br/>(XGBoost/Transformer)]
        Feat --> Umpire[Umpire Model<br/>(Probabilistic Zone)]
        Physics & Batter & Umpire --> State[State Generator]
        State --> Reward[Reward Calculator]
    end

    subgraph "Agent Core (RL)"
        State --> Policy[Policy Network<br/>(Actor-Critic / LSTM)]
        Policy --> Action[Action Selection<br/>(Pitch Type, Location)]
        Action --> Physics
        Reward --> Policy
    end

    subgraph "Application Layer"
        Policy --> API[Inference API<br/>(FastAPI)]
        API --> UI[Analyst Dashboard<br/>(Streamlit/React)]
    end
2. Component Details
2.1 Data Lakehouse (DuckDB)

Storage: 2015-2025 MLB Pitch Data (savant.duckdb).

Feature Engineering: dbt를 사용하여 plate_x, plate_z 등의 좌표 데이터와 release_spin_rate 등을 결합, 타자별 Hot/Cold Zone 및 Pitch Value를 사전 계산.

Optimization: 인메모리 OLAP 처리를 통해 학습 시 Replay Buffer로의 고속 데이터 전송 지원.

2.2 The Digital Twin (Environment)

실제 경기와 동일한 보상과 상태 전이를 제공하는 가상 환경입니다.

Batter Simulator:

입력: 투구 정보(구종, 구속, 위치, 무브먼트) + 타자 ID + 카운트.

출력: Swing 여부, Contact 품질(Exit Velocity, Launch Angle).

모델: Gradient Boosting Machine (XGBoost) 기반, 타자별 개별 모델링.

Physics Engine: 공기역학적 항력(Drag)과 마그누스 효과를 고려하여 릴리스 포인트에서 포수 미트까지의 궤적 계산.

2.3 Agent Core (The Brain)

Algorithm: PPO (Proximal Policy Optimization) with LSTM Support.

Network: 투구 시퀀스(이전 공들의 정보)를 기억하기 위한 Recurrent Neural Network 구조 사용.

Initialization: Behavior Cloning(BC)을 통해 실제 MLB 에이스 투수들의 정책으로 사전 학습(Pre-training) 후 강화학습 적용.

3. Infrastructure & MLOps
Containerization: 모든 모듈은 Docker로 컨테이너화되어 의존성 충돌 방지.

Experiment Tracking: TensorBoard 및 MLflow를 사용하여 Reward 곡선 및 하이퍼파라미터 추적.


---