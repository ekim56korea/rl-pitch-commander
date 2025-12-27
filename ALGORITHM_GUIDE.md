```markdown
# 🧠 Algorithm & Math Guide

본 문서는 투구 전략 최적화를 위해 정의된 **Markov Decision Process (MDP)**의 수학적 구조와 강화학습 방법론을 상세히 기술합니다.

## 1. Problem Formulation (MDP Definition)

우리는 투구 시퀀싱 문제를 **Episodic MDP**로 정의합니다. 한 타석(Plate Appearance)이 하나의 에피소드입니다.

### 1.1 State Space ($S_t$)
투수가 의사결정을 내리기 위해 필요한 모든 관측 정보입니다. 차원 축소를 위해 핵심 변수만 선별합니다.

* **Game Context:** $C_{game} = \{ \text{Balls}, \text{Strikes}, \text{Outs}, \text{BaseRunners}, \text{ScoreDiff} \}$
* **Pitcher State:** $P_{state} = \{ \text{PitchCount}, \text{FatigueLevel}, \text{PrevPitchType}, \text{PrevPitchLoc} \}$
* **Batter Context:** $B_{context} = \{ \text{Handedness}, \text{HotZoneMap}_{9 \times 9}, \text{WhiffRate}_{fastball} \}$

$$S_t = [C_{game}, P_{state}, B_{context}]$$

### 1.2 Action Space ($A_t$)
투수가 제어 가능한 변수들입니다.
* **Pitch Type (Discrete):** $\{ \text{FF(4-Seam)}, \text{SL(Slider)}, \text{CH(Changeup)}, \text{CU(Curve)}, \dots \}$
* **Location (Continuous/Discrete Grid):** 홈 플레이트 상의 좌표 $(x, z)$. 학습 안정성을 위해 $5 \times 5$ 그리드로 이산화하거나, 연속 공간으로 정의합니다.

### 1.3 Reward Function ($R_t$)
가장 중요한 부분으로, 에이전트가 승리(Run Expectancy 최소화)를 지향하도록 설계합니다.

$$R_t = R_{outcome} + \lambda_1 R_{deception} - \lambda_2 R_{fatigue}$$

1.  **Outcome Reward ($R_{outcome}$):** **Delta Run Value (RE24)** 기반.
    * Strike: $+0.05$ (상황에 따라 가변)
    * Ball: $-0.06$
    * Strikeout: $+0.25$
    * Home Run: $-1.40$
2.  **Deception Bonus ($R_{deception}$):** 피치 터널링(Tunneling) 효과.
    * 직전 투구와 릴리스 포인트 및 초반 궤적이 유사할수록 보상 부여.
3.  **Fatigue Penalty ($R_{fatigue}$):**
    * 최대 구속 투구를 연속으로 할 경우 페널티 부여 (생체역학적 보호).

## 2. Batter Behavior Modeling (The World Model)
강화학습 환경의 핵심인 타자 모델 $P(O_t | S_t, A_t)$은 다음과 같이 구성됩니다.

* **Swing Probability:**
    $$P(\text{Swing}) = \sigma(W \cdot \phi(S_t, A_t) + b)$$
    ($\sigma$: Sigmoid, $\phi$: Feature Vector derived from XGBoost)
* **Contact Quality:**
    스윙 시, `Launch Angle`과 `Exit Velocity`는 타자의 과거 타구 데이터 분포(KDE)와 투구의 물리적 특성(VAA, Spin)을 조건부 확률로 샘플링하여 결정합니다.

## 3. Training Strategy

### Phase 1: Behavior Cloning (BC)
Random Initialization 문제를 해결하기 위해, 2023년 MLB 상위 10% 투수(ERA 기준)의 (State, Action) 쌍을 지도 학습합니다.
$$\mathcal{L}_{BC} = - \sum \log \pi_\theta(a_{expert} | s)$$

### Phase 2: Proximal Policy Optimization (PPO)
BC로 초기화된 정책 $\pi_\theta$를 시작점으로 하여, PPO를 통해 기대 보상(Run Value 최소화)을 극대화합니다.
$$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t [\min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)]$$

---
**References:**
- *Sidiger et al., "Optimizing Pitch Sequencing with Deep RL", MIT Sloan Sports Analytics Conference.*
- *Tango et al., "The Book: Playing the Percentages in Baseball".*