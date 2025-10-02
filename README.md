# F1-Race-Result-Predictor
# 🏎️ F1 Predictor

A statistical and simulation-based framework to forecast Formula 1 race outcomes and the World Drivers’ Championship (WDC).  
The model blends **driver momentum, qualifying pace, track history, reliability, and season context** into win probabilities, then uses **Monte Carlo** to generate position probabilities and finishing orders.

---

## 📊 Methods & Formulas

### 1) Rolling Form (Driver Momentum)
\[
\text{RollingForm}_d = \frac{\sum_{i=1}^n w_i \cdot \text{Points}_{d,i}}{\sum_{i=1}^n w_i}, \quad w_i = (0.75)^{(n-1-i)}
\]
Scaled to 0–100 by dividing by 25 (max FIA race points).  
**Why:** Emphasizes **recent form**. Each older race is worth ¾ of the next-most-recent, so breakthroughs (e.g., rookies improving fast) aren’t dragged down by early results.

### 2) Qualifying Gap to Pole
\[
\text{GapToPole}_d = t_d^{best} - t_{pole}^{best}
\]
Where \(t_d^{best}\) is the driver’s best of Q1–Q3.  
**Why:** Direct measure of **raw car+driver speed**, less noisy than race results (safety cars, strategy).

### 3) Pace Score (Pace100)
\[
\text{Pace100}_d = \max\!\Big( S_{min},\; 100 \cdot e^{-\frac{\min(g_d, g_{cap})}{\tau}} \Big),\quad
\tau = \frac{g_{cap}}{\ln(100/S_{cap})}
\]
With \(g_{cap}\) at the 90th percentile; e.g., \(S_{cap}=20,\; S_{min}=12\).  
**Why:** Converts time gaps to a **0–100** speed score; reflects that front-end gaps matter **non-linearly** more than back-end gaps and keeps everyone in-model via a floor.

### 4) Track Performance Score (Driver vs Track)
\[
\text{TrackScore}_{d,\text{track}} = \frac{\sum_{r \in \text{track}} f(\text{Pos}_{d,r})}{N_{d,\text{track}}}
\]
Using a custom scoring table (1st=40, 2nd=35, 3rd=28, …, 20th=1).  
**Why:** Encodes **driver–track history**; rewards podiums more than midfield consistency and averages across seasons (2018–2025) to reduce noise.

### 5) DNF Chance
\[
P(\text{DNF}_{d,\text{track}}) = \frac{\text{DNFs}_{d,\text{track}}}{\text{Starts}_{d,\text{track}}}
\]
**Why:** Introduces **reliability risk**; some circuits/drivers have higher DNF rates (e.g., street tracks).

### 6) Championship Rating
\[
\text{Champ01}_d = \max\!\left(S_{min},\; 0.5^{\frac{\text{Rank}_d - 1}{H_{champ}}}\right)
\]
**Why:** Adds **season context** via an exponential rank decay (half-life \(H_{champ}\)), rewarding contenders while keeping a floor for backmarkers.

### 7) Driver Strength & Win Probability
\[
S_{d,\text{track}} = a\cdot\text{Pace}_d + b\cdot\text{TrackScore}_d + c\cdot\text{RollingForm}_d + d\cdot(1-P(\text{DNF})) + e\cdot\text{Champ01}_d
\]
\[
P(\text{Win}_{d,\text{track}}) = \frac{\exp(S_{d,\text{track}}/T)}{\sum_j \exp(S_{j,\text{track}}/T)}
\]
**Why:** Blends short-term and long-term factors; **softmax temperature** \(T>1\) flattens probabilities to reflect F1 uncertainty.

### 8) Monte Carlo Simulation
- **Stage 1 (≈100k sims):** Plackett–Luce (Gumbel–Max) draws → **position probabilities** P1–P20 per race.  
- **Stage 2 (≈10M sims):** Repeated draws using position probabilities → **finishing orders** and tables per GP.  
**Why:** Captures full **outcome distributions** (including upsets), not just point estimates.

### 9) Model Evaluation (Calibration, Optional)
- **Brier Score**: average squared error of predicted category probabilities. Lower = better calibration.  
- **Log Loss**: penalizes overconfident wrong predictions. Lower = better.

---

## 📂 Project Structure
