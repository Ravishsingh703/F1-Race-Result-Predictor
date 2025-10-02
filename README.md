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

```
Data/                         # Raw, uncleaned FIA/fastf1 data (qualifying + race)
  2018/
  2019/
  ...
  2025/

Data_Processed/
  fastf1/
    2018/                     # Clean per-race feature CSVs (e.g., 2018_R09_features.csv)
    ...
    2025/
  driver_stats_over_years/
    current_driver_stats_2025.csv    # Rolling form (0–100) + season-to-date stats
    ... (derived from 2018–2025 to compute:)
      - finish rate
      - podium rate
      - DNF rate
      - penalty rate
      - pit efficiency
      - (then Rolling Form computed using these + points)
  track_scores/
    track_scores_2018_2025.csv       # Historical driver vs track scoring (2018–2025)
    track_scores_2025.csv
  prediction_data/
    pace_scores_2025_R17_features.csv
    dnf_chance_2025_R17_features.csv
    win_probs_Azerbaijan_Grand_Prix.csv
    17_position_probability.csv
    17_final_prediction.csv
    ...                               # Files produced by event predictor modules
  wdc_prediction/
    pace_by_track_gridexp_2018to2025.csv
    driver_vs_track_score_matrix.csv
    dnf_chance_matrix.csv
    win_probabilities_remaining_2025.csv
    finish_position_probabilities_2025.csv
    ...                               # WDC-level matrices & outputs
  quali_feat/                         # Used in race predictor (not in WDC pipeline)
    ... (quali-derived features per event)

Scripts/
  fastf1.py                           # Cleans raw data → Data_Processed/fastf1/YYYY
  rolling_form.py                     # Builds rolling form, pace scores, DNF statistics
  quali_feat.py                       # Qualifying feature builder (for race predictor only)
  predictor_with_quali_feat.py        # Event-level predictor (pace, DNF, win prob) post-quali
  remaining_race_simulator.py         # Season-level predictor: sims → position prob → finishing order
```

---

## 🔧 Running the Pipeline (Order Matters)

```bash
# 1) Clean & process raw GP data into per-race features
python fastf1.py

# 2) Build rolling form, DNF stats, and pace scores from processed data
python rolling_form.py

# 3) (Optional) Build qualifying-based features for event-level prediction
python quali_feat.py

# 4) Event predictor: compute pace scores, DNF chances, win probabilities post-quali
python predictor_with_quali_feat.py

# 5) Season predictor: run simulations for remaining races → position probabilities and finishing order
python remaining_race_simulator.py
```

Outputs are written into the appropriate subfolders in `Data_Processed/`, notably:
- **Event-level** artifacts → `Data_Processed/prediction_data/`
- **WDC-level** artifacts → `Data_Processed/wdc_prediction/`

---

## 🗺️ Data Flow (Mermaid)

```mermaid
flowchart LR
  A[Data/<year>/ (raw quali + race)] --> B[fastf1.py]
  B --> C[Data_Processed/fastf1/<year>/
clean per-race features]

  C --> D[rolling_form.py]
  D --> D1[Data_Processed/driver_stats_over_years/
current_driver_stats_2025.csv]
  D --> D2[Data_Processed/wdc_prediction/
pace_by_track_gridexp_2018to2025.csv]
  D --> D3[Data_Processed/wdc_prediction/
dnf_chance_matrix.csv]

  C --> E[quali_feat.py]
  E --> E1[Data_Processed/quali_feat/
(event-only features)]

  E1 --> F[predictor_with_quali_feat.py]
  D2 --> F
  D3 --> F
  D1 --> F
  F --> G[Data_Processed/prediction_data/
win_probs_<GP>.csv,
position_prob_<GP>.csv,
final_prediction_<GP>.csv]

  D1 --> H[remaining_race_simulator.py]
  D2 --> H
  D3 --> H
  H --> I[Data_Processed/wdc_prediction/
win_probabilities_remaining_2025.csv,
finish_position_probabilities_2025.csv]
```

---

## ✅ Summary

The **F1 Predictor** combines:
- **Momentum (Rolling Form)**  
- **Raw speed (Pace Score)**  
- **Track history (Driver vs Track Score)**  
- **Reliability (DNF Chance)**  
- **Season context (Championship Rating)**  

into win probabilities (softmax with temperature), then uses **Monte Carlo** to produce **position probabilities** and **finishing orders**.  
Calibration and accuracy can be monitored with **Brier Score** and **Log Loss**.
