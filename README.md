# 📌 F1 Predictor

## Overview
An F1 World Drivers’ Championship (WDC) prediction model that combines **driver form, qualifying pace, track history, reliability, and championship standings** into a statistical framework. Predictions are simulated using **Monte Carlo methods** to generate both race-level and season-level outcomes.

---

## ⚙️ Methods & Formulas

### 1. Rolling Form (Driver Momentum)
```
RollingForm_d = ( Σ (0.75)^(n-1-i) * Points_{d,i} ) / ( Σ (0.75)^(n-1-i) )
```
Scaled 0–100 by dividing by 25 (max FIA points).  
**Why**: Recent races matter more. Prevents rookies (e.g., Piastri) from being unfairly penalized by lack of long history.

---

### 2. Qualifying Gap to Pole
```
GapToPole_d = t_d_best – t_pole_best
```
**Why**: Direct raw measure of car + driver pace, less noisy than race results.

---

### 3. Pace Score
Grid-position exponential decay:  
- P1 = 100  
- Score halves every *k* positions (half-life ~ 7).  

**Why**: Captures that small differences at the front matter more, while backmarkers converge.

---

### 4. Track Performance Score
Custom scoring table (1st=40, 2nd=35, 3rd=28 … 20th=1) averaged per track.  
**Why**: Encodes historical driver–track skill without overweighting midfield consistency.

---

### 5. DNF Chance
```
P(DNF_{d,track}) = DNFs_{d,track} / Starts_{d,track}
```
**Why**: Introduces reliability risk into predictions. Defaults to 0 if no history.

---

### 6. Driver Score → Win Probability
```
Score_d = a·RollingForm + b·Pace + c·TrackScore + d·Survival + e·ChampRating
```
Converted with **softmax** (temperature-scaled) → win probabilities.  
**Why**: Softmax allows all drivers to have >0 chances, while *temperature* controls uncertainty (higher temp = more randomness, lower temp = sharper prediction).

---

### 7. Monte Carlo Simulations
Randomized simulations model finishing orders and season outcomes.  
Captures the **inherent unpredictability** of Formula 1 by accounting for rare events and random variance.

---

## 📂 Project Structure

```
F1-Predictor/
│
├── Code/
│   ├── fastf1.py                   # Cleans & processes raw race/quali data
│   ├── rolling_form.py             # Builds rolling form, DNF chance, pace scores
│   ├── quali_feat.py               # Features for quali-based predictions
│   ├── predictor_with_quali_results.py # Predictor using quali data
│   ├── remaining_race_simulator.py # Final simulator for WDC outcomes
│
├── Data/
│   └── fastf1/                     # Raw race + quali data (2018–2025)
│
├── Data_Processed/
│   ├── fastf1/                     # Cleaned data (per year)
│   ├── driver_stats_over_years/    # DNF, podium rate, pit efficiency, rolling form
│   ├── track_scores/               # Driver vs. track performance files
│   ├── quali_feat/                 # Used in quali predictor
│   ├── prediction_data/            # Intermediate pace, DNF, win prob files
│   └── wdc_prediction/             # Final probabilities & predictions
│
└── README.md
```

---

## ▶️ Running Order

1. **fastf1.py** → process raw data → `Data_Processed/fastf1/`  
2. **rolling_form.py** → generate rolling form, DNF chance, pace scores → saved under `driver_stats_over_years/`  
3. **quali_feat.py** → build quali features  
4. **predictor_with_quali_results.py** → compute race-level predictions after quali  
5. **remaining_race_simulator.py** → run Monte Carlo sims → WDC predictions in `wdc_prediction/`

---

## 📊 Statistics & Evaluation

This project applies several statistical models and concepts to ensure predictions are both realistic and robust:

- **Softmax with Temperature Scaling**  
  Converts driver scores into win probabilities.  
  - Low temperature (<1): sharper, more confident predictions.  
  - High temperature (>1): flatter, more uncertain predictions.

- **Exponential Decay Weighting**  
  Used in Rolling Form and Pace Score. Ensures **recent performance weighs more heavily**, while older results fade smoothly.  

- **Monte Carlo Simulation**  
  Randomized simulations are run repeatedly to model finishing orders and season outcomes.  
  Captures the **inherent unpredictability** of F1 by accounting for rare events and random variance.

---

## 🚀 How It Works
- **Input**: Raw FIA race + quali timing data (2018–2025).  
- **Processing**: Extract features (form, pace, track history, reliability).  
- **Prediction**: Weighted driver strength → win probability.  
- **Simulation**: Monte Carlo → race distributions → season outcome.  

