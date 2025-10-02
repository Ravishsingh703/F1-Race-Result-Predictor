# F1-Project-Folder
# 🏎️ Formula 1 World Drivers’ Championship Predictor

This project implements a **statistical and simulation-based predictor** for the Formula 1 World Drivers’ Championship (WDC). It combines race history, driver form, track performance, and probabilistic modeling to forecast remaining race outcomes and the championship standings.

---

## 📊 Methods & Formulas

The predictor uses a combination of **statistical scoring functions** and **Monte Carlo simulation**:

### 1. Rolling Form (Driver Momentum)
Exponential decay weighted average of FIA points scored:
\[
\text{RollingForm}_d = \frac{\sum_{i=1}^n w_i \cdot \text{Points}_{d,i}}{\sum_{i=1}^n w_i}, \quad w_i = \text{DECAY}^{(n-1-i)}
\]
Scaled to 0–100.

### 2. Qualifying Gap to Pole
Difference between driver’s best quali time and pole time:
\[
\text{GapToPole}_d = t_d^{best} - t_{pole}^{best}
\]

### 3. Pace Score
Exponential decay function of quali gap (or grid position fallback):
\[
\text{Pace100}_d = \max \left( S_{min}, \, 100 \cdot e^{-\frac{\min(g_d, g_{cap})}{\tau}} \right)
\]

### 4. Track Performance Score
Historical performance by driver on each circuit (2018–2025), using custom position → score table, averaged per track.

### 5. DNF Chance
Simple empirical probability:
\[
P(\text{DNF}_{d,track}) = \frac{\text{DNFs}_{d,track}}{\text{Starts}_{d,track}}
\]

### 6. Championship Rating
Exponential rank decay based on total championship points:
\[
\text{Champ01}_d = \max\left(S_{min}, 0.5^{\frac{(\text{Rank}_d - 1)}{H_{champ}}}\right)
\]

### 7. Driver Strength & Win Probability
Combined score:
\[
S_{d,track} = a \cdot \text{Pace}_d + b \cdot \text{TrackScore}_d + c \cdot \text{RollingForm}_d + d \cdot (1-P(\text{DNF})) + e \cdot \text{Champ01}_d
\]
Converted via softmax:
\[
P(\text{Win}_{d,track}) = \frac{e^{S_{d,track}/T}}{\sum_j e^{S_{j,track}/T}}
\]

### 8. Monte Carlo Simulation
- **Stage 1 (100k sims):** Plackett–Luce sampling of finish orders, → position probabilities.  
- **Stage 2 (10M sims):** Recursive draws to build full finishing orders, → final tables.  

### 9. Recursive Update
After each simulated race:
- Update **Rolling Form** with expected points.  
- Update **Championship Rating** with cumulative points.  
- Predict the next race based on the updated state.

---

## 📂 Project Structure

```
Data_Processed/
  fastf1/2025/                  # per-race feature files
  driver_stats_over_years/      # rolling form seeds
  wdc_prediction/               # baseline predictors
  new_model_test/               # recursive predictor outputs

Scripts:
  fastf1.py
  rolling_form.py
  quali_feat.py
  predictor_with_quali_results.py
  remaining_race_simulator.py
```

---

## ⚙️ How to Run

1. **Preprocess Data**  
   Ensure `fastf1/2018–2025` features are processed into CSVs. Run:
   ```bash
   python fastf1.py
   ```

2. **Compute Rolling Features**  
   Generates rolling form, pace, and DNF stats:
   ```bash
   python rolling_form.py
   ```

3. **Add Quali Features (Optional)**  
   For predictions after qualifying:
   ```bash
   python quali_feat.py
   ```

4. **Predict Remaining Season**  
   Run main simulator (baseline or recursive):
   ```bash
   python remaining_race_simulator.py
   ```

This generates:
- `win_probabilities_remaining_2025.csv`
- `finish_position_probabilities_2025.csv`

---

## 🔑 Key Statistical Ideas
- **Exponential decay weighting** → emphasizes recent performance.  
- **Softmax with temperature** → transforms driver scores into win probabilities.  
- **Monte Carlo sampling** → captures uncertainty in race results.  
- **Recursive updating** → integrates predictions dynamically, using new race outcomes.  

---

## 🚀 Why This Matters
- Combines **sports analytics** with **probabilistic forecasting**.  
- Applies **Bayesian-style updating** to account for rolling form and championship dynamics.  
- Provides fans, analysts, and developers with **data-driven projections** of the F1 season.  
