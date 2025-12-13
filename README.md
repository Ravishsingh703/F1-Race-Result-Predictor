# 📌 F1 Predictor

## Overview
An end-to-end Formula 1 race and World Drivers’ Championship (WDC) prediction system that combines driver form, qualifying pace, track history, reliability, and championship context into a structured statistical framework. The project produces race-level predictions (pre- and post-qualifying) and evaluates them against actual race outcomes via an interactive public website. Predictions are generated using Monte Carlo simulation to model uncertainty and variability inherent in Formula 1.

## 🌐 Live Website
The project is deployed as an interactive website that allows users to explore predicted vs actual race results, pre-qualifying and post-qualifying predictions (when available), and race-by-race performance from the United States Grand Prix onwards. The site dynamically loads race data from CSV files using a manifest-based structure, allowing new races to be added without modifying frontend code.

Live site: https://ravishsingh703.github.io/F1-Race-Result-Predictor/  
Repository: https://github.com/Ravishsingh703/F1-Race-Result-Predictor

## ⚙️ Methods & Formulas

### Rolling Form (Driver Momentum)
RollingForm_d = ( Σ (0.75)^(n-1-i) · Points_{d,i} ) / ( Σ (0.75)^(n-1-i) )

Scaled to 0–100 by dividing by 25 (maximum FIA race points). Recent races are weighted more heavily to capture current momentum while avoiding unfair penalties for drivers with limited historical data.

### Qualifying Gap to Pole
GapToPole_d = t_d_best − t_pole_best

Provides a direct, low-noise measure of raw car and driver pace, independent of race incidents and strategy effects.

### Pace Score
Grid-position-based exponential decay where P1 = 100 and the score approximately halves every seven grid positions. This formulation preserves front-running separation while compressing midfield and backmarker differences.

### Track Performance Score
Custom historical scoring table (1st = 40, 2nd = 35, 3rd = 28 … 20th = 1), averaged per driver-track combination. This encodes driver–track affinity without over-rewarding midfield consistency.

### DNF Probability
P(DNF_d,track) = DNFs_d,track / Starts_d,track

Introduces reliability risk into predictions, defaulting to zero where historical data is unavailable.

### Driver Score to Win Probability
Score_d = a·RollingForm + b·Pace + c·TrackScore + d·Survival + e·ChampRating

Scores are converted to probabilities using softmax with temperature scaling, ensuring all drivers retain non-zero probability while controlling prediction sharpness.

### Monte Carlo Simulation
Repeated randomized simulations generate full finishing order distributions, podium probabilities, and season-level outcomes. This captures irreducible uncertainty and rare events inherent to Formula 1.

## 📊 Statistics & Evaluation
The project applies softmax with temperature scaling, exponential decay weighting, and Monte Carlo simulation to ensure predictions remain interpretable, probabilistic, and robust. Evaluation is performed by comparing predicted finishing positions against actual race results, focusing on top-10 positional accuracy, podium hit rate, and average positional error. These comparisons are visualized on the deployed website.

## 📂 Project Structure
F1-Predictor/  
├── Code/ – data processing, feature engineering, and prediction logic  
├── docs/ – GitHub Pages website  
│   ├── index.html – homepage (predicted vs actual)  
│   ├── predictions.html – detailed prediction views  
│   ├── how-it-works.html – methodology explanation  
│   ├── assets/ – visual assets  
│   └── data/ – CSV files and manifest  
│       ├── actual-results/  
│       ├── pre-quali-predictions/  
│       └── post-quali-predictions/  
└── README.md

## ▶️ Running Order
fastf1.py processes raw race and qualifying data. rolling_form.py generates rolling form, reliability metrics, and pace features. quali_feat.py builds qualifying-based features. predictor_with_quali_results.py produces race-level predictions after qualifying. remaining_race_simulator.py runs Monte Carlo simulations for race and WDC outcome estimation.

## 🚀 How It Works
Raw FIA race and qualifying data is processed into structured performance features. These features are combined into weighted driver scores, converted into probabilistic outcomes, and simulated repeatedly using Monte Carlo methods. Predictions are then evaluated against real race results and presented through the interactive website.

## 🔮 Next Steps
The next project will focus on building a song recommendation system using user interaction data and machine learning techniques to generate personalized recommendations that improve over time.
