# STAC WARS - Team Vision

Solutions for the STAC WARS Data Analysis, Astronomy, and Rocketry challenges.

## Team Members
- **Kanishk** (Team Leader)
- **Abhijeet Kumar**
- **Priyanshu**
- **Aditya Kumar**

---

## 1. PS1: OpenRocket Extraterrestrial Flight Design (Operation Ember Ascent)

Designed a two-stage sounding rocket targeting a 4 km apogee under Mustafar's extreme atmospheric conditions (700 K temperature, 75 kPa surface pressure, $0.374 \text{ kg/m}^3$ air density, and $6.50 \text{ m/s}^2$ surface gravity).

### Key Design Parameters
- **Apogee & Payload:** Hit 4,000 m apogee while deploying a 1.0 kg scientific payload.
- **Constraints Met:** Total rocket length $\le 220 \text{ cm}$, total motor impulse $\le 2800 \text{ N}\cdot\text{s}$, and descent rate kept between 4–8 m/s.

---

## 2. PS2: Solar Radiation Data Analysis & Prediction

Analyzed historical solar logs from the Voyage-123 probe to identify solar flare patterns, aggregate daily statistics, and build a radiation prediction model.

### Technical Steps
- **Data Visualization & Insights:** Evaluated variations of solar radiation against time, temperature, humidity, and wind speed to highlight correlations and flag sensor outliers.
- **Day-Wise Aggregations:** Converted timestamps to datetime objects, grouped records by date, and extracted daily peak/average radiation along with peak timing relative to sunrise/sunset.
- **Regression Modeling:** Built a model using temporal and environmental features to predict solar radiation, evaluated using RMSE and $R^2$ scores.

---

## 3. PS3: Exoplanet Hunters (Operation Dim Star)

Analyzed time-series stellar brightness logs for STAC-1, STAC-2, and STAC-3 to detect transiting exoplanets, determine their physical scale, map their orbits, and evaluate habitability candidates.

### Key Methodology
1. **Transit Detection:** Calculated fractional brightness dip ($\delta = \frac{F_{\text{normal}} - F_{\text{transit}}}{F_{\text{normal}}}$) and identified transit periodicity for host stars.
2. **Radius Estimation:** Estimated planetary radius $R_p$ using star radii $R_s$ and depth $\delta \approx (R_p / R_s)^2$, classifying candidate sizes.
3. **Orbital Distances:** Calculated orbital distance $a$ in AU and kilometers using Kepler's Third Law ($P^2 = a^3 / M$).
4. **Habitability Evaluation:** Compared orbital distances against stellar habitable zones ($d_{HZ} \approx \sqrt{L / L_\odot}$) to identify the best target for life-support potential.

---

## Setup & Running Code

### Running Python Notebooks
```bash
# Clone repository
git clone [https://github.com/YourUsername/TeamVision_STAC_WARS.git](https://github.com/YourUsername/TeamVision_STAC_WARS.git)
cd TeamVision_STAC_WARS

# Install required dependencies
pip install numpy pandas matplotlib scikit-learn jupyter

# Launch notebooks
jupyter notebook
