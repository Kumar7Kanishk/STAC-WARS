"""
STAC WARS - PS2: Solar radiation analysis and prediction
Parts A (plots), B (inferences come from the printed numbers), C (day-wise report)
and the Bonus (regression model).

Run:  python solar_analysis.py
Needs SolarPrediction.csv in the same folder. Plots go to ./figs, numbers to results.txt
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

sns.set_theme(style="whitegrid")
os.makedirs("figs", exist_ok=True)
out = open("results.txt", "w")


def say(*args):
    """print to screen and also save in results.txt (so the report numbers match)"""
    print(*args)
    print(*args, file=out)


# ---------------------------------------------------------------- loading
df = pd.read_csv("SolarPrediction.csv")

# the date column is called "Data" and has a useless 12:00:00 AM stuck on it,
# so take only the date part and glue the real clock time on
df["datetime"] = pd.to_datetime(df["Data"].str.split(" ").str[0] + " " + df["Time"])
df = df.sort_values("datetime").reset_index(drop=True)   # file is newest-first
df = df.rename(columns={"WindDirection(Degrees)": "WindDir", "Speed": "WindSpeed"})

df["date"] = df["datetime"].dt.date
df["hour"] = df["datetime"].dt.hour
df["month"] = df["datetime"].dt.month
df["hour_dec"] = df["datetime"].dt.hour + df["datetime"].dt.minute / 60

# sunrise/sunset are only clock times, make them proper datetimes on the same day
day_str = df["datetime"].dt.strftime("%Y-%m-%d")
df["sunrise"] = pd.to_datetime(day_str + " " + df["TimeSunRise"])
df["sunset"] = pd.to_datetime(day_str + " " + df["TimeSunSet"])

say("rows:", len(df), "| missing values:", int(df.isna().sum().sum()))
say("period:", df["datetime"].min(), "->", df["datetime"].max())
say("distinct days:", df["date"].nunique(), "(Sep 1 - Dec 31 would be 122)")
gaps = df["datetime"].diff()
say("biggest gap between readings:", gaps.max(), "ending", df.loc[gaps.idxmax(), "datetime"])
say("humidity readings above 100%:", int((df["Humidity"] > 100).sum()))
say()

num_cols = ["Radiation", "Temperature", "Pressure", "Humidity", "WindDir", "WindSpeed"]

# ================================================================ PART A
# 1. radiation over time ------------------------------------------------
daily = df.groupby("date")["Radiation"].agg(["mean", "max"])
daily.index = pd.to_datetime(daily.index)

fig, ax = plt.subplots(2, 1, figsize=(11, 8))
ax[0].plot(daily.index, daily["max"], label="daily max", color="tab:red", lw=1.2)
ax[0].plot(daily.index, daily["mean"], label="daily mean", color="tab:blue", lw=1.2)
ax[0].set_title("Solar radiation over the whole period (Sep - Dec 2016)")
ax[0].set_ylabel("Radiation (W/m²)")
ax[0].legend()

one_day = df[df["date"] == pd.to_datetime("2016-09-04").date()]
ax[1].plot(one_day["datetime"], one_day["Radiation"], color="orange")
ax[1].set_title("A single day close up: 4 Sep 2016 (the day with the highest reading)")
ax[1].set_ylabel("Radiation (W/m²)")
ax[1].set_xlabel("time of day")
plt.tight_layout()
plt.savefig("figs/01_radiation_vs_time.png", dpi=130)
plt.close()

# 2. average radiation by hour of day ----------------------------------
hourly = df.groupby("hour")["Radiation"].mean()
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.bar(hourly.index, hourly.values, color="goldenrod")
ax.set_xticks(range(0, 24))
ax.set_xlabel("hour of day")
ax.set_ylabel("mean radiation (W/m²)")
ax.set_title("Average radiation by hour of day")
plt.tight_layout()
plt.savefig("figs/02_radiation_by_hour.png", dpi=130)
plt.close()
say("mean radiation by hour:")
say(hourly.round(1).to_string())
say("peak hour:", hourly.idxmax(), "->", round(hourly.max(), 1))
say()

# 3. radiation vs other parameters --------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for a, col in zip(axes.flat, ["Temperature", "Humidity", "Pressure", "WindSpeed", "WindDir"]):
    a.scatter(df[col], df["Radiation"], s=3, alpha=0.25)
    a.set_xlabel(col)
    a.set_ylabel("Radiation")
    r = df["Radiation"].corr(df[col])
    a.set_title(f"Radiation vs {col}  (r = {r:.2f})")
axes.flat[-1].axis("off")
plt.tight_layout()
plt.savefig("figs/03_radiation_vs_parameters.png", dpi=130)
plt.close()

corr = df[num_cols].corr()
say("correlation with Radiation (all hours):")
say(corr["Radiation"].drop("Radiation").round(3).to_string())

# night readings sit at ~1.2 and dilute everything, so also look at daytime only
day_only = df[(df["datetime"] >= df["sunrise"]) & (df["datetime"] <= df["sunset"])]
say("\ncorrelation with Radiation (daytime only):")
say(day_only[num_cols].corr()["Radiation"].drop("Radiation").round(3).to_string())
say()

# 4. heatmap ----------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5.5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Correlation matrix")
plt.tight_layout()
plt.savefig("figs/04_correlation_heatmap.png", dpi=130)
plt.close()

# 5. radiation vs temperature, coloured by humidity (daytime) -------------
fig, ax = plt.subplots(figsize=(8, 5.5))
sc = ax.scatter(day_only["Temperature"], day_only["Radiation"], c=day_only["Humidity"],
                cmap="viridis_r", s=6, alpha=0.5)
plt.colorbar(sc, label="Humidity (%)")
ax.set_xlabel("Temperature (°F)")
ax.set_ylabel("Radiation (W/m²)")
ax.set_title("Daytime radiation vs temperature, coloured by humidity")
plt.tight_layout()
plt.savefig("figs/05_temp_humidity_radiation.png", dpi=130)
plt.close()

hum_bins = pd.cut(day_only["Humidity"], [0, 40, 60, 80, 100, 110])
say("daytime mean radiation by humidity band:")
say(day_only.groupby(hum_bins, observed=True)["Radiation"].agg(["mean", "count"]).round(1).to_string())
say()

# 6. outliers -----------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
sns.boxplot(x="hour", y="Radiation", data=df, ax=ax[0], fliersize=2)
ax[0].set_title("Radiation distribution per hour (dots = outliers)")
sns.boxplot(y=df["WindSpeed"], ax=ax[1], fliersize=2)
ax[1].set_title("Wind speed")
plt.tight_layout()
plt.savefig("figs/06_outliers.png", dpi=130)
plt.close()

q1, q3 = day_only["Radiation"].quantile([0.25, 0.75])
say("daytime radiation IQR fences:", round(q1 - 1.5 * (q3 - q1), 1), "to", round(q3 + 1.5 * (q3 - q1), 1))
say("readings above 1200 W/m²:", int((df["Radiation"] > 1200).sum()))
say(df.nlargest(8, "Radiation")[["datetime", "Radiation", "Temperature", "Humidity", "WindSpeed"]].to_string(index=False))
ws_hi = df["WindSpeed"].quantile(0.99)
say("wind speed 99th percentile:", ws_hi, "| max:", df["WindSpeed"].max())
# sudden drops in the middle of the day = clouds, look at how often they happen
midday = df[(df["hour"] >= 10) & (df["hour"] <= 14)]
say("midday (10-14h) readings under 200 W/m²:", int((midday["Radiation"] < 200).sum()),
    "out of", len(midday))
say()

# 7. BONUS plot: daily profile per month + temperature lag -----------------
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
names = {9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
for m, g in day_only.groupby("month"):
    prof = g.groupby("hour")["Radiation"].mean()
    ax[0].plot(prof.index, prof.values, marker="o", label=names[m])
ax[0].set_title("Average daytime profile for each month")
ax[0].set_xlabel("hour of day")
ax[0].set_ylabel("mean radiation (W/m²)")
ax[0].legend()

# hourly bins hide the lag (both peak in the 12h bin), so use half-hour bins
half = (df["datetime"].dt.hour * 2 + df["datetime"].dt.minute // 30) / 2
by_hour = df.groupby(half)[["Radiation", "Temperature"]].mean()
by_hour = by_hour.loc[6:19]
scaled = (by_hour - by_hour.min()) / (by_hour.max() - by_hour.min())
ax[1].plot(scaled.index, scaled["Radiation"], label="radiation", color="orange", marker="o", ms=3)
ax[1].plot(scaled.index, scaled["Temperature"], label="temperature", color="red", marker="s", ms=3)
ax[1].set_title("Temperature peaks about an hour after radiation (rescaled 0-1)")
ax[1].set_xlabel("hour of day (half-hour bins)")
ax[1].legend()
plt.tight_layout()
plt.savefig("figs/07_bonus_monthly_and_lag.png", dpi=130)
plt.close()

monthly = day_only.groupby("month")["Radiation"].mean().round(1)
say("daytime mean radiation per month:", monthly.to_dict())
say("time of max mean radiation:", by_hour["Radiation"].idxmax(),
    "h | time of max mean temperature:", by_hour["Temperature"].idxmax(), "h")
# how humid was the record-breaking day? (spike on a cloudy day = suspicious)
d4 = df[df["date"] == pd.to_datetime("2016-09-04").date()]
say("4 Sep: mean humidity", round(d4["Humidity"].mean(), 1), "| mean radiation",
    round(d4["Radiation"].mean(), 1), "| overall mean", round(df["Radiation"].mean(), 1))
say()

# ================================================================ PART C
say("=" * 60)
say("PART C - day wise report")
say("=" * 60)

report = df.groupby("date").agg(
    max_radiation=("Radiation", "max"),
    avg_radiation=("Radiation", "mean"),
    readings=("Radiation", "size"),
)
report["avg_radiation"] = report["avg_radiation"].round(2)

# which row is the all-time peak
peak_row = df.loc[df["Radiation"].idxmax()]
peak_date = peak_row["date"]
say("date with highest radiation:", peak_date)
say("max radiation:", peak_row["Radiation"], "W/m²")
say("time of the peak:", peak_row["datetime"].strftime("%H:%M:%S"))
say("temperature then:", peak_row["Temperature"], "°F")
say("humidity then:", peak_row["Humidity"], "%")
say("wind speed then:", peak_row["WindSpeed"], "mph")

sunrise, sunset = peak_row["sunrise"], peak_row["sunset"]
after_rise = peak_row["datetime"] - sunrise
daylen = sunset - sunrise
say("sunrise:", sunrise.strftime("%H:%M"), "| sunset:", sunset.strftime("%H:%M"),
    "| day length:", daylen)
say("peak came", after_rise, "after sunrise  (~", round(after_rise.total_seconds() / 3600, 2), "hours)")
say("that is", round(100 * after_rise / daylen, 1), "% of the way through the day")
say("time left before sunset:", sunset - peak_row["datetime"])

# do the same for every day so we can see if the peak is normally near solar noon
idx = df.groupby("date")["Radiation"].idxmax()
peaks = df.loc[idx]
hrs_after = (peaks["datetime"] - peaks["sunrise"]).dt.total_seconds() / 3600
say("\nacross all days, peak occurs on average", round(hrs_after.mean(), 2),
    "h after sunrise (median", round(hrs_after.median(), 2), ")")

top5 = report.sort_values("avg_radiation", ascending=False).head(5)
say("\ntop 5 days by average radiation:")
say(top5.to_string())
say("\ntop 5 days by max radiation (for comparison):")
say(report.sort_values("max_radiation", ascending=False).head(5).to_string())

report.to_csv("daily_solar_report.csv")
say("\nwhole daily report saved to daily_solar_report.csv", report.shape)
say()

# ================================================================ BONUS ML
say("=" * 60)
say("BONUS - predicting radiation")
say("=" * 60)

# features: weather readings + time info. I use minutes since sunrise and
# minutes until sunset because they carry the 'where are we in the day' info.
# I leave out UNIXTime (it would just let the model memorise dates).
ml = df.copy()
ml["day_of_year"] = ml["datetime"].dt.dayofyear
ml["mins_since_sunrise"] = (ml["datetime"] - ml["sunrise"]).dt.total_seconds() / 60
ml["mins_to_sunset"] = (ml["sunset"] - ml["datetime"]).dt.total_seconds() / 60
ml["day_length_min"] = (ml["sunset"] - ml["sunrise"]).dt.total_seconds() / 60

features = ["Temperature", "Pressure", "Humidity", "WindDir", "WindSpeed",
            "hour_dec", "day_of_year", "mins_since_sunrise", "mins_to_sunset"]
X = ml[features]
y = ml["Radiation"]

# random split like the task asks. (readings 5 min apart are very similar so
# a random split is a bit optimistic - I also check a split by date below)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def scores(name, model, Xtr, Xte, ytr, yte):
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    rmse = np.sqrt(mean_squared_error(yte, pred))
    r2 = r2_score(yte, pred)
    say(f"{name:<28} RMSE = {rmse:7.2f}   R2 = {r2:.4f}")
    return model, pred, rmse, r2


lin, _, _, _ = scores("Linear regression", LinearRegression(), X_train, X_test, y_train, y_test)
rf, rf_pred, _, _ = scores("Random forest (200 trees)",
                           RandomForestRegressor(n_estimators=200, min_samples_leaf=2,
                                                 n_jobs=-1, random_state=42),
                           X_train, X_test, y_train, y_test)

# stricter check: train on the first 80% of the days, test on the last 20%
dates = np.sort(ml["date"].unique())
cut = dates[int(len(dates) * 0.8)]
tr = ml["date"] < cut
rf2 = RandomForestRegressor(n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=42)
say("\nsplit by date (train before", cut, ", test after):")
_, _, _, _ = scores("Random forest, date split", rf2, X[tr], X[~tr], y[tr], y[~tr])

# feature importance
imp = pd.Series(rf.feature_importances_, index=features).sort_values()
say("\nrandom forest feature importance:")
say(imp.sort_values(ascending=False).round(4).to_string())

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
imp.plot.barh(ax=ax[0], color="teal")
ax[0].set_title("Random forest feature importance")
ax[1].scatter(y_test, rf_pred, s=3, alpha=0.3)
lim = [0, y_test.max()]
ax[1].plot(lim, lim, "r--", lw=1)
ax[1].set_xlabel("actual radiation")
ax[1].set_ylabel("predicted radiation")
ax[1].set_title("Predicted vs actual (test set)")
plt.tight_layout()
plt.savefig("figs/08_model_results.png", dpi=130)
plt.close()

# what happens if we drop the time features? just weather
weather = ["Temperature", "Pressure", "Humidity", "WindDir", "WindSpeed"]
rf3 = RandomForestRegressor(n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=42)
say("\nweather columns only (no time features):")
scores("Random forest, weather only", rf3, X_train[weather], X_test[weather], y_train, y_test)

out.close()
print("\ndone - plots in ./figs, numbers in results.txt")
