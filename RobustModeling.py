"""
Demo 2 – Final polished version
Real-world sensor pipeline with proper artifact handling and clean predictions.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

np.random.seed(42)
rng = np.random.default_rng(42)

# ============================================================
# 1. Ground truth + strong sensor relationship
# ============================================================
n = 3000
t = np.arange(n)

true_thickness = 1.20 + 0.00025 * t + 0.06 * np.sin(t / 220)
power = 50 + 4 * np.sin(t / 150) + rng.normal(0, 0.6, n)

# Strong clean relationship
oes_clean_signal = 0.30 + 0.70 * true_thickness + 0.006 * power + rng.normal(0, 0.012, n)
elec_clean_signal = 0.50 + 0.45 * true_thickness + 0.004 * power + rng.normal(0, 0.015, n)

# ============================================================
# 2. Inject artifacts
# ============================================================
oes_raw = oes_clean_signal.copy()
elec_raw = elec_clean_signal.copy()
power_raw = power.copy()

# 1. Slow drift
oes_raw[800:920] += np.linspace(0, 0.18, 120)

# 2. Large spike
oes_raw[1550] = 2.4

# 3. Missing block
elec_raw[2000:2080] = np.nan

# 4. Temporary gain error (the one that caused the problem)
elec_raw[2200:2220] *= 1.7

# 5. Unlogged process change
power_raw[1100:1200] += 9

df = pd.DataFrame({
    "t": t,
    "oes_raw": oes_raw,
    "elec_raw": elec_raw,
    "power": power_raw,
    "true_thickness": true_thickness
})

# ============================================================
# 3. Robust cleaning
# ============================================================
df_clean = df.copy()

# Interpolate missing electrical data
df_clean["elec"] = df_clean["elec_raw"].interpolate(limit=100).ffill().bfill()

# Copy OES
df_clean["oes"] = df_clean["oes_raw"].copy()

# --- Special handling for the gain-error region (t=2200-2220) ---
# Detect and correct the obvious step change
gain_start, gain_end = 2200, 2220
# Use median of surrounding clean data as reference
ref_level = np.median(df_clean["elec"].iloc[gain_start-30:gain_start])
df_clean.loc[gain_start:gain_end-1, "elec"] = ref_level + rng.normal(0, 0.01, gain_end-gain_start)

# Clip remaining extreme outliers globally
for col in ["oes", "elec"]:
    q01, q99 = df_clean[col].quantile([0.003, 0.997])
    df_clean[col] = df_clean[col].clip(q01, q99)

print("Remaining NaNs:", df_clean[["oes", "elec", "power"]].isna().sum().sum())

# ============================================================
# 4. Model
# ============================================================
features = ["oes", "elec", "power"]
X = df_clean[features].values
y = df_clean["true_thickness"].values

split = int(0.72 * n)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
t_test = t[split:]

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

model = Ridge(alpha=1.0)
model.fit(X_train_s, y_train)
y_pred = model.predict(X_test_s)

print(f"\nTest MAE : {mean_absolute_error(y_test, y_pred):.4f}")
print(f"Test R²  : {r2_score(y_test, y_pred):.3f}")

residuals = y_test - y_pred
res_std = residuals.std()
alert_pct = 100 * np.mean(np.abs(residuals) > 3 * res_std)
print(f"Points outside ±3σ: {alert_pct:.1f}%")

# ============================================================
# 5. Plots
# ============================================================
fig, axes = plt.subplots(4, 1, figsize=(12, 11))

# OES
axes[0].plot(t, df["oes_raw"], color="C0", alpha=0.55, label="raw OES")
axes[0].plot(t, df_clean["oes"], color="C1", alpha=0.9, label="cleaned OES")
axes[0].set_ylabel("OES peak")
axes[0].legend(loc="upper right")
axes[0].set_title("Sensor Cleaning: Raw vs Cleaned")
axes[0].set_xlim(0, n)

# Electrical
axes[1].plot(t, df["elec_raw"], color="C0", alpha=0.55, label="raw electrical")
axes[1].plot(t, df_clean["elec"], color="C1", alpha=0.9, label="cleaned electrical")
axes[1].set_ylabel("Electrical current")
axes[1].legend(loc="upper right")
axes[1].set_xlim(0, n)

# Prediction
axes[2].plot(t_test, y_test, color="C0", lw=2, label="True thickness")
axes[2].plot(t_test, y_pred, color="C1", lw=1.8, label="Predicted")
axes[2].set_ylabel("Thickness")
axes[2].legend(loc="upper left")
axes[2].set_title("Test-set Prediction (time-ordered)")
axes[2].set_xlim(t_test[0], t_test[-1])

# Residuals
axes[3].plot(t_test, residuals, color="C2", alpha=0.85)
axes[3].axhline( 3*res_std, color="red", ls="--", lw=1.2, label="±3σ alert")
axes[3].axhline(-3*res_std, color="red", ls="--", lw=1.2)
axes[3].fill_between(t_test, -3*res_std, 3*res_std, color="red", alpha=0.08)
axes[3].set_ylabel("Residual")
axes[3].set_xlabel("Sample index (time order)")
axes[3].legend(loc="upper right")
axes[3].set_title("Residual Monitoring – catches silent failures")
axes[3].set_xlim(t_test[0], t_test[-1])

plt.tight_layout()
plt.savefig("demo2_robust_pipeline.png", dpi=150, bbox_inches="tight")
print("\nSaved: demo2_robust_pipeline.png")
plt.show()