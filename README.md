# Robust Real-World Sensor Pipeline

This demo shows a complete, practical pipeline for working with noisy, real-world plasma sensor data. It focuses on the kinds of problems that appear in industrial settings and how to handle them before modeling.

It is designed to illustrate skills directly relevant to the SirenOpt Senior Scientist role:
- Detecting and cleaning sensor artifacts
- Building trustworthy features from imperfect data
- Training a robust model
- Monitoring predictions for silent failures

---

## Overview

We simulate a realistic sensor stream that includes:

| Artifact                    | Description                              |
|----------------------------|------------------------------------------|
| Slow drift                 | Gradual baseline shift in the OES signal |
| Spike                      | Sudden large outlier                     |
| Missing block              | Short gap in the electrical signal       |
| Temporary gain error       | Abrupt scaling of the electrical signal  |
| Unlogged process change    | Step change in power                     |

The pipeline then:

1. Detects these problems
2. Cleans the signals
3. Trains a model to predict material thickness
4. Monitors residuals to catch bad predictions

---

## What is OES?

**OES = Optical Emission Spectroscopy** (or Optical Emission Spectrum).

When cold atmospheric plasma interacts with a material, it emits light at characteristic wavelengths. An optical emission spectrometer records this light. The resulting spectrum (or features extracted from it) contains information about the plasma state and the material properties.

In this demo, the `oes` feature is a simplified stand-in for a key spectral indicator.

---

## Model: Ridge Regression

We use **Ridge Regression**, a linear model with L2 regularization:

\[
L(w) = \| y - Xw \|^2_2 + \alpha \| w \|^2_2
\]

### Why Ridge?

- The relationship between the cleaned sensors and thickness is strong and approximately linear
- Ridge is stable when features are correlated
- Coefficients are easy to interpret
- It extrapolates smoothly when the test data continues a trend seen during training

Example coefficients from a typical run (after standardization):

```text
OES          : +0.067
Electrical   : +0.090
Power        : -0.021
Intercept    : +1.482
