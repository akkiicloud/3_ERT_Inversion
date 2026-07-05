# 3_ERT_Inversion

## Overview

This repository contains the Python scripts developed for the inversion of Electrical Resistivity Tomography (ERT) data using **PyGIMLi**. The scripts implement different inversion strategies, including classical smoothness-constrained inversion, geostatistical regularization, inversion with alternative starting models, and cross-array forward modelling.

> **Important**
>
> The inversion parameters adopted in these scripts (e.g., regularization strength, correlation lengths, error models, vertical weighting (*zWeight*), and other inversion settings) were selected specifically for the Wendelsheim and Pfäffingen datasets investigated in this study. These parameters should **not** be directly transferred to other ERT datasets. Instead, they should be adapted according to the data quality, acquisition geometry, geological setting, and inversion objectives of the study area.

---

# Repository Contents

## 1. `Inversion_01_pg_err_thesis_model.py`

This script performs the **classical smoothness-constrained inversion** using PyGIMLi.

The workflow includes:

- importing the ERT dataset,
- constructing the elevation profile,
- creating the error model,
- generating the inversion mesh using the default PyGIMLi mesh,
- applying a homogeneous starting model, and
- performing the smoothness-constrained inversion.

---

## 2. `Inversion_04_pg_err_thesis_correlation length.py`

This script performs **geostatistical-based inversion**.

The workflow includes:

- constructing the elevation profile,
- creating the error model,
- generating the default PyGIMLi inversion mesh,
- applying a homogeneous starting model,
- defining horizontal and vertical correlation lengths, and
- performing the inversion using geostatistical regularization.

---

## 3. `Inversion_09_force_inversion_thesis.py`

This script performs **classical smoothness-constrained inversion using different starting models**.

Unlike the default homogeneous initialization, alternative starting resistivity models are prescribed to investigate their influence on the inversion results.

The workflow includes:

- elevation profile construction,
- error model generation,
- mesh generation,
- assignment of user-defined starting models, and
- inversion.

---

## 4. `Inversion_11_pg_err_MG_DD_forward_response.py`

This script performs a **cross-array forward modelling analysis**.

The resistivity model obtained from the **Dipole–Dipole (DD)** inversion is used to forward simulate the apparent resistivity response for the **Multi-gradient (MG)** electrode configuration.

The resulting synthetic MG pseudosection is then compared with the measured MG field pseudosection to evaluate the consistency between the two acquisition arrays and to assess whether the interpreted subsurface model explains the observations from both configurations.

---

## 5. `Inversion_12_pg_err_DD_MG_forward_other_way_response.py`

This script performs the reverse **cross-array forward modelling analysis**.

The resistivity model obtained from the **Multi-gradient (MG)** inversion is used to simulate the apparent resistivity response for the **Dipole–Dipole (DD)** configuration.

The synthetic DD pseudosection is subsequently compared with the measured DD pseudosection to evaluate the agreement between the inversion model and the independent array response.

---
