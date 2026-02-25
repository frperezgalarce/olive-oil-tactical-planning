# Olive model parameter update (JSONC) 

This document provides the active parameter set used by your pipeline, in JSON-with-comments (JSONC).
Each parameter keeps the original value and clarifies whether it is (i) used by the Moriondo-style process model (phenology–canopy–water–yield) or (ii) used by the simplified yield proxy in `run_simple_model()` (CRUE/CHI/CCh-based).

> **Important consistency note (radiation driver):**
>
> * In the **Moriondo-style module**, biomass uses `DM_pot = Int_OT × RAD × RUE` (your `eq4_dm_pot`).
> * In the **simplified yield proxy** (`run_simple_model`), daily yield uses `daily_yield = CCh × EF_t × min(FHeat, f_water)` where `CCh = CRUE × CHI`.
>   These two pathways are **not interchangeable** unless you explicitly connect them in the code.

---

```jsonc
{
  // =========================
  // Grove architecture
  // =========================
  "PlantD": 1600.0,                // original: 1600.0 (plants/ha) | pipeline: used in eq6_k_prime()
  "PlantA": 6.25,                  // original: 6.25 (m²/plant)    | derived: 10000/PlantD (geometry helper; not directly used if LAI_ini is set)
  "LAD": 2.0,                      // original: 2.0                | pipeline: used in eq6_k_prime(), eq7_v_from_lai()

  // =========================
  // Phenology (paper-style)
  // =========================
  "phenology_chill_model": "unichill",   // original: "unichill"  | pipeline: uses eq2_unichill()
  "phenology_forcing_model": "logistic", // original: "logistic"  | pipeline: uses eq3_forcing_logistic()

  "tb_budbreak": 8.5,              // original: 8.5 (°C)          | pipeline: budbreak GDH base temp (kept as in paper calibration narrative) :contentReference[oaicite:0]{index=0}

  "chill_a": 0.01,                 // original: 0.01              | pipeline: eq2_unichill()
  "chill_b": -0.5,                 // original: -0.5              | pipeline: eq2_unichill()
  "chill_c": 7.2,                  // original: 7.2               | pipeline: eq2_unichill()
  "Ccrit": 350.0,                  // original: 350.0 (CU)         | pipeline: chill threshold (cultivar/site calibration placeholder)

  "forcing_d": 0.5,                // original: 0.5               | pipeline: eq3_forcing_logistic()
  "forcing_e": 9.0,                // original: 9.0               | pipeline: eq3_forcing_logistic()
  "FcritFlo": 450.0,               // original: 450.0 (FU)         | pipeline: forcing threshold (cultivar/site calibration placeholder)

  // Optional alternative phenology switches (implemented in code but not default)
  "T_opt_chill_alt": 7.2,          // original: 7.2 (°C)           | pipeline: alt_chill_triangular() only if selected
  "T_max_chill_alt": 16.0,         // original: 16.0 (°C)          | pipeline: alt_chill_triangular() only if selected
  "Tb_forcing_alt": 9.0,           // original: 9.0 (°C)           | pipeline: alt_forcing_gdd() only if selected

  // =========================
  // Canopy / biomass (Moriondo-style path)
  // =========================
  "RUE_ol": 0.98,                  // original: 0.98               | pipeline default for eq4_dm_pot(); keep unless you explicitly rewire to CRUE
  "SLA_ol": 5.2,                   // original: 5.2 (m²/kg)         | pipeline: used in eq8_lai_inc_pot()
  "PClf_pot_base": 0.18,           // original: 0.18                | pipeline: leaf partition baseline for eq8_lai_inc_pot()

  // =========================
  // Harvest index + harvest timing (Moriondo-style path)
  // =========================
  "HI_pot_base": 0.35,             // original: 0.35                | pipeline: matches paper “unstressed value (0.35)” :contentReference[oaicite:1]{index=1}
  "harvest_doy": 120,              // original: 120                 | pipeline: harvest stop day-of-year (site/cultivar calibration)

  // =========================================
  // Canopy light extinction k' (paper-style)
  // =========================================
  "Ck1": 0.52,                     // original: 0.52                | pipeline: eq6_k_prime()
  "Ck2": 0.000788,                 // original: 0.000788            | pipeline: eq6_k_prime()
  "Ck3": 0.76,                     // original: 0.76                | pipeline: eq6_k_prime()
  "Ck4": 1.25,                     // original: 1.25                | pipeline: eq6_k_prime()

  // ======================
  // Soil water (two layers)
  // ======================
  "TTSW1": 70.0,                   // original: 70.0 (mm)           | pipeline: eq17_ttsw(), eq18_19_recharge()
  "TTSW2": 110.0,                  // original: 110.0 (mm)          | pipeline: eq17_ttsw(), eq18_19_recharge()
  "Initial_Saturation": 0.9,       // original: 0.9                | pipeline: initial condition (ATSW fractions)

  "root_frac_layer1": 0.3333333333333333, // original: 0.3333333333333333 | pipeline: available for Eq.30–31 style partitioning if activated
  "root_frac_layer2": 0.6666666666666666, // original: 0.6666666666666666 | pipeline: available for Eq.30–31 style partitioning if activated

  // ==================================
  // Water-stress response (paper-style)
  // ==================================
  "RelTr_a": 6.17,                 // original: 6.17               | pipeline: eq22_rel_factor() for transpiration
  "RelTr_b": 13.45,                // original: 13.45              | pipeline: eq22_rel_factor() for transpiration
  "RelLAI_a": 78.24,               // original: 78.24              | pipeline: eq22_rel_factor() for LAI growth
  "RelLAI_b": 21.42,               // original: 21.42              | pipeline: eq22_rel_factor() for LAI growth

  "TE_coeff": 4.0,                 // original: 4.0                | pipeline: used as Kd (see eq16_te()) if that TE block is executed

  // =====================
  // Irrigation management
  // =====================
  "Irrigation_Threshold": 0.2,     // original: 0.2                | pipeline NOTE: run_simple_model() currently ignores this key
  "Irrigation_Efficiency": 0.95,   // original: 0.95               | pipeline: management assumption (apply when you implement Ir with efficiency)

  // =========================
  // ET / soil evaporation block
  // =========================
  "SALB": 0.2,                     // original: 0.2                | pipeline: eq32_sevp_pot()
  "gamma_kpa": 0.68,               // original: 0.68               | pipeline: matches Eq.(32) denominator term “... + 0.68” in the paper :contentReference[oaicite:2]{index=2}

  // =======================
  // Yield stress thresholds
  // =======================
  "FTSWo": 0.4,                    // original: 0.4                | pipeline: Eq.(35) “FTSWo is 0.4” :contentReference[oaicite:3]{index=3}
  "FTSWm": 0.05,                   // original: 0.05               | pipeline: lower bound in code for HIws (kept)
  "TMAXo": 30.0,                   // original: 30.0               | pipeline: Eq.(36) threshold “higher than 30°C (TMAXo)” :contentReference[oaicite:4]{index=4}
  "TMAXm": 40.0,                   // original: 40.0               | pipeline: Eq.(36) “TMaxm = 40°C” :contentReference[oaicite:5]{index=5}

  "anthesis_window_days": 7,       // original: 7                  | pipeline: used to average around anthesis
  "fresh_factor": 2.2,             // original: 2.2                | pipeline: conversion factor (fresh/dry), if used downstream
  "LAI_ini": 3.5                   // original: 3.5                | pipeline: initial canopy state
}
```

