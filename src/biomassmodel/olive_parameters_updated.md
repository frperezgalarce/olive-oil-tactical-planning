# Olive model parameter update (JSONC)

This file provides an **updated parameter set** in JSON-with-comments format (JSONC).  
Each updated item keeps the **original value** and includes a **validation source**.

> **Important unit note (RUE):**
> - `RUE_ol` is intended for **intercepted PAR** (MJ m⁻² d⁻¹) as radiation driver.  
> - If your code feeds **global shortwave radiation (Rs)** instead, use approximately  
>   `RUE_ol_Rs ≈ RUE_ol_PAR × 0.48` (PAR ≈ 0.48·Rs in many agro-meteorological applications).

```jsonc
{
  "PlantD": 1600.0,                // original: 1600.0 (plants/ha) | source: design input (orchard geometry)
  "PlantA": 6.25,                  // original: 6.25 (m²/plant)    | derived: 10000/PlantD
  "LAD": 2.0,                      // original: 2.0                | source: model calibration (site/canopy-specific)

  "phenology_chill_model": "unichill",   // original: "unichill"  | source: modeling choice
  "phenology_forcing_model": "logistic", // original: "logistic"  | source: modeling choice

  "tb_budbreak": 8.5,              // original: 8.5 (°C)          | source: cultivar/site calibration (keep until local phenology fit)

  "chill_a": 0.01,                 // original: 0.01              | source: chill model calibration (cultivar-specific)
  "chill_b": -0.5,                 // original: -0.5              | source: chill model calibration (cultivar-specific)
  "chill_c": 7.2,                  // original: 7.2               | source: chill model calibration (cultivar-specific)
  "Ccrit": 350.0,                  // original: 350.0 (CU)         | source: chill requirement placeholder (cultivar-specific)

  "forcing_d": 0.5,                // original: 0.5               | source: forcing model calibration (cultivar-specific)
  "forcing_e": 9.0,                // original: 9.0               | source: forcing model calibration (cultivar-specific)
  "FcritFlo": 450.0,               // original: 450.0 (GDD)         | source: forcing requirement placeholder (cultivar-specific)

  "T_opt_chill_alt": 7.2,          // original: 7.2 (°C)           | source: chill response parameter (cultivar-specific)
  "T_max_chill_alt": 16.0,         // original: 16.0 (°C)          | source: chill response parameter (cultivar-specific)
  "Tb_forcing_alt": 9.0,           // original: 9.0 (°C)           | source: forcing base temperature (cultivar-specific)

  // =========================
  // Yield / biomass parameters
  // =========================
  "RUE_ol": 0.86,                  // original: 0.98
                                  // updated: 0.86 g DM / MJ intercepted PAR
                                  // source: Villalobos et al. (2006) – abstract reports RUE≈0.86 g DM MJ⁻¹ PAR (Eur. J. Agron. 24:296–303; DOI: 10.1016/j.eja.2005.07.002)

  "SLA_ol": 5.2,                   // original: 5.2 (m²/kg)        | keep: plausible olive SLA; calibrate if leaf traits measured

  "HI_pot_base": 0.50,             // original: 0.35
                                  // updated: 0.50 (unitless)
                                  // source: adult olive HI≈0.50 used/assumed in modeling literature citing Villalobos et al. (2006) (e.g., Frontiers in Sustainable Food Systems, 2024)

  "harvest_doy": 120,              // original: 120                 | keep: site/cultivar dependent (Talca: calibrate using local harvest records)
  "PClf_pot_base": 0.18,           // original: 0.18                | keep: model-calibrated partition coefficient

  // =========================================
  // Canopy light extinction (retain; validated)
  // =========================================
  "Ck1": 0.52,                     // original: 0.52               | source: Villalobos-type k' parameterization (see Moriondo et al., 2019, citing Villalobos et al.)
  "Ck2": 0.000788,                 // original: 0.000788           | source: same as above
  "Ck3": 0.76,                     // original: 0.76               | source: same as above
  "Ck4": 1.25,                     // original: 1.25               | source: same as above

  // ======================
  // Soil water (TAW split)
  // ======================
  "TTSW1": 62.0,                   // original: 70.0
                                  // updated: 62.0 (mm)
                                  // rationale: rescaled so (TTSW1+TTSW2)=160 mm to match TAW used in your parameter notes

  "TTSW2": 98.0,                   // original: 110.0
                                  // updated: 98.0 (mm)
                                  // rationale: rescaled so (TTSW1+TTSW2)=160 mm to match TAW used in your parameter notes

  "Initial_Saturation": 0.9,       // original: 0.9                | initial condition (field-capacity assumption)

  "root_frac_layer1": 0.60,        // original: 0.3333333333333333
                                  // updated: 0.60 (top layer share)
                                  // source: olive root activity typically concentrates in upper soil under drip irrigation; validate with Fernández et al. (1991) Plant and Soil 133:239–251 (root distribution under drip)

  "root_frac_layer2": 0.40,        // original: 0.6666666666666666
                                  // updated: 0.40 (bottom layer share)
                                  // source: see above

  // ==================================
  // Stress-response shape (calibrated)
  // ==================================
  "RelTr_a": 6.17,                 // original: 6.17               | keep: calibrated
  "RelTr_b": 13.45,                // original: 13.45              | keep: calibrated
  "RelLAI_a": 78.24,               // original: 78.24              | keep: calibrated
  "RelLAI_b": 21.42,               // original: 21.42              | keep: calibrated

  "TE_coeff": 4.0,                 // original: 4.0                | keep: acceptable; optional literature default often ~5.0 Pa (Tanner & Sinclair theory)

  // =====================
  // Irrigation management
  // =====================
  "Irrigation_Threshold": 0.35,    // original: 0.2
                                  // updated: 0.35 (fraction of TAW remaining)
                                  // rationale: aligns with p≈0.65 for olives (irrigate near 65% depletion => ~35% remaining)
                                  // source: FAO-56 indicates p≈0.65 for olives (Table of rooting depth & p)

  "Irrigation_Efficiency": 0.95,   // original: 0.95               | keep: well-managed drip assumption

  // =========================
  // ET0 / radiation constants
  // =========================
  "SALB": 0.2,                     // original: 0.2                | keep: typical soil albedo assumption

  "gamma_kpa": 0.067,              // original: 0.68
                                  // updated: 0.067 (kPa/°C)
                                  // source: FAO-56 psychrometric constant γ ≈ 0.665×10⁻³·P; at ~101 kPa => ~0.067 kPa/°C

  // =======================
  // Yield stress thresholds
  // =======================
  "FTSWo": 0.4,                    // original: 0.4                | keep: calibrated threshold
  "FTSWm": 0.05,                   // original: 0.05               | keep: calibrated threshold
  "TMAXo": 30.0,                   // original: 30.0               | keep: calibrated threshold
  "TMAXm": 40.0,                   // original: 40.0               | keep: calibrated threshold

  "anthesis_window_days": 7,       // original: 7                  | keep: simplifying assumption
  "fresh_factor": 2.2,             // original: 2.2                | keep: moisture conversion (fresh/dry)
  "LAI_ini": 3.5                   // original: 3.5                | keep: initial canopy state (site-specific)
}
```

## Validation links (quick access)
- Villalobos et al. (2006), *European Journal of Agronomy* 24:296–303. DOI: 10.1016/j.eja.2005.07.002  
- FAO-56 (Allen et al., 1998), *Crop Evapotranspiration*. (psychrometric constant γ; olives p≈0.65)  
- Fernández et al. (1991), *Plant and Soil* 133:239–251. (olive root distribution under drip)  
- Moriondo et al. (2019), *European Journal of Agronomy*. (k' parameterization with Ck1–Ck4 values)
