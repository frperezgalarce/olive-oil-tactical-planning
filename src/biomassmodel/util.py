import numpy as np
import math
import pandas as pd
# --- Yield Parameters ---
CRUE = 0.135       # [g/MJ] Biomass RUE (total solar), adapted from literature
CHI = 0.3        # [unitless] Olive Harvest Index (Dry Fruit / Total Biomass)
CCh = CRUE * CHI # [g(yield)/MJ] Composite Crop Coefficient (0.3)

# --- Phenology & Temperature ---
fSolarmax = 0.90   # [unitless] Max fraction of radiation intercepted
I50A = 492         # [°C·day] GDD for canopy closure at 50%
I50B = 145         # [°C·day] GDD from senescence start to Tsum
Tsum = 2000        # [°C·day] GDD required for harvest (Flowering->Harvest)
Tbase = 9          # [°C] Baseline temperature for olive growth
Topt = 25          # [°C] Optimal temperature
Theat = 35         # [°C] Temperature above which heat stress begins
Textreme = 45      # [°C] Temperature at which growth stops

# --- Water & Soil ---
Swater = 0.6               # [unitless] RUE sensitivity to drought
critical_depletion = 80    # [mm] RAW: Threshold for stress (Irrigation Trigger)
critical_water = 160       # [mm] TAW: Max water storage (Total Available Water)
p = 0.5                    # [unitless] Depletion factor

# --- Environmental & Site ---
SCO2 = 1.0       # [unitless] CRITICAL: Neutralized CO2 multiplier
alfa = 0.23      # [unitless] Surface albedo
SIGMA = 4.903e-9 # [W·m⁻²·K⁻⁴] Stefan-Boltzmann constant
Altitude = 150     # [m] Elevation for Siracusa, Chile
n = 2.45         # [mol/mol]
E = 0.622        # [unitless]
Cp = 1.013 / 1000  # [MJ/kg·°C]
G = 0            # [MJ/m²/day] Soil heat flux


def eq1_gdh_daily(Tavg, tb):
    """[Eq. 1] GDH daily approximation: GDH = max(0, Tavg - tb)."""
    return max(0.0, Tavg - tb)

def eq2_unichill(Tavg, a, b, c):
    """
    [Eq. 2] Chill Units (UniChill):
      CU = 1/(1 + exp(a*(T-c)^2 + b*(T-c))).
    """
    z = a * (Tavg - c) ** 2 + b * (Tavg - c)
    return 1.0 / (1.0 + np.exp(z))

def eq3_forcing_logistic(Tavg, d, e):
    """
    [Eq. 3] Forcing Units:
      FU = 1/(1 + exp(-d*(T-e))).
    """
    z = -d * (Tavg - e)
    return 1.0 / (1.0 + np.exp(z))

# --------------------------
# [ALT] Chill triangular (NOT Eq. 2)
# --------------------------
def alt_chill_triangular(Tmean, Topt, Tmax):
    """[ALT] Triangular chill (NOT Eq. 2)."""
    if Tmean <= 0 or Tmean >= Tmax:
        return 0.0
    return (Tmean / Topt) if Tmean <= Topt else ((Tmax - Tmean) / (Tmax - Topt))

# --------------------------
# [ALT] Forcing via GDD (NOT Eq. 3)
# --------------------------
def alt_forcing_gdd(Tavg, Tb):
    """[ALT] GDD forcing (NOT Eq. 3): max(0, Tavg - Tb)."""
    return max(0.0, Tavg - Tb)

# --------------------------
# [Eq. 6] k'
# --------------------------
def eq6_k_prime(PlantD, LAD, p):
    """
    [Eq. 6] Extinction coefficient:
      k' = Ck1 + Ck2*PlantD - Ck3*exp(-Ck4*LAD).
    """
    return p["Ck1"] + p["Ck2"] * PlantD - p["Ck3"] * np.exp(-p["Ck4"] * LAD)

# --------------------------
# [Eq. 7/9] v from LAI and LAD
# --------------------------
def eq7_v_from_lai(LAI, LAD):
    """
    [Eq. 7/9] Canopy volume per ground area (consistent state derivation):
      v = LAI / LAD  (units: m3/m2).
    """
    return LAI / max(1e-9, LAD)

# --------------------------
# [Eq. 5] Int_OT
# --------------------------
def eq5_int_ot(k_prime, v):
    """
    [Eq. 5] Intercepted radiation fraction:
      Int_OT = 1 - exp(-k' * v).
    """
    return 1.0 - np.exp(-k_prime * v)

# --------------------------
# [Eq. 4] DM potential
# NOTE: Use PAR (MJ/m²/day) as the radiation driver for biomass (common in crop models).
# --------------------------
def eq4_dm_pot(Int_OT, RAD_MJ_for_DM, RUE):
    """
    [Eq. 4] DM_pot = Int_OT * RAD * RUE
    Returns DM_pot in g/m²/day if RAD is MJ/m²/day and RUE is g/MJ.
    """
    return Int_OT * RAD_MJ_for_DM * RUE

# --------------------------
# [Eq. 16] TE
# --------------------------
def eq16_te(Kd_pa, VPD_kpa):
    """
    [Eq. 16] TE = Kd / VPD
    Paper uses Kd in Pa and VPD in kPa.
    """
    return Kd_pa / max(1e-6, VPD_kpa)

# --------------------------
# [Eq. 15] Transpiration potential
# --------------------------
def eq15_tr_pot(DM_pot_gm2, TE):
    """
    [Eq. 15] Tr_pot = DM_pot / TE
    Used consistently as a water-demand proxy.
    """
    return DM_pot_gm2 / max(1e-9, TE)

# --------------------------
# [Eq. 17] TTSW total
# --------------------------
def eq17_ttsw(TTSW1, TTSW2):
    """[Eq. 17] TTSW = TTSW1 + TTSW2."""
    return TTSW1 + TTSW2

# --------------------------
# [Eq. 18–19] Soil recharge (two layers)
# --------------------------
def eq18_19_recharge(ATSW1, ATSW2, Rain, Ir, TTSW1, TTSW2):
    """
    [Eq. 18] ATSW1_t = min(ATSW1_{t-1} + Rain + Ir, TTSW1)
    [Eq. 19] ATSW2_t = min(ATSW2_{t-1} + excess_from_layer1, TTSW2)
    """
    ATSW1_new = ATSW1 + Rain + Ir
    excess = max(0.0, ATSW1_new - TTSW1)
    ATSW1_new = min(ATSW1_new, TTSW1)
    ATSW2_new = min(ATSW2 + excess, TTSW2)
    return ATSW1_new, ATSW2_new

# --------------------------
# [Eq. 20] FTSW1
# --------------------------
def eq20_ftsw1(ATSW1, TTSW1):
    """[Eq. 20] FTSW1 = ATSW1/TTSW1."""
    return ATSW1 / max(1e-9, TTSW1)

# --------------------------
# [Eq. 21] FTSW (whole profile)
# --------------------------
def eq21_ftsw(ATSW1, ATSW2, TTSW1, TTSW2):
    """[Eq. 21] FTSW = (ATSW1+ATSW2)/(TTSW1+TTSW2)."""
    return (ATSW1 + ATSW2) / max(1e-9, (TTSW1 + TTSW2))

# --------------------------
# [Eq. 22] Relative stress factor
# --------------------------
def eq22_rel_factor(FTSW, a, b):
    """[Eq. 22] Rel = 1/(1 + a*exp(-b*FTSW))."""
    return 1.0 / (1.0 + a * np.exp(-b * FTSW))

# --------------------------
# [Eq. 23] Actual transpiration
# --------------------------
def eq23_atr(Tr_pot, RelTr):
    """[Eq. 23] Tr_act = Tr_pot * RelTr."""
    return Tr_pot * RelTr

# --------------------------
# [Eq. 24] Actual leaf-area increment
# --------------------------
def eq24_alai_inc(LAI_inc_pot, RelLAI):
    """[Eq. 24] ALAI_inc = LAI_inc_pot * RelLAI."""
    return LAI_inc_pot * RelLAI

# --------------------------
# [Eq. 27] RelTE
# --------------------------
def eq27_relte(RelTr):
    """[Eq. 27] RelTE = -0.74*RelTr + 1.74."""
    return -0.74 * RelTr + 1.74

# --------------------------
# [Eq. 28] Actual biomass
# --------------------------
def eq28_dm_act(DM_pot, RelTr, RelTE):
    """[Eq. 28] DM_act = DM_pot * RelTr * RelTE."""
    return DM_pot * RelTr * RelTE

# --------------------------
# [Eq. 8] LAI increment from biomass partition to leaves
# --------------------------
def eq8_lai_inc_pot(DM_pot_gm2, PClf_pot, SLA_m2_per_kg):
    """
    [Eq. 8] LAI_inc_pot = (DM_pot * PClf_pot) * SLA
    DM_pot is g/m²/day -> convert to kg/m²/day before SLA.
    """
    DM_leaf_kgm2 = (DM_pot_gm2 * PClf_pot) / 1000.0
    return DM_leaf_kgm2 * SLA_m2_per_kg

# --------------------------
# [Eq. 10] LAI senescence (distributed over DOY window using YLAI_{y-2})
# --------------------------
def eq10_lai_senescence(doy, YLAI_y_minus_2, DOY_ini=250, DOY_end=330):
    """
    [Eq. 10] LAI_sen = YLAI_{y-2}/(DOY_end-DOY_ini+1) within window, else 0.
    NOTE: DOY_ini/DOY_end should be calibrated to olive phenology; kept as placeholders.
    """
    if YLAI_y_minus_2 <= 0:
        return 0.0
    if DOY_ini <= doy <= DOY_end:
        return YLAI_y_minus_2 / float(DOY_end - DOY_ini + 1)
    return 0.0

# --------------------------
# [Eq. 33] Delta (slope of saturation vapour pressure curve)
# --------------------------
def eq33_delta_kpa_per_c(Tavg):
    """[Eq. 33] Delta from Tetens (kPa/°C)."""
    es = 0.6108 * np.exp(17.27 * Tavg / (Tavg + 237.3))
    return 4098.0 * es / ((Tavg + 237.3) ** 2)

# --------------------------
# [Eq. 32] Potential soil evaporation (energy-limited form)
# --------------------------
def eq32_sevp_pot(SRAD_MJ, SALB, INT_tot, Delta, gamma_kpa=0.68, lambda_mj_per_mm=2.45):
    """
    [Eq. 32] SEVP_pot = SRAD*(1-SALB)*(1-INT_tot) * Delta/(Delta+gamma)
    Convert MJ/m² to mm via lambda (2.45 MJ/mm).
    """
    net_rad = SRAD_MJ * (1.0 - SALB) * (1.0 - INT_tot)
    mm_equiv = net_rad / max(1e-9, lambda_mj_per_mm)
    return mm_equiv * (Delta / max(1e-9, (Delta + gamma_kpa)))

# --------------------------
# [Eq. 34] Actual soil evaporation under drying cycle
# --------------------------
def eq34_sevp(sevp_pot, DYSE):
    """[Eq. 34] SEVP = SEVP_pot*(sqrt(DYSE)-sqrt(DYSE-1))."""
    DYSE = max(1.0, float(DYSE))
    return sevp_pot * (np.sqrt(DYSE) - np.sqrt(max(0.0, DYSE - 1.0)))

# --------------------------
# [Eq. 35] HI water-stress at anthesis
# --------------------------
def eq35_hiws(HI_pot, FTSWant, FTSWo=0.40, FTSWm=0.05):
    """
    [Eq. 35] HIws = HI_pot if FTSWant>FTSWo
                  else HI_pot*(1-(FTSWo-FTSWant)/(FTSWo-FTSWm)).
    """
    if FTSWant >= FTSWo:
        return HI_pot
    if FTSWant <= FTSWm:
        return 0.0
    return HI_pot * (1.0 - (FTSWo - FTSWant) / max(1e-9, (FTSWo - FTSWm)))

# --------------------------
# [Eq. 36] HI heat-stress at anthesis
# --------------------------
def eq36_hihs(HI_pot, TMAXant, TMAXo=30.0, TMAXm=40.0):
    """
    [Eq. 36] HIhs = HI_pot if TMAXant<TMAXo
                  else HI_pot*(1-(TMAXant-TMAXo)/(TMAXm-TMAXo)).
    """
    if TMAXant <= TMAXo:
        return HI_pot
    if TMAXant >= TMAXm:
        return 0.0
    return HI_pot * (1.0 - (TMAXant - TMAXo) / max(1e-9, (TMAXm - TMAXo)))

# --------------------------
# [Eq. 11] Yield
# --------------------------
def eq11_yield(HI_actual, DM_cum_gm2):
    """[Eq. 11] Yield_gm2 = HI * DM_cum (up to harvest)."""
    return HI_actual * DM_cum_gm2

# --------------------------
# [Eq. 37–40] Alternate bearing updates (apply at season boundary)
# --------------------------
def eq37_40_update_next_season(HIa, HI_pot_base=0.35, PClf_pot_base=0.18):
    """
    [Eq. 37–40] Unified forms (paper-derived):
      HI_pot_next = 0.70 - HIa      (clamped)
      PClf_next   = PClf_base*(0.65 + HIa)
    """
    HI_pot_next = max(0.0, min(0.70, 0.70 - HIa))
    PClf_next = max(0.0, PClf_pot_base * (0.65 + HIa))
    return HI_pot_next, PClf_next

# --------------------------
# [Eq. 41] Alternate Bearing Index (ABI)
# --------------------------
def eq41_abi(yields):
    """[Eq. 41] ABI = (1/(n-1)) * sum(|Yi - Yi+1|/(Yi + Yi+1))."""
    ys = [y for y in yields if y is not None and not np.isnan(y)]
    n = len(ys)
    if n < 2:
        return np.nan
    terms = []
    for i in range(n - 1):
        denom = ys[i] + ys[i + 1]
        terms.append(abs(ys[i] - ys[i + 1]) / denom if denom > 0 else 0.0)
    return sum(terms) / (n - 1)

def run_simple_model(weather_data, irrigation_strategy):
    """
    Runs a daily water balance and yield simulation.

    irrigation_strategy: 'rainfed' or 'optimized'
    """

    # Initialize state variables
    # Start soil at the stress threshold
    Depletion_t = critical_depletion

    # Lists to store results
    daily_yield_list = []
    f_water_list = []
    depletion_list = []
    irrigation_list = []
    paw_list = []

    for i, day in weather_data.iterrows():

        # --- 1. Calculate Start-of-Day Water Stress ---
        # PAW (Plant Available Water) = Total Capacity - Current Depletion
        PAW_t = max(0, critical_water - Depletion_t)

        # Calculate f_water (Water Stress Factor)
        f_water_t = 1.0
        if day['ETO'] > 0:
            # ARID factor from the original fWater function
            ARID_t = 1 - (min(day['ETO'], 0.096 * PAW_t) / day['ETO'])
            f_water_t = 1 - (Swater * ARID_t)

        # --- 2. Determine Irrigation for the Day ---
        irrigation_today = 0.0
        if irrigation_strategy == 'optimized':
            # SIMPLE RULE: If depletion is past the trigger, irrigate
            # to meet the crop's demand for that day.
            if Depletion_t > critical_depletion:
                irrigation_today = day['ETC']

        # --- 3. Calculate Final Yield for the Day ---
        # Yield is limited by the *most* stressful factor (heat or water)
        f_stress_t = min(day['FHeat'], f_water_t)
        daily_yield = CCh * day['EF_t'] * f_stress_t
        daily_yield = max(0, daily_yield) # Yield cannot be negative

        # --- 4. Run Daily Water Balance (to find next day's Depletion) ---
        Evaporated = (day['PRECIPITATION'] * 0.2) if (day['PRECIPITATION'] > 0 and day['PRECIPITATION'] >= day['ETO'] * 0.2) else (day['ETO'] * 0.2 if day['PRECIPITATION'] == 0 else day['PRECIPITATION'])
        Evaporated = max(0, Evaporated)

        Water_Gains = day['PRECIPITATION'] + irrigation_today
        Water_Losses = day['ETC'] + Evaporated # Simplified, ignores deep percolation for this model

        # Update Depletion: Depletion_t = Depletion_t-1 + Losses - Gains
        Depletion_t += Water_Losses - Water_Gains

        # Constrain depletion to be within 0 and TAW (critical_water)
        Depletion_t = max(0, min(Depletion_t, critical_water))

        # --- 5. Store all daily results ---
        daily_yield_list.append(daily_yield)
        f_water_list.append(f_water_t)
        depletion_list.append(Depletion_t)
        irrigation_list.append(irrigation_today)
        paw_list.append(PAW_t)

    # --- 6. Create results DataFrame ---
    results_df = pd.DataFrame({
        'Daily_Yield_gm2': daily_yield_list,
        'f_water': f_water_list,
        'Depletion_mm': depletion_list,
        'Irrigation_mm': irrigation_list,
        'PAW_mm': paw_list
    }, index=weather_data.index)

    results_df['Cumulative_Yield_kgHa'] = (results_df['Daily_Yield_gm2'].cumsum()) * 10

    return results_df


def fTemp(Tbase, Topt, T_series):
    T = T_series.values
    ftemp = np.zeros(len(T))
    ftemp[T < Tbase] = 0
    ftemp[(T >= Tbase) & (T <= Topt)] = (T[(T >= Tbase) & (T <= Topt)] - Tbase) / (Topt - Tbase)
    ftemp[T > Topt] = 1
    return ftemp

def fHeat(Theat, Textreme, Tmax_series):
    Tmax = Tmax_series.values
    fheat = np.ones(len(Tmax))
    fheat[(Tmax > Theat) & (Tmax <= Textreme)] = 1 - np.round((Tmax[(Tmax > Theat) & (Tmax <= Textreme)] - Theat) / (Textreme - Theat), 3)
    fheat[Tmax > Textreme] = 0
    return fheat

def fSolar(fsolar, i50a, i50b, tsum, tbase, temp_series):
    temp_values = temp_series.values
    temp_adj = [(abs(i - tbase) + (i - tbase)) / 2 for i in temp_values]
    tt = [0.0]
    for i in range(1, len(temp_adj)):
        tt.append(temp_adj[i] + tt[i - 1])

    growth = [fsolar / (1 + math.exp(-0.01 * (i - i50a))) for i in tt]
    senescence = [fsolar / (1 + math.exp(0.01 * (i - (tsum - i50b)))) for i in tt]
    result = [min(growth[i], senescence[i]) for i in range(len(tt))]
    return result

def RN(alfa, RS_s, RSO_s, SIGMA, TMIN_s, TMAX_s, RHMEAN_s, temp_s, Altitude, n, E, Cp, G, WS2M_s):
    TMAX = TMAX_s.values
    TMIN = TMIN_s.values
    RHMEAN = RHMEAN_s.values
    RS = RS_s.values
    RSO = RSO_s.values
    temp = temp_s.values
    WS2M = WS2M_s.values

    eTMAX = [0.6108 * math.exp((17.27 * TMAX[i]) / (TMAX[i] + 237.3)) for i in range(len(TMAX))]
    eTMIN = [0.6108 * math.exp((17.27 * TMIN[i]) / (TMIN[i] + 237.3)) for i in range(len(TMIN))]
    ea = [(RHMEAN[i] / 100) * ((eTMAX[i] + eTMIN[i]) / 2) for i in range(len(RHMEAN))]
    TMAXK = [TMAX[i] + 273.16 for i in range(len(TMAX))]
    TMINK = [TMIN[i] + 273.16 for i in range(len(TMIN))]
    RNS = [(1 - alfa) * RS[i] for i in range(len(RS))]

    RSO_safe = [max(r, 0.001) for r in RSO]
    RS_RSO_ratio = [min(RS[i] / RSO_safe[i], 1.0) for i in range(len(RS))]

    RNL = [SIGMA * ((TMINK[i]**4 + TMAXK[i]**4) / 2) * (0.34 - 0.14 * (ea[i]**0.5)) * (1.35 * RS_RSO_ratio[i] - 0.35) for i in range(len(TMIN))]
    RN = [RNS[i] - RNL[i] for i in range(len(TMIN))]

    delta = [4098 * (0.6108 * math.exp(17.27 * temp[i] / (temp[i] + 237.3))) / ((temp[i] + 237.3) ** 2) for i in range(len(temp))]
    es = [(eTMAX[i] + eTMIN[i]) / 2 for i in range(len(TMIN))]
    es_ea = [max(0, es[i] - ea[i]) for i in range(len(temp))]
    P = 101.3 * ((293 - 0.0065 * Altitude) / 293) ** 5.26
    y = (Cp * P) / (n * E)

    ETO = [(0.408 * delta[i] * (RN[i] - G) + y * (900 / (temp[i] + 273)) * WS2M[i] * es_ea[i]) /
           (delta[i] + y * (1 + 0.34 * WS2M[i])) for i in range(len(temp))]

    ETO = [max(0, e) for e in ETO]
    KS = [1.0] * len(ETO) # Olive crop coefficient
    ETC = [KS[i] * ETO[i] for i in range(len(ETO))]

    return ETO, ETC