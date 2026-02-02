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