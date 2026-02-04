from util import *

def get_estimation(df_season, params, VERBOSE=False):
    results = []
    LAI = params["LAI_ini"]
    ATSW1 = params["TTSW1"] * params["Initial_Saturation"]
    ATSW2 = params["TTSW2"] * params["Initial_Saturation"]
    DYSE = 1

    # Phenology accumulators (per season)
    chill_cum = 0.0
    forcing_cum = 0.0
    dormancy_released = False
    anthesis_occurred = False
    anthesis_date = None

    # HI state (per season)
    HI_pot = params["HI_pot_base"]
    HIa = HI_pot
    HIa_fixed = False
    anthesis_FTSW_samples = []
    anthesis_TMAX_samples = []

    # Biomass accumulation for [Eq. 11] (per season; stops at harvest)
    DM_cum = 0.0

    # Alternate bearing / memory across seasons
    year_state = dict(
        YLAI_prod_by_season={},
        HIa_by_season={},
        yield_by_season={},
    )
    YLAI_produced_this_season = 0.0

    for date, row in df_season.iterrows():
        year = int(date.year)
        doy = int(date.dayofyear)
        season_year = year if date.month >= 7 else year - 1

        # ------------------------------------------------------
        # SEASON RESET at July 1 (Southern Hemisphere)
        # ------------------------------------------------------
        if (date.month == 7) and (date.day == 1):
            prev_season = season_year - 1

            if prev_season not in year_state["YLAI_prod_by_season"]:
                year_state["YLAI_prod_by_season"][prev_season] = YLAI_produced_this_season
            YLAI_produced_this_season = 0.0

            prev_HIa = year_state["HIa_by_season"].get(prev_season, params["HI_pot_base"])
            HI_pot, PClf_next = eq37_40_update_next_season(
                prev_HIa, HI_pot_base=params["HI_pot_base"], PClf_pot_base=params["PClf_pot_base"]
            )
            params["PClf_pot_base"] = PClf_next

            chill_cum = 0.0
            forcing_cum = 0.0
            dormancy_released = False
            anthesis_occurred = False
            anthesis_date = None
            HIa = HI_pot
            HIa_fixed = False
            anthesis_FTSW_samples = []
            anthesis_TMAX_samples = []
            DM_cum = 0.0
            DYSE = 1

            if VERBOSE:
                print(f"\n--- SEASON {season_year}-{season_year+1} RESET (Jul 1) ---")
                print(f"    HI_pot updated [Eq.37–40] -> {HI_pot:.3f} | PClf_pot -> {params['PClf_pot_base']:.3f}")

        # ------------------------------------------------------
        # PHENOLOGY: Eq. (2) and Eq. (3) with daily TAVG
        # IMPORTANT: Eq.(2)-(3) are hourly in the paper; using daily TAVG
        # we scale by 24 to approximate hourly integration magnitudes.
        # ------------------------------------------------------
        Tavg = float(row["TAVG"])

        if not dormancy_released:
            if params["phenology_chill_model"] == "unichill":
                CU = eq2_unichill(Tavg, params["chill_a"], params["chill_b"], params["chill_c"])  # [Eq. 2]
                chill_cum += CU * 24.0
            else:
                CU_alt = alt_chill_triangular(Tavg, params["T_opt_chill_alt"], params["T_max_chill_alt"])
                chill_cum += CU_alt * 24.0

            if chill_cum >= params["Ccrit"]:
                dormancy_released = True
                if VERBOSE:
                    print(f"[PHENO] {date.date()} dormancy released | chill_cum={chill_cum:.1f}")

        elif not anthesis_occurred:
            if params["phenology_forcing_model"] == "logistic":
                FU = eq3_forcing_logistic(Tavg, params["forcing_d"], params["forcing_e"])          # [Eq. 3]
                forcing_cum += FU * 24.0
            else:
                forcing_cum += alt_forcing_gdd(Tavg, params["Tb_forcing_alt"])

            if forcing_cum >= params["FcritFlo"]:
                anthesis_occurred = True
                anthesis_date = date
                if VERBOSE:
                    print(f"[PHENO] {date.date()} anthesis reached")

        # ------------------------------------------------------
        # CANOPY / LIGHT: Eq. (6) -> Eq. (7/9) -> Eq. (5)
        # ------------------------------------------------------
        k_prime = eq6_k_prime(params["PlantD"], params["LAD"], params)   # [Eq. 6]
        v = eq7_v_from_lai(LAI, params["LAD"])                           # [Eq. 7/9]
        Int_OT = eq5_int_ot(k_prime, v)                                  # [Eq. 5]

        # ------------------------------------------------------
        # POTENTIAL GROWTH: Eq. (4) using PAR_MJ as radiation input
        # ------------------------------------------------------
        PAR_MJ = float(row["PAR_MJ"])
        DM_pot = eq4_dm_pot(Int_OT, PAR_MJ, params["RUE_ol"])            # [Eq. 4]

        # ------------------------------------------------------
        # IRRIGATION + SOIL RECHARGE: Eq. (18–19), status Eq. (20–21)
        # ------------------------------------------------------
        total_capacity = eq17_ttsw(params["TTSW1"], params["TTSW2"])      # [Eq. 17]
        current_FTSW = eq21_ftsw(ATSW1, ATSW2, params["TTSW1"], params["TTSW2"])  # [Eq. 21]

        irrigation_today = 0.0
        if (current_FTSW < params["Irrigation_Threshold"]) and (date.month in [9, 10, 11, 12, 1, 2, 3, 4]):
            deficit = total_capacity - (ATSW1 + ATSW2)
            irrigation_today = deficit / max(1e-9, params["Irrigation_Efficiency"])

        Rain = float(row["RAIN_mm"])
        Ir = float(irrigation_today)

        ATSW1, ATSW2 = eq18_19_recharge(ATSW1, ATSW2, Rain, Ir, params["TTSW1"], params["TTSW2"])  # [Eq. 18–19]

        FTSW1 = eq20_ftsw1(ATSW1, params["TTSW1"])                       # [Eq. 20]
        FTSW = eq21_ftsw(ATSW1, ATSW2, params["TTSW1"], params["TTSW2"]) # [Eq. 21]

        # ------------------------------------------------------
        # STRESS: Eq. (22), Eq. (27), Eq. (28)
        # ------------------------------------------------------
        RelTr = eq22_rel_factor(FTSW, params["RelTr_a"], params["RelTr_b"])        # [Eq. 22]
        RelLAI = eq22_rel_factor(FTSW, params["RelLAI_a"], params["RelLAI_b"])     # [Eq. 22]
        RelTE = eq27_relte(RelTr)                                                  # [Eq. 27]
        DM_act = eq28_dm_act(DM_pot, RelTr, RelTE)                                 # [Eq. 28]

        # ------------------------------------------------------
        # TRANSPIRATION: Eq. (16) -> Eq. (15) -> Eq. (23)
        # ------------------------------------------------------
        VPD_kPa = float(row["VPD_kPa"])
        TE = eq16_te(params["TE_coeff"], VPD_kPa)                                  # [Eq. 16]
        Tr_pot = eq15_tr_pot(DM_pot, TE)                                           # [Eq. 15]
        Tr_act = eq23_atr(Tr_pot, RelTr)                                           # [Eq. 23]

        # ------------------------------------------------------
        # SOIL EVAPORATION: Eq. (32–34) uses SRAD_MJ (total energy, not PAR)
        # ------------------------------------------------------
        if (Rain + Ir) > 0.5:
            DYSE = 1
        else:
            DYSE += 1

        SRAD_MJ = float(row["SRAD_MJ"])
        Delta = eq33_delta_kpa_per_c(Tavg)                                         # [Eq. 33]
        SEVP_pot = eq32_sevp_pot(SRAD_MJ, params["SALB"], Int_OT, Delta, params["gamma_kpa"])  # [Eq. 32]
        SEVP = eq34_sevp(SEVP_pot, DYSE)                                           # [Eq. 34]

        # ------------------------------------------------------
        # WATER UPTAKE: Eq. (30–31) operationalized via root fractions
        # ------------------------------------------------------
        Tr1 = Tr_act * params["root_frac_layer1"]
        Tr2 = Tr_act * params["root_frac_layer2"]

        ATSW1 -= (Tr1 + SEVP)
        if ATSW1 < 0:
            deficit = -ATSW1
            ATSW1 = 0.0
            ATSW2 = max(0.0, ATSW2 - deficit)

        ATSW2 = max(0.0, ATSW2 - Tr2)

        FTSW1 = eq20_ftsw1(ATSW1, params["TTSW1"])                                  # [Eq. 20]
        FTSW = eq21_ftsw(ATSW1, ATSW2, params["TTSW1"], params["TTSW2"])            # [Eq. 21]

        # ------------------------------------------------------
        # HI AT ANTHESIS: Eq. (35–36), additive HIa rule
        # ------------------------------------------------------
        if anthesis_occurred and (not HIa_fixed):
            anthesis_FTSW_samples.append(FTSW)
            anthesis_TMAX_samples.append(float(row["T2M_MAX"]))

            if len(anthesis_FTSW_samples) >= params["anthesis_window_days"]:
                FTSWant = float(np.mean(anthesis_FTSW_samples))
                TMAXant = float(np.mean(anthesis_TMAX_samples))

                HIws = eq35_hiws(HI_pot, FTSWant, params["FTSWo"], params["FTSWm"])  # [Eq. 35]
                HIhs = eq36_hihs(HI_pot, TMAXant, params["TMAXo"], params["TMAXm"])  # [Eq. 36]

                HIa = max(0.0, min(HI_pot, HIws + HIhs - HI_pot))
                HIa_fixed = True

                year_state["HIa_by_season"][season_year] = HIa

                if VERBOSE:
                    print(f"[HI] {date.date()} HIa fixed | HIws={HIws:.3f}, HIhs={HIhs:.3f}, HIa={HIa:.3f}")

        # ------------------------------------------------------
        # LEAF GROWTH: Eq. (8) + Eq. (24); Senescence Eq. (10); LAI update
        # ------------------------------------------------------
        PClf_pot = params["PClf_pot_base"]

        LAI_inc_pot = eq8_lai_inc_pot(DM_pot, PClf_pot, params["SLA_ol"])          # [Eq. 8]
        ALAI_inc = eq24_alai_inc(LAI_inc_pot, RelLAI)                               # [Eq. 24]

        if ALAI_inc > 0:
            YLAI_produced_this_season += ALAI_inc

        YLAI_y_minus_2 = year_state["YLAI_prod_by_season"].get(season_year - 2, 0.0)
        LAI_sen = eq10_lai_senescence(doy, YLAI_y_minus_2)                          # [Eq. 10]

        LAI = max(0.1, LAI + ALAI_inc - LAI_sen)

        # ------------------------------------------------------
        # YIELD ACCUMULATION: Eq. (11), STOP at harvest DOY
        # ------------------------------------------------------
        harvest_passed = (date.month <= 6) and (doy >= int(params["harvest_doy"]))

        if dormancy_released and (not harvest_passed):
            DM_cum += DM_act

        yield_gm2 = eq11_yield(HIa if HIa_fixed else HI_pot, DM_cum)                # [Eq. 11]

        results.append(
            dict(
                DATE=date,
                SEASON_YEAR=season_year,
                DOY=doy,
                LAI=LAI,
                v=v,
                k_prime=k_prime,
                Int_OT=Int_OT,
                DM_pot=DM_pot,
                DM_act=DM_act,
                DM_cum=DM_cum,
                HI_pot=HI_pot,
                HIa=HIa if HIa_fixed else np.nan,
                Yield_gm2=yield_gm2,
                ATSW1=ATSW1,
                ATSW2=ATSW2,
                FTSW1=FTSW1,
                FTSW=FTSW,
                RelTr=RelTr,
                RelLAI=RelLAI,
                RelTE=RelTE,
                TE=TE,
                Tr_pot=Tr_pot,
                Tr_act=Tr_act,
                SEVP_pot=SEVP_pot,
                SEVP=SEVP,
                DYSE=DYSE,
                Rain=Rain,
                Irrigation=Ir,
                VPD_kPa=VPD_kPa,
                dormancy_released=dormancy_released,
                anthesis_occurred=anthesis_occurred,
                harvest_passed=harvest_passed,
            )
        )


    # ==========================================================
    # 6. RESULTS & AUDIT (Chile-consistent: group by SEASON_YEAR)
    # Includes ABI [Eq. 41]
    # ==========================================================
    res_df = pd.DataFrame(results).set_index("DATE")

    seasonal_yield_gm2 = res_df.groupby("SEASON_YEAR")["Yield_gm2"].max()
    seasonal_yield_kg_ha_dm = seasonal_yield_gm2 * 10.0
    seasonal_yield_kg_ha_fresh = seasonal_yield_kg_ha_dm * params["fresh_factor"]

    seasonal_irrig_mm = res_df.groupby("SEASON_YEAR")["Irrigation"].sum()

    abi = eq41_abi([seasonal_yield_kg_ha_dm.loc[s] for s in seasonal_yield_kg_ha_dm.index])

    # With one season, ABI will be NaN by definition; keep behavior explicit.
    if len(seasonal_yield_kg_ha_dm) >= 3:
        core_seasons = seasonal_yield_kg_ha_dm.index[1:-1]
        avg_yield_dm = seasonal_yield_kg_ha_dm.loc[core_seasons].mean()
        avg_yield_fresh = seasonal_yield_kg_ha_fresh.loc[core_seasons].mean()
        avg_irrig = seasonal_irrig_mm.loc[core_seasons].mean()
    else:
        avg_yield_dm = seasonal_yield_kg_ha_dm.mean()
        avg_yield_fresh = seasonal_yield_kg_ha_fresh.mean()
        avg_irrig = seasonal_irrig_mm.mean()

    print("\n" + "=" * 80)
    print(f"FINAL AUDIT (Chile) | LOC")
    print("=" * 80)
    print(f"Seasons evaluated (SEASON_YEAR): {list(seasonal_yield_kg_ha_dm.index)}")
    print(f"Phenology models: Chill={params['phenology_chill_model']} | Forcing={params['phenology_forcing_model']}")
    print(f"Harvest DOY: {params['harvest_doy']}  | Irrigation threshold FTSW={params['Irrigation_Threshold']:.2f}")
    print("-" * 80)
    print(f"Average Yield (DM):      {avg_yield_dm:.0f} kg/ha")
    print(f"Average Yield (Fresh*):  {avg_yield_fresh:.0f} kg/ha   (*fresh_factor={params['fresh_factor']})")
    print(f"Average Irrigation:      {avg_irrig:.0f} mm/season-year")
    print(f"Alternate Bearing Index: {abi:.3f}   [Eq. 41]")
    print("-" * 80)

    return avg_yield_fresh
