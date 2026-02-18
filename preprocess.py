import pandas as pd
import numpy as np
from constants import ATECO_CATEOGORIES, FEATURE_COLS

def safe_div(num, denum):
    num, denum = pd.Series(num), pd.Series(denum)
    result = np.where(denum.isna() | num.isna() | (denum == 0), np.nan, num / denum)
    return pd.Series(result, index=num.index)

def fill_null_values(df):
    df_new = df.copy()

    df_new["debt_tot"] = df_new["debt_st"] + df_new["debt_lt"]
    df_new["liab_st"] = df_new["asst_current"] - df_new["wc_net"]
    df_new["fixed_asst_tot"] = df_new["asst_intang_fixed"] + df_new["asst_tang_fixed"] + df_new["asst_fixed_fin"]
    df_new["prof_operations"] = df_new["prof_operations"].fillna(df_new["rev_operating"] - df_new["COGS"])
    df_new["prof_operations"] = np.where(df_new['prof_operations'].isna() & df_new['roa'].notna() & df_new['asst_tot'].notna(),
                                (df_new['roa'] * df_new['asst_tot']) / 100, df_new['prof_operations'])
    df_new["rev_operating"] = df_new["rev_operating"].fillna(df_new["prof_operations"] + df_new["COGS"])
    df_new["COGS"] = df_new["COGS"].fillna(df_new["rev_operating"] - df_new["prof_operations"])
    df_new["roa"] = np.where(df_new["roa"].isna() & df_new['prof_operations'].notna() & df_new['asst_tot'].notna(),
                            safe_div(df_new['prof_operations'] * 100, df_new['asst_tot']), df_new['roa'])
    df_new["asst_tot"] = np.where(df_new['asst_tot'].isna() & df_new['roa'].notna() & df_new['prof_operations'].notna(),
                                safe_div(df_new['prof_operations'] * 100, df_new['roa']), df_new['asst_tot'])
    df_new["roe"] = np.where(df_new['roe'].isna() & df_new['profit'].notna() & df_new['eqty_tot'].notna(),
                            safe_div(df_new['profit'] * 100, df_new['eqty_tot']), df_new['roe'])
    df_new["profit"] = np.where(df_new['profit'].isna() & df_new['roe'].notna() & df_new['eqty_tot'].notna(),
                                (df_new['roe'] * df_new['eqty_tot']) / 100, df_new['profit'])
    df_new["eqty_tot"] = np.where(df_new['eqty_tot'].isna() & df_new['roe'].notna() & df_new['profit'].notna(),
                                safe_div(df_new['profit'] * 100, df_new['roe']), df_new['eqty_tot'])
    df_new["liab_tot"] = df_new["asst_tot"] - df_new["eqty_tot"]
    df_new["fixed_asst_tot"] = df_new["fixed_asst_tot"].fillna(df_new["eqty_tot"] - df_new["margin_fin"])

    return df_new


def fill_growth(df):
    latest_firm_df = pd.read_csv('latest_firm_data.csv')
    latest_revenue = latest_firm_df.set_index('id')['rev_operating'].to_dict()
    latest_assets  = latest_firm_df.set_index('id')['asst_tot'].to_dict()
    latest_year = latest_firm_df.set_index('id')['fs_year'].to_dict()

    df_new = df.copy()

    df_new['revenue_growth'] = df_new.apply(
        lambda row: ((row['rev_operating'] - latest_revenue[row['id']])/(row['fs_year'] - latest_year[row['id']]))
        if pd.notnull(row['fs_year']) and row['id'] in latest_revenue and (row['fs_year'] - latest_year[row['id']]) != 0
        else 0,
        axis=1
    )

    df_new['asset_growth'] = df_new.apply(
        lambda row: ((row['asst_tot'] - latest_assets[row['id']])/(row['fs_year'] - latest_year[row['id']]))
        if pd.notnull(row['fs_year']) and row['id'] in latest_assets and (row['fs_year'] - latest_year[row['id']]) != 0
        else 0,
        axis=1
    )

    return df_new


def calculate_ratios(df):
    df_new = df.copy()

    df_new["EBITDA_Margin"] = safe_div(df_new["ebitda"], df_new["rev_operating"])
    df_new["EBITDA_COGS"] = safe_div(df_new["ebitda"], df_new["COGS"])
    df_new["Liability_to_Asset"] = safe_div(df_new["liab_tot"], df_new["asst_tot"])
    df_new["Cash_Ratio"] = safe_div(df_new["cash_and_equiv"], df_new["liab_st"])
    df_new["Operating_Cash_Flow_Ratio"] = safe_div(df_new["cf_operations"], df_new["liab_st"])
    df_new["Fixed_to_Total_Assets"] = safe_div(df_new["fixed_asst_tot"], df_new["asst_tot"])
    df_new["Receivables_Turnover"] = safe_div(df_new["rev_operating"], df_new["AR"])
    df_new["Cash_Return_on_Assets"] = safe_div(df_new["cf_operations"], df_new["asst_tot"])

    return df_new

def set_ateco_sectors(df):
    ATECO_CATEOGORIES = {
        "Primary": [(1,3), (5,9)],
        "Manufacturing": [(10,33)],
        "Utilities": [(35,35), (36,39)],
        "Construction": [(41,43)],
        "Trade": [(45,47)],
        "Transportation": [(49,53)],
        "Hospitality": [(55,56)],
        "Information Technology": [(58,63)],
        "Financial Services": [(64,66), (68,68)],
        "Administrative Services": [(69,75), (77,82)],
        "Healthcare": [(84,84), (85,85), (86,88)],
        "Personal Services": [(90,93), (94,96), (97,98), (99,99)]
    }


def map_ateco_category(code):
    if pd.isna(code):
        return np.nan
    code = int(code)
    for cat, ranges in ATECO_CATEOGORIES.items():
        for start, end in ranges:
            if start <= code <= end:
                return cat
    return "Unclassified"

def map_ateco_categories(df):
    df_new = df.copy()
    df_new["ateco_category"] = df_new["ateco_sector"].apply(map_ateco_category)

    return df_new

def fill_null_by_sector_medians(df):
    latest_median_by_sector = pd.read_csv('median_by_category.csv')
    median_lookup = latest_median_by_sector.set_index(["ateco_category"])

    df_new = df.copy()

    for col in FEATURE_COLS:
        df_new[col] = df_new.apply(
            lambda row: median_lookup.loc[(row['ateco_category']), col] if pd.isna(row[col]) else row[col],
            axis=1
        )
    
    return df_new


def preprocess_data(df):
    df_new = fill_null_values(df)
    df_new = fill_growth(df_new)
    df_new = calculate_ratios(df_new)
    df_new = map_ateco_categories(df_new)
    df_new = fill_null_by_sector_medians(df_new)

    return df_new