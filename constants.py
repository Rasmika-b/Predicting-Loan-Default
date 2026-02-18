FEATURE_COLS = ['roa', 'EBITDA_Margin', 'EBITDA_COGS', 'Liability_to_Asset', 'Cash_Ratio', 'Operating_Cash_Flow_Ratio',
                'Fixed_to_Total_Assets', 'Receivables_Turnover', 'Cash_Return_on_Assets', 'asset_growth',
                'revenue_growth']

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

Q_LOW = {
    'roa': -21.44,
    'EBITDA_Margin': -11.56301296860138,
    'EBITDA_COGS': -1.0,
    'Liability_to_Asset': 0.014456530474395279,
    'Cash_Ratio': 0.0,
    'Operating_Cash_Flow_Ratio': -1.6687544574867383,
    'Fixed_to_Total_Assets': 0.0,
    'Receivables_Turnover': 0.0,
    'Cash_Return_on_Assets': -0.19486425602100674,
    'asset_growth': -0.42042134551628296,
    'revenue_growth': -0.9999988209037918
}

Q_HIGH = {
    'roa': 30.603699999999954,
    'EBITDA_Margin': 0.9265326146968058,
    'EBITDA_COGS': 3.4076340183772706,
    'Liability_to_Asset': 1.0884562083321074,
    'Cash_Ratio': 13.942473018371766,
    'Operating_Cash_Flow_Ratio': 6.665984293553062,
    'Fixed_to_Total_Assets': 0.9946892146230364,
    'Receivables_Turnover': 83.67812148949446,
    'Cash_Return_on_Assets': 0.27679296235321044,
    'asset_growth': 1.1635564364301927,
    'revenue_growth': 8.921571348563248
}