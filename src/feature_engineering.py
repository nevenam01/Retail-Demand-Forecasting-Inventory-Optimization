import pandas as pd
import numpy as np
import itertools
from . import vreme_feature_engineering as vreme_fe


def get_features(df):
    """
    Main feature engineering pipeline. Takes raw daily data and returns 
    weekly aggregated features ready for model training.
    
    Args:
        df: DataFrame from data_load.load_data()
        
    Returns:
        tuple: (processed_df, city_name)
    """
    # Add WeekStartDate column for weekly aggregation
    df['WeekStartDate'] = pd.to_datetime(df['date']) - pd.to_timedelta(
        pd.to_datetime(df['date']).dt.weekday, unit='d'
    )

    # Fill missing values
    df['onpromotion'] = df['onpromotion'].fillna(False)
    df['holiday_type'] = df['holiday_type'].fillna('Non existing')
    df['locale'] = df['locale'].fillna('Non existing')
    df['locale_name'] = df['locale_name'].fillna('Non existing')
    df['description'] = df['description'].fillna('Non existing')
    df['transferred'] = df['transferred'].fillna(False)
    oil_median = df['dcoilwtico'].median(skipna=True)
    df['dcoilwtico'] = df['dcoilwtico'].fillna(oil_median)
    df['transactions'] = df['transactions'].fillna(0)

    city = df['city'].iloc[0]
    state = df['state'].iloc[0]

    local_holiday = city + '_local_holiday'
    regional_holiday = state + '_regional_holiday'

    # Create dummy columns for holiday locations
    df[local_holiday] = ((df['locale_name'] == city) & (df['locale'] == 'Local')).astype(int)
    df[regional_holiday] = ((df['locale_name'] == state) & (df['locale'] == 'Regional')).astype(int)
    df['Ecuador_national_holiday'] = ((df['locale_name'] == 'Ecuador') & (df['locale'] == 'National')).astype(int)

    # Drop unnecessary holiday columns
    df = df.drop(columns=['locale', 'locale_name', 'description'])

    # Create dummy columns for holiday types
    holiday_dummies = pd.get_dummies(df['holiday_type'], prefix='holiday')
    df = pd.concat([df, holiday_dummies], axis=1)
    location_mask = (df[local_holiday] == 1) | (df[regional_holiday] == 1) | (df['Ecuador_national_holiday'] == 1)
    for col in holiday_dummies.columns:
        df[col] = df[col].where(location_mask, 0)

    # Rename column for consistent naming
    if 'holiday_Work Day' in df.columns:
        df.rename(columns={'holiday_Work Day': 'holiday_Work_Day'}, inplace=True)

    # Group by week and item
    df_grouped = df.groupby(['WeekStartDate', 'item_nbr']).agg({
        'store_nbr': 'first',
        'unit_sales': 'sum',
        'onpromotion': 'max',
        'transferred': 'min',
        'family': 'first',
        'perishable': 'first',
        'transactions': 'sum',
        'city': 'first',
        'state': 'first'
    }).reset_index()

    # Group date-level features separately
    df_datumi = df.groupby('WeekStartDate').agg({
        'dcoilwtico': 'mean',
        local_holiday: 'max',
        regional_holiday: 'max',
        'Ecuador_national_holiday': 'max',
        'holiday_Additional': 'max',
        'holiday_Bridge': 'max',
        'holiday_Event': 'max',
        'holiday_Holiday': 'max',
        'holiday_Transfer': 'max',
        'holiday_Work_Day': 'max'
    })
    
    df_grouped = df_grouped.merge(df_datumi, on='WeekStartDate', how='left')

    # Create target variable (1 = product was sold that week)
    df_grouped['Ordered'] = 1

    # Create all week-item combinations (including weeks with no sales)
    all_weeks = df_grouped['WeekStartDate'].unique()
    all_items = df_grouped['item_nbr'].unique()

    all_combinations = pd.DataFrame(
        list(itertools.product(all_weeks, all_items)),
        columns=['WeekStartDate', 'item_nbr']
    )

    df_full = all_combinations.merge(df_grouped, on=['WeekStartDate', 'item_nbr'], how='left')
    df_full['Ordered'] = df_full['Ordered'].fillna(0)

    # Fill missing values
    for col in ['store_nbr', 'city', 'state']:
        df_full[col] = df_full[col].ffill().bfill()

    df_full['unit_sales'] = df_full['unit_sales'].fillna(0)

    for col in ['transferred', 'dcoilwtico', 'transactions', local_holiday,
                regional_holiday, 'Ecuador_national_holiday', 'holiday_Additional', 
                'holiday_Bridge', 'holiday_Event', 'holiday_Holiday',
                'holiday_Transfer', 'holiday_Work_Day']:
        df_full[col] = df_full.groupby('WeekStartDate')[col].ffill().bfill()
    
    df_full['family'] = df_full.groupby('item_nbr')['family'].ffill().bfill()
    df_full['perishable'] = df_full.groupby('item_nbr')['perishable'].ffill().bfill()
    df_full['onpromotion'] = df_full.groupby(['WeekStartDate', 'item_nbr'])['onpromotion'].ffill().bfill()

    # Drop constant columns
    df_full = df_full.drop(columns=['city', 'state'])

    # Create lag features for target (1-12 weeks)
    df_full = df_full.sort_values(['item_nbr', 'WeekStartDate'])
    for lag in range(1, 13):
        col_name = f"Ordered_lag_{lag}"
        df_full[col_name] = df_full.groupby('item_nbr')['Ordered'].shift(lag).fillna(0)

    # Create lag features for unit sales
    df_full = df_full.sort_values(['item_nbr', 'WeekStartDate'])
    for lag in range(1, 13):
        col_name = f"unit_sales_lag_{lag}"
        df_full[col_name] = df_full.groupby('item_nbr')['unit_sales'].shift(lag).fillna(0)

    # Create oil price lags (date-level only)
    df_oil = (
        df_full[['WeekStartDate', 'dcoilwtico']]
        .drop_duplicates()
        .sort_values('WeekStartDate')
        .reset_index(drop=True)
    )
    
    for lag in range(1, 13):
        df_oil[f'dcoilwtico_lag_{lag}'] = df_oil['dcoilwtico'].shift(lag).fillna(0)

    for lag in range(1, 13):
        lag_col = f'dcoilwtico_lag_{lag}'
        mapping = pd.Series(df_oil[lag_col].values, index=df_oil['WeekStartDate'])
        df_full[lag_col] = df_full['WeekStartDate'].map(mapping)

    df_oil = df_oil.sort_values('WeekStartDate')

    # Create rolling means for oil prices
    rolling_windows = [4, 8, 12]
    for win in rolling_windows:
        df_oil[f'dcoilwtico_rolling_mean_{win}'] = (
            df_oil['dcoilwtico'].rolling(window=win, min_periods=1).mean()
        )
    
    for win in rolling_windows:
        roll_col = f'dcoilwtico_rolling_mean_{win}'
        mapping = pd.Series(df_oil[roll_col].values, index=df_oil['WeekStartDate'])
        df_full[roll_col] = df_full['WeekStartDate'].map(mapping)

    df_full = df_full.sort_values(['item_nbr', 'WeekStartDate'])

    # Create lag features for holidays
    holiday_cols = [col for col in df_full.columns if 'holiday' in col]
    for col in holiday_cols:
        lag_col = f"{col}_lag_1"
        df_full[lag_col] = df_full.groupby('item_nbr')[col].shift(1).fillna(0)

    df_full = df_full.sort_values(['item_nbr', 'WeekStartDate'])

    # Create rolling means for target variables
    rolling_windows = [4, 8, 12]
    target_cols = ['Ordered', 'unit_sales']
    for col in target_cols:
        for win in rolling_windows:
            new_col = f"{col}_rolling_mean_{win}"
            df_full[new_col] = (
                df_full
                .groupby('item_nbr')[col]
                .transform(lambda x: x.rolling(window=win, min_periods=1).mean())
            )

    # Create seasonality features (Fourier transforms)
    df_full['WeekNumber'] = df_full['WeekStartDate'].dt.isocalendar().week
    
    frequencies = [0.5, 1, 2, 4]
    for freq in frequencies:
        sin_col = f"sin_{freq}_pi"
        cos_col = f"cos_{freq}_pi"
        df_full[sin_col] = np.sin(freq * np.pi * df_full['WeekNumber'] / 52)
        df_full[cos_col] = np.cos(freq * np.pi * df_full['WeekNumber'] / 52)

    # Add weather features
    df_full = vreme_fe.add_vreme(df_full)

    # Create target: whether product will be ordered next week
    df_full['Ordered_next_week'] = df_full.groupby('item_nbr')['Ordered'].shift(-1)
    df_full = df_full.dropna(subset=['Ordered_next_week'])

    # Fill missing weather values
    cols_to_fill = [
        'MaxTemp_max', 'MaxTemp_avg', 'MinTemp_min', 'MinTemp_avg',
        'BadWeather_days', 'Cloudy_days', 'Sunny_days'
    ]
    df_full = df_full.sort_values('WeekStartDate')
    for col in cols_to_fill:
        df_full[col] = df_full[col].ffill().bfill()

    df_full = df_full.sort_values(['item_nbr', 'WeekStartDate'])
    
    return df_full, city
