import pandas as pd
import numpy as np
from pathlib import Path


# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'


def add_vreme(df_features):
    """
    Adds weather features to the dataset by merging with scraped weather data.
    
    Args:
        df_features: DataFrame with WeekStartDate column
        
    Returns:
        DataFrame with added weather features:
        - MaxTemp_max, MaxTemp_avg: Max temperature stats
        - MinTemp_min, MinTemp_avg: Min temperature stats  
        - BadWeather_days, Cloudy_days, Sunny_days: Weather condition counts
    """
    # Load weather data
    df = pd.read_csv(DATA_DIR / 'vreme_latacunga.csv')

    # Create WeekStartDate (Monday of each week)
    df['WeekStartDate'] = pd.to_datetime(df['Datum']) - pd.to_timedelta(
        pd.to_datetime(df['Datum']).dt.weekday, unit='d'
    )

    # Unify weather descriptions
    df['Opis'] = df['Opis'].replace(['Mostly Cloudy', 'Partly Cloudy'], 'Cloudy')
    df['Opis'] = df['Opis'].replace(['Tornado', 'Foggy', 'Scattered Showers'], 'Bad Weather')
    df['Opis'] = df['Opis'].replace('Mostly Sunny', 'Sunny')

    # Create dummy columns for weather types
    df = pd.get_dummies(df, columns=['Opis'])

    # Drop unnecessary column
    df.drop(columns='Padavine (in)', inplace=True)

    # Aggregate by week
    df_grouped = df.groupby('WeekStartDate').agg({
        'MaxTemp': ['max', 'mean'],
        'MinTemp': ['min', 'mean'],
        'Opis_Bad Weather': 'sum',
        'Opis_Cloudy': 'sum',
        'Opis_Sunny': 'sum'
    })

    # Rename columns
    df_grouped.columns = ['{}_{}'.format(col[0], col[1]) if col[1] else col[0] 
                          for col in df_grouped.columns]
    df_grouped.rename(columns={
        'MaxTemp_max': 'MaxTemp_max',
        'MaxTemp_mean': 'MaxTemp_avg',
        'MinTemp_min': 'MinTemp_min',
        'MinTemp_mean': 'MinTemp_avg',
        'Opis_Bad Weather_sum': 'BadWeather_days',
        'Opis_Cloudy_sum': 'Cloudy_days',
        'Opis_Sunny_sum': 'Sunny_days'
    }, inplace=True)
    
    df_grouped = df_grouped.reset_index()

    # Merge with features dataframe
    df_features = df_features.merge(df_grouped, on='WeekStartDate', how='left')

    return df_features
