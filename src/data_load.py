import pandas as pd
import numpy as np
from pathlib import Path


# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'


def load_data(store_number):
    """
    Loads and merges all data sources for a specific store.
    
    Args:
        store_number: Store ID to filter data for
        
    Returns:
        DataFrame with merged data from all sources
    """
    # Load all data files
    df = pd.read_csv(DATA_DIR / 'train.csv')
    df_holidays = pd.read_csv(DATA_DIR / 'holidays_events.csv')
    df_oil = pd.read_csv(DATA_DIR / 'oil.csv')
    df_items = pd.read_csv(DATA_DIR / 'items.csv')
    df_stores = pd.read_csv(DATA_DIR / 'stores.csv')
    df_transactions = pd.read_csv(DATA_DIR / 'transactions.csv')

    # Filter for the selected store
    df_store = df[df['store_nbr'] == store_number]
    
    # Merge with other dataframes
    df_store = df_store.merge(df_holidays, on='date', how='left')
    df_store = df_store.merge(df_oil, on='date', how='left')
    df_store = df_store.merge(df_items, on='item_nbr', how='left')
    df_store = df_store.merge(df_transactions, on=['store_nbr', 'date'], how='left')
    df_store = df_store.merge(df_stores, on='store_nbr', how='left')

    # Drop columns that are constant for a single store
    df_store = df_store.drop(columns=['type_y', 'cluster'])

    # Rename type column to avoid confusion (exists in both holidays and stores)
    df_store.rename(columns={'type_x': 'holiday_type'}, inplace=True)

    # Convert store_nbr to int8 for memory efficiency
    df_store['store_nbr'] = df_store['store_nbr'].astype('int8')
    df_store['date'] = pd.to_datetime(df_store['date'])

    # Convert promotion column from string to boolean
    df_store['onpromotion'] = df_store['onpromotion'].replace({
        'False': False,
        'True': True
    })
    
    return df_store
