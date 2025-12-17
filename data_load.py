import pandas as pd
import numpy as np


def load_data(store_number):

    # data loading
    df = pd.read_csv('train.csv', parse_dates=['date'])
    df_items = pd.read_csv('items.csv')
    df_stores = pd.read_csv('stores.csv')
    df_holidays = pd.read_csv('holidays_events.csv', parse_dates=['date'])
    df_oil = pd.read_csv('oil.csv', parse_dates=['date'])
    df_transactions = pd.read_csv('transactions.csv', parse_dates=['date'])


    # filtriranje zeljene prodavnice
    df_st_13 = df[df['store_nbr'] == store_number]
    #df_st_13['store_nbr'] = df_st_13['store_nbr'].astype(Int8)
    # spajanje sa ostalim dataframe-ovima
    df_st_13 = df_st_13.merge(df_holidays, on='date', how='left')
    df_st_13 = df_st_13.merge(df_oil, on='date', how='left')
    df_st_13 = df_st_13.merge(df_items, on='item_nbr', how='left')
    df_st_13 = df_st_13.merge(df_transactions, on=['store_nbr', 'date'], how='left')
    df_st_13 = df_st_13.merge(df_stores, on='store_nbr', how='left')

    # ove kolone se izbacuju jer ce uvek biti jedna te ista vrednost
    # city i state neka ostanu mozda pomognu za holiday ako je lokalni
    df_st_13 = df_st_13.drop(columns=['type_y', 'cluster'])

    # problem je kolona type koja postoji i u holidays datasetu i u stores datasetu, pa onda postoje type_x i type_y kolone
    df_st_13.rename(columns={
        'type_x': 'holiday_type'
    }, inplace=True)

    # konverzija store_nbr u int8 radi ustede prostora
    df_st_13['store_nbr'] = df_st_13['store_nbr'].astype('int8')
    df_st_13['date'] = pd.to_datetime(df_st_13['date'])

    # string u boolean
    df_st_13['onpromotion'].replace({
    'False': False,
    'True': True
    }, inplace=True)
    
    return df_st_13