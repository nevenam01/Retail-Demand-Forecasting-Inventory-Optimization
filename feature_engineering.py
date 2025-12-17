import pandas as pd
import numpy as np
import time
import math
from datetime import datetime
import itertools
import vreme_feature_engineering as vreme_fe


def trim_by_quantiles(df, column_name, lower_quantile=0.05, upper_quantile=0.95):
    """
    Odseca vrednosti ispod i iznad datih kvantila iz specificirane kolone.
    """
    lower = df[column_name].quantile(lower_quantile)
    upper = df[column_name].quantile(upper_quantile)
    trimmed_df = df[(df[column_name] >= lower) & (df[column_name] <= upper)]
    return trimmed_df


def get_features(df):
    # sys.getsizeof() moze da nadje koliko memorije zauzima neki objekat
    # isprobati
    # moze jos da se relaksira dataset zbog brze izvrsavanja
    # mozda i na uzorku eksperimentisati, pa pustiti kad nadjemo sta hocemo na ceo dataset

    # dodavanje WeekStartDate kolone
    df['WeekStartDate'] = pd.to_datetime(df['date']) - pd.to_timedelta(pd.to_datetime(df['date']).dt.weekday, unit='d')

    # popunjavanje nedostajucih vrednosti
    df['onpromotion'] = df['onpromotion'].fillna(False)
    df['holiday_type'] = df['holiday_type'].fillna('Non existing')
    df['locale'] = df['locale'].fillna('Non existing')
    df['locale_name'] = df['locale_name'].fillna('Non existing')
    df['description'] = df['description'].fillna('Non existing')
    df['transferred'] = df['transferred'].fillna(False)
    nafta_medijana = df['dcoilwtico'].median(skipna=True)
    df['dcoilwtico'] = df['dcoilwtico'].fillna(nafta_medijana)
    df['transactions'] = df['transactions'].fillna(0)

    city = df['city'].iloc[0]
    state = df['state'].iloc[0]

    local_holiday = city + '_local_holiday'
    regional_holiday = state + '_regional_holiday'

    # dummy kolone za praznike
    df[local_holiday] = ((df['locale_name'] == city) & (df['locale'] == 'Local')).astype(int)
    df[regional_holiday]  = ((df['locale_name'] == state) & (df['locale'] == 'Regional')).astype(int)
    df['Ecuador_national_holiday']   = ((df['locale_name'] == 'Ecuador') & (df['locale'] == 'National')).astype(int)


    # izbacivanje nepotrebnih kolona za praznike
    df = df.drop(columns=['locale', 'locale_name', 'description'])

    # dobijanje dummy kolona za tipove praznika
    holiday_dummies = pd.get_dummies(df['holiday_type'], prefix='holiday')
    df = pd.concat([df, holiday_dummies], axis=1)
    location_mask = (df[local_holiday] == 1) | (df[regional_holiday] == 1) | (df['Ecuador_national_holiday'] == 1)
    for col in holiday_dummies.columns:
        df[col] = df[col].where(location_mask, 0)

    # rename kolone da bi nazivi kolona bili uniformni
    if 'holiday_Work Day' in df.columns:
        df.rename(columns={
            'holiday_Work Day': 'holiday_Work_Day'
        }, inplace=True)


    # grupisanje po WeekStartDate i item_nbr
    df_grouped = df.groupby(['WeekStartDate', 'item_nbr']).agg({
        'store_nbr': 'first',
        'unit_sales': 'sum',
        'onpromotion': 'max', # ako je bio na promociji u barem jednom danu, racunamo da je bio na promociji tokom cele nedelje
        # description necemo da uzimamo
        'transferred': 'min', # promene praznika, pomeraji se desavaju unutar nedelje najcesce (najverovatnije), i zato uzimamo min vrednost, ako bas svaki dan postoji da je pomeren praznik neka bude 1, ali uglavnom ce biti 0 i verovatno izbacujemo ovu varijablu
        #'dcoilwtico': 'mean', # prosecna cena nafte u toj nedelji
        'family': 'first', # mozda najbolje da ne uzimamo nego da koristimo kao filter, da uzmemo 1 ili 2 kategorije, zavisi koliko nam bude bilo potrebno, ima previse vrednosti
        #'class': '', # ne znamo sta je varijabla, necemo ni da uzimamo
        'perishable': 'first', # ovo je najbolje uzeti prvi jer je pretpostavka da ce za svaki item biti isti perishable, da se nece menjati na nivou itema u nedelji, a first bi trebalo da je brze od min i max npr
        'transactions': 'sum', # ovo je na nivou prodavnice, zato mislim da mozda treba i izbaciti varijablu jer ce biti ista za svaku nedelju, ali mozda moze posluziti za razlicite nedelje kada se saberu sve transakcije
        # npr neki proizvod ce se prodavati ako je bilo vise transakcija u toj nedelji nego obicno
        'city': 'first',
        'state': 'first'
    }).reset_index()

    # grupisanje samo po WeekStartDate kolonama
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
    # spajanje dva grupisana dataseta u jedan
    df_grouped = df_grouped.merge(df_datumi, on='WeekStartDate', how='left')

    # !!! KREIRANJE IZLAZNE VARIJABLE !!!
    df_grouped['Ordered'] = 1

    # !!! DODAVANJE SVIH KOMBINACIJA WEEKSTARTDATE I ITEM_NBR U DATAFRAME
    sve_nedelje = df_grouped['WeekStartDate'].unique()
    svi_proizvodi = df_grouped['item_nbr'].unique()

    sve_kombinacije = pd.DataFrame(list(itertools.product(sve_nedelje, svi_proizvodi)),
                                    columns=['WeekStartDate', 'item_nbr'])

    # spajanje onih kombinacija koje su se desile sa kombinacijama koje nisu
    df_popunjen = sve_kombinacije.merge(df_grouped, on=['WeekStartDate', 'item_nbr'], how='left')
    # oznacavamo kada se nije desila prodaja
    df_popunjen['Ordered'] = df_popunjen['Ordered'].fillna(0)

    # popunjavanje nedostajucih vrednosti novog dataframea sa svim kombinacijama
    for col in ['store_nbr', 'city', 'state']:
        df_popunjen[col] = df_popunjen[col].ffill().bfill()

    df_popunjen['unit_sales'] = df_popunjen['unit_sales'].fillna(0)

    for col in ['transferred', 'dcoilwtico', 'transactions', local_holiday,
                regional_holiday, 'Ecuador_national_holiday', 'holiday_Additional', 
                'holiday_Bridge', 'holiday_Event', 'holiday_Holiday',
                'holiday_Transfer', 'holiday_Work_Day']:
        df_popunjen[col] = df_popunjen.groupby('WeekStartDate')[col].ffill().bfill()
    
    df_popunjen['family'] = df_popunjen.groupby('item_nbr')['family'].ffill().bfill()
    df_popunjen['perishable'] = df_popunjen.groupby('item_nbr')['perishable'].ffill().bfill()
    df_popunjen['onpromotion'] = df_popunjen.groupby(['WeekStartDate', 'item_nbr'])['onpromotion'].ffill().bfill()

    # izbacivanje nepotrebnih kolona jer svuda imaju istu vrednost
    df_popunjen = df_popunjen.drop(columns=['city', 'state'])

    # kreiranje lagova za izlaznu varijablu
    df_popunjen = df_popunjen.sort_values(['item_nbr', 'WeekStartDate'])
    for lag in range(1,13):
        col_name = f"Ordered_lag_{lag}"
        df_popunjen[col_name] = df_popunjen.groupby('item_nbr')['Ordered'].shift(lag).fillna(0)

    # kreiranje lagova za unit_sales
    df_popunjen = df_popunjen.sort_values(['item_nbr', 'WeekStartDate'])
    for lag in range(1,13):
        col_name = f"unit_sales_lag_{lag}"
        df_popunjen[col_name] = df_popunjen.groupby(['item_nbr'])['unit_sales'].shift(lag).fillna(0)


    # kreiranje pomocnog dataframe-a za naftu jer se radi samo sa datumima a ne sa itemima
    # ovo drop duplicates je veoma bitno jer znaci da necemo imati 2 puta isti datum pa mozemo raditi bez ikakvog grupisanja
    df_oil = df_popunjen[['WeekStartDate', 'dcoilwtico']].drop_duplicates().sort_values('WeekStartDate')

    df_oil = (
        df_popunjen[['WeekStartDate', 'dcoilwtico']]
        .drop_duplicates()
        .sort_values('WeekStartDate')
        .reset_index(drop=True)
    )
    # kreiranje lagova u pomocnom dataframe-u da bi se lakse spojili sa originalnim dataframeom
    for lag in range(1, 13):
        df_oil[f'dcoilwtico_lag_{lag}'] = df_oil['dcoilwtico'].shift(lag).fillna(0)

    # dodavanje lagova u df_popunjen pomocu mapiranja jer join ne radi kako treba zbog velike kolicine podataka
    for lag in range(1, 13):
        lag_col = f'dcoilwtico_lag_{lag}'
        mapping = pd.Series(df_oil[lag_col].values, index=df_oil['WeekStartDate'])
        df_popunjen[lag_col] = df_popunjen['WeekStartDate'].map(mapping)

    df_oil = df_oil.sort_values('WeekStartDate')  # obavezno sortiranje

    rolling_windows = [4, 8, 12]
    # kreiranje rolling means kolona
    for win in rolling_windows:
        df_oil[f'dcoilwtico_rolling_mean_{win}'] = (
            df_oil['dcoilwtico'].rolling(window=win, min_periods=1).mean()
        )
    # dodavanja rolling means kolona u df_popunjen
    for win in rolling_windows:
        roll_col = f'dcoilwtico_rolling_mean_{win}'
        mapping = pd.Series(df_oil[roll_col].values, index=df_oil['WeekStartDate'])
        df_popunjen[roll_col] = df_popunjen['WeekStartDate'].map(mapping)


    df_popunjen = df_popunjen.sort_values(['item_nbr', 'WeekStartDate'])

    # Pronađi sve kolone koje sadrže 'holiday' u imenu
    holiday_cols = [col for col in df_popunjen.columns if 'holiday' in col]

    # Napravi lag_1 kolone grupisano po item_nbr
    for col in holiday_cols:
        lag_col = f"{col}_lag_1"
        df_popunjen[lag_col] = (
            df_popunjen.groupby('item_nbr')[col].shift(1).fillna(0)
        )

    # sortiranje vrednosti po itemu i datumu jer se radi sa item/datum varijablama
    df_popunjen = df_popunjen.sort_values(['item_nbr', 'WeekStartDate'])

    rolling_windows = [4, 8, 12]
    target_cols = ['Ordered', 'unit_sales']
    # kreiranje rolling means za gore pomenute kolone
    for col in target_cols:
        for win in rolling_windows:
            new_col = f"{col}_rolling_mean_{win}"
            df_popunjen[new_col] = (
                df_popunjen
                .groupby('item_nbr')[col]
                .transform(lambda x: x.rolling(window=win, min_periods=1).mean())
            )

    # kreiranje kolone WeekNumber (1-52) koja pomaze pri funkcijama za sezonalitet
    df_popunjen['WeekNumber'] = df_popunjen['WeekStartDate'].dt.isocalendar().week
    
    uglovi = [0.5, 1, 2, 4] # sa koliko mnozimo pi
    # kreiranje sin i cos funkcija za sezonalitet pomocu fourier transformacije
    for ugao in uglovi:
        sin_col = f"sin_{ugao}_pi"
        cos_col = f"cos_{ugao}_pi"

        df_popunjen[sin_col] = np.sin(ugao * np.pi * df_popunjen['WeekNumber'] / 52) # / 52 jer ima 52 nedelje u godini
        df_popunjen[cos_col] = np.cos(ugao * np.pi * df_popunjen['WeekNumber'] / 52) # / 52 jer ima 52 nedelje u godini


    # dodavanje vremenske prognoze
    df_popunjen = vreme_fe.add_vreme(df_popunjen)
    # kreiranje OUTPUT VARIJABLE

    # nema popunjavanja nulom
    df_popunjen['Ordered_next_week'] = df_popunjen.groupby('item_nbr')['Ordered'].shift(-1)
    df_popunjen = df_popunjen.dropna(subset=['Ordered_next_week'])

    # Lista kolona za popunjavanje
    cols_to_fill = [
        'MaxTemp_max',
        'MaxTemp_avg',
        'MinTemp_min',
        'MinTemp_avg',
        'BadWeather_days',
        'Cloudy_days',
        'Sunny_days'
    ]

    # Sortiraj po datumu
    df_popunjen = df_popunjen.sort_values('WeekStartDate')

    # Popuni nedostajuće vrednosti redom: ffill pa bfill
    for col in cols_to_fill:
        df_popunjen[col] = df_popunjen[col].ffill().bfill()

    df_popunjen = df_popunjen.sort_values(['item_nbr', 'WeekStartDate'])

    trimmed_df = trim_by_quantiles(df, 'unit_sales', 0.03, 0.9)
    trimmed_df = trim_by_quantiles(trimmed_df, 'MaxTemp_max', 0.01, 0.97)
    trimmed_df = trim_by_quantiles(trimmed_df, 'MaxTemp_avg', 0.02, 0.97)
    trimmed_df = trim_by_quantiles(trimmed_df, 'MinTemp_avg', 0.1, 1)
    trimmed_df = trim_by_quantiles(trimmed_df, 'transactions', 0, 0.99)

    return trimmed_df, city

    # poslednji red kod shifta -1 da izbacimo