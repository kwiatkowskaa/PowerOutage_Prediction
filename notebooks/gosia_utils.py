import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.express as px
import geopandas as gpd
import geodatasets
import os
from matplotlib.colors import ListedColormap, Normalize
from statsmodels.tsa.stattools import grangercausalitytests, ccf
from datetime import datetime
import warnings




def plot_event_counts(df):
    event_counts = df['EVENT_TYPE'].value_counts()

    year = int(df[['YEAR']].iloc[0,0])

    plt.figure(figsize=(12, 6))
    sns.barplot(x=event_counts.index, y=event_counts.values, palette='viridis')
    
    plt.title(f'Najczęściej raportowane katastrofy pogodowe w {year} roku', fontsize=14)
    plt.xlabel('Typ katastrofy', fontsize=12)
    plt.ylabel('Liczba zgłoszeń', fontsize=12)
    plt.xticks(rotation=90)
    
    plt.show()

    

def plot_event_trends(df):
    df['BEGIN_DATE_TIME'] = pd.to_datetime(df['BEGIN_DATE_TIME'])
    df['YEAR_MONTH'] = df['BEGIN_DATE_TIME'].dt.to_period('M')

    event_types = ['Thunderstorm Wind', 'Hail', 'Flash Flood']
    event_counts = {}

    for event in event_types:
        event_df = df[df['EVENT_TYPE'] == event]
        event_counts[event] = event_df['YEAR_MONTH'].value_counts().sort_index()

    plt.figure(figsize=(12, 6))

    markers = {'Thunderstorm Wind': 'o', 'Hail': 's', 'Flash Flood': 'D'}
    
    for event, counts in event_counts.items():
        plt.plot(counts.index.astype(str), counts.values, label=event, marker=markers[event])

    plt.title('Liczba zdarzeń pogodowych w czasie', fontsize=14)
    plt.xlabel('Miesiąc', fontsize=12)
    plt.ylabel('Liczba zdarzeń', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)

    plt.show()


def plot_state_event_counts(df):
    event_types = ['Thunderstorm Wind', 'Hail', 'Flash Flood']
    df_filtered = df[df['EVENT_TYPE'].isin(event_types)]

    state_counts = df_filtered.groupby(['STATE', 'EVENT_TYPE']).size().unstack().fillna(0)

    plt.figure(figsize=(12, 6))
    state_counts.sort_values(by=event_types, ascending=False).plot(kind='bar', stacked=True, figsize=(12, 6), colormap='viridis')
    plt.title('Liczba zgłoszeń Thunderstorm Wind, Hail i Flash Flood w podziale na stany', fontsize=14)
    plt.xlabel('Stan', fontsize=12)
    plt.ylabel('Liczba zgłoszeń', fontsize=12)
    plt.xticks(rotation=90)
    plt.legend(title="Typ katastrofy")
    plt.grid(axis='y')
    plt.show()

def plot_county_event_counts(df):
    event_types = ['Thunderstorm Wind', 'Hail', 'Flash Flood']
    df_filtered = df[df['EVENT_TYPE'].isin(event_types)]

    df_filtered['COUNTY_STATE'] = df_filtered['CZ_NAME'] + " (" + df_filtered['STATE'] + ")"

    county_counts = df_filtered.groupby(['COUNTY_STATE', 'EVENT_TYPE']).size().unstack().fillna(0)

    plt.figure(figsize=(12, 6))
    county_counts.sort_values(by=event_types, ascending=False).head(20).plot(kind='bar', stacked=True, figsize=(12, 6), colormap='plasma')
    plt.title('Liczba zgłoszeń Thunderstorm Wind, Hail i Flash Flood w podziale na hrabstwa (Top 20)', fontsize=14)
    plt.xlabel('Hrabstwo (Stan)', fontsize=12)
    plt.ylabel('Liczba zgłoszeń', fontsize=12)
    plt.xticks(rotation=90)
    plt.legend(title="Typ katastrofy")
    plt.grid(axis='y')
    plt.show()


def plot_disaster_map(df):
    
    df = df[df['BEGIN_LON'].between(-125, -66) & df['BEGIN_LAT'].between(24, 49)]
    
    event_types = ['Thunderstorm Wind', 'Hail', 'Flash Flood']
    df_filtered = df[df['EVENT_TYPE'].isin(event_types)]

    fig = px.scatter_mapbox(
        df_filtered,
        lat="BEGIN_LAT",
        lon="BEGIN_LON",
        color="EVENT_TYPE",
        title="Rozmieszczenie katastrof pogodowych w USA",
        mapbox_style="carto-positron",
        zoom=3,
        hover_data=["EVENT_TYPE", "STATE", "CZ_NAME"]
    )

    fig.show()

def plot_event_seasonality(df):

    df['BEGIN_DATE_TIME'] = pd.to_datetime(df['BEGIN_DATE_TIME'])
    
    df['MONTH'] = df['BEGIN_DATE_TIME'].dt.month
    event_types = ['Thunderstorm Wind', 'Hail', 'Flash Flood']
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='MONTH', y='EVENT_TYPE', data=df[df['EVENT_TYPE'].isin(event_types)])
    
    plt.title('Sezonowość Thunderstorm Wind, Hail i Flash Flood')
    plt.xlabel('Miesiąc')
    plt.ylabel('Typ katastrofy')
    plt.xticks(range(12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.grid(True)
    plt.show()

def plot_customer_outages_seasonality(df):

    df['run_start_time'] = pd.to_datetime(df['run_start_time'], format='%Y-%m-%d %H:%M:%S')

    df['MONTH'] = df['run_start_time'].dt.month

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='MONTH', y='customers_out', data=df)

    plt.title('Sezonowość liczby awarii prądu (customers_out)')
    plt.xlabel('Miesiąc')
    plt.ylabel('Liczba klientów bez prądu')
    plt.xticks(range(12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.grid(True)
    plt.yscale("log")
    plt.show()

def plot_monthly_avg_outages(df):
    df['run_start_time'] = pd.to_datetime(df['run_start_time'], format='%Y-%m-%d %H:%M:%S')

    df['MONTH'] = df['run_start_time'].dt.month
    monthly_avg_outages = df.groupby('MONTH')['customers_out'].mean()
    
    plt.figure(figsize=(12, 6))
    plt.bar(monthly_avg_outages.index, monthly_avg_outages.values, color='blue')
    plt.xlabel('Miesiąc')
    plt.ylabel('Średnia liczba klientów bez prądu')
    plt.title('Średnia liczba klientów bez prądu w podziale na miesiące')
    plt.xticks(range(1, 13))
    plt.grid(True)
    plt.show()

def plot_monthly_damage(df):
    df['MONTH'] = df['BEGIN_DATE_TIME'].dt.month
    monthly_damage = df.groupby('MONTH')['DAMAGE_PROPERTY'].sum()
    
    plt.figure(figsize=(12, 6))
    plt.plot(monthly_damage.index, monthly_damage.values, label='Straty materialne', color='orange', marker='o')
    plt.xlabel('Miesiąc')
    plt.ylabel('Straty materialne ($)')
    plt.title('Straty materialne w podziale na miesiące')
    plt.xticks(range(1, 13))
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_top_damage_events(df):
    top_damage = df.groupby('EVENT_TYPE')['DAMAGE_PROPERTY'].sum().sort_values(ascending=False).head(10)
    
    plt.figure(figsize=(12, 6))
    plt.barh(top_damage.index, top_damage.values, color='orange')
    plt.xlabel('Straty materialne ($)')
    plt.ylabel('Typ zdarzenia')
    plt.title('Top 10 kategorii katastrof z największymi stratami materialnymi')
    plt.gca().invert_yaxis()
    plt.show()

def make_ts_power(county,
                  start_year,
                  start_month,
                  start_day,
                  end_year,
                  end_month,
                  end_day,
                  data_directory = './data/data/eaglei_data'):


    df_list = []
    for year in range(start_year, end_year + 1):
        file_name = f"eaglei_outages_{year}.csv"
        file_path = os.path.join(data_directory, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_name} does not exist in the directory {data_directory}.")

        df = pd.read_csv(file_path)
        df['run_start_time'] = pd.to_datetime(df['run_start_time'])
        df.dropna(subset=['customers_out'],inplace=True)

        df_state = df[df['county'].str.upper()==county.upper()].copy(deep=True)
        df_state_ts_cus = df_state.groupby('run_start_time')['customers_out'].sum().reset_index()
        df_state_ts_cus.drop(df_state_ts_cus.index[-1], inplace=True)
        df_state_ts_cus.set_index('run_start_time', inplace=True)
        df_state_ts_cus.rename_axis('time', inplace=True)


        df_list.append(df_state_ts_cus)

    concat_df = pd.concat(df_list, ignore_index=False)
    start_date = pd.Timestamp(year=start_year, month=start_month, day=start_day)
    end_date = pd.Timestamp(year=end_year, month=end_month, day=end_day+1)
    df_state_ts_power = concat_df.loc[start_date:end_date].copy(deep=True)
    df_state_ts_power.drop(df_state_ts_power.index[-1], inplace=True)

    return df_state_ts_power


def make_ts_events(county, event_types, start_year, start_month, start_day, end_year, end_month, end_day, df):

    start_date = datetime(start_year, start_month, start_day)
    end_date = datetime(end_year, end_month, end_day, 23, 45) 
    time_index = pd.date_range(start=start_date, end=end_date, freq='15min')

    new_df = pd.DataFrame({'time': time_index})

    for event_type in event_types:
        new_df[f'event_count {event_type}'] = 0  

    avg_cols = ['injuries_direct_avg', 'injuries_indirect_avg', 'deaths_direct_avg', 'deaths_indirect_avg']
    for col in avg_cols:
        new_df[col] = 0.0

    df['BEGIN_DATETIME'] = pd.to_datetime(
        df['BEGIN_YEARMONTH'].astype(str) + df['BEGIN_DAY'].astype(str).str.zfill(2) +
        df['BEGIN_TIME'].astype(str).str.zfill(4), format='%Y%m%d%H%M'
    )
    df['END_DATETIME'] = pd.to_datetime(
        df['END_YEARMONTH'].astype(str) + df['END_DAY'].astype(str).str.zfill(2) +
        df['END_TIME'].astype(str).str.zfill(4), format='%Y%m%d%H%M'
    )

    filtered_df = df[
        (df['CZ_NAME'].str.upper() == county.upper()) &
        (df['EVENT_TYPE'].isin(event_types)) &
        (df['END_DATETIME'] >= start_date) &
        (df['BEGIN_DATETIME'] <= end_date)
    ].copy(deep=True)

    for event_type in event_types:
        event_subset = filtered_df[filtered_df['EVENT_TYPE'] == event_type]

        for _, row in event_subset.iterrows():
            event_start = row['BEGIN_DATETIME']
            event_end = row['END_DATETIME']

            event_start_rounded = event_start.round('15min')
            event_end_rounded = event_end.round('15min')

            start_idx = new_df['time'].searchsorted(event_start_rounded)
            end_idx = new_df['time'].searchsorted(event_end_rounded)

            if start_idx < len(new_df) and end_idx <= len(new_df):
                new_df.loc[start_idx:end_idx, f'event_count {event_type}'] += 1
                new_df.loc[start_idx:end_idx, 'injuries_direct_avg'] += row['INJURIES_DIRECT']
                new_df.loc[start_idx:end_idx, 'injuries_indirect_avg'] += row['INJURIES_INDIRECT']
                new_df.loc[start_idx:end_idx, 'deaths_direct_avg'] += row['DEATHS_DIRECT']
                new_df.loc[start_idx:end_idx, 'deaths_indirect_avg'] += row['DEATHS_INDIRECT']

    total_events = new_df[[f'event_count {event_type}' for event_type in event_types]].sum(axis=1)
    for col in avg_cols:
        new_df[col] = new_df[col] / total_events.replace(0, 1) 

    new_df['YEAR'] = new_df['time'].dt.year
    new_df['MONTH'] = new_df['time'].dt.month
    new_df['DAY'] = new_df['time'].dt.day

    cols_order = ['YEAR', 'MONTH', 'DAY', 'time'] + avg_cols + [col for col in new_df.columns if col not in ['YEAR', 'MONTH', 'DAY', 'time'] + avg_cols]
    new_df = new_df[cols_order]

    return new_df
def aggregate_ts(df, agg_type):

    df.index = pd.to_datetime(df.index)

    if agg_type == 'hour':
        df_agg = df.groupby(pd.Grouper(freq='h')).mean()
    elif agg_type == 'day':
        df_agg = df.groupby(pd.Grouper(freq='D')).mean()
    else:
        raise ValueError("Invalid aggregation type. Use 'hour' or 'day'.")

    df_agg.fillna(0, inplace=True)

    return df_agg

def combine_agg_ts(county,
                   start_year,
                   start_month,
                   start_day,
                   end_year,
                   end_month,
                   end_day,
                   data_directory_power = './data/data/eaglei_data',
                   data_directory_events = './data/data/NOAA_StormEvents'):


    df_state_ts_power = make_ts_power(county = county,
                                      start_year = start_year,
                                      start_month = start_month,
                                      start_day = start_day,
                                      end_year = end_year,
                                      end_month = end_month,
                                      end_day = end_day,
                                      data_directory = data_directory_power)

    df_state_ts_power_hr = aggregate_ts(df_state_ts_power, 'hour')
    df_state_ts_power_day = aggregate_ts(df_state_ts_power, 'day')



    df_events = pd.read_csv(os.path.join(data_directory_events, "StormEvents_2014_2024.csv"))
    df_state_events=df_events[df_events['CZ_NAME'].str.upper()==county.upper()].copy(deep=True)
    event_types_state = list(df_state_events['EVENT_TYPE'].unique())
    df_state_ts_events = make_ts_events(county = county,
                                        event_types= event_types_state,
                                        start_year = start_year,
                                        start_month = start_month,
                                        start_day = start_day,
                                        end_year = end_year,
                                        end_month = end_month,
                                        end_day = end_day,
                                        df=df_events)
    
    print(df_state_ts_events[df_state_ts_events['event_count Wildfire'] > 1])

    df_state_ts_events['time'] = pd.to_datetime(df_state_ts_events['time'])
    df_state_ts_events.set_index('time', inplace=True)
    df_state_ts_events.drop(columns=['YEAR', 'DAY', 'MONTH'], inplace=True)

    df_state_ts_events_hr = aggregate_ts(df_state_ts_events, 'hour')
    df_state_ts_events_day = aggregate_ts(df_state_ts_events, 'day')

    df_state_ts_comb_hr = pd.merge(df_state_ts_events_hr, df_state_ts_power_hr, left_index=True, right_index=True)
    df_state_ts_comb_day = pd.merge(df_state_ts_events_day, df_state_ts_power_day, left_index=True, right_index=True)

    return df_state_ts_comb_hr, df_state_ts_comb_day