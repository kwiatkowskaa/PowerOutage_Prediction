import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from itertools import product


def summarize_overall_missing(dfs, names=None):
    summaries = []
    
    for i, df in enumerate(dfs):
        dataset_name = names[i] if names else f"stormEvents_{2014+i}"
        
        missing_count = df['MAGNITUDE'].isna().sum()
        present_count = df['MAGNITUDE'].notna().sum()
        total_count = len(df)
        
        summaries.append({
            'DATASET': dataset_name,
            'MISSING_MAGNITUDE': missing_count,
            'PRESENT_MAGNITUDE': present_count,
            'MISSING_PERCENT': round((missing_count / total_count) * 100, 2)
        })
    
    return pd.DataFrame(summaries)



def filter_counties(outages_dfs, mcc_df):
    """
    Delete counties, that have never apperar in any outages datasets from MCC
    
    :param outages_dfs: DataFrame List containing outage reports
    :param mcc_df: DataFrame containing County_FIPS and Customers
    :return: Filteres DataFrame MCC
    """
    appeared_fips = set()

    for df in outages_dfs:
        if 'fips_code' in df.columns:
            appeared_fips.update(df['fips_code'].dropna().astype(str).unique())

    mcc_df['County_FIPS'] = mcc_df['County_FIPS'].astype(str)

    filtered_mcc_df = mcc_df[mcc_df['County_FIPS'].isin(appeared_fips)]

    return filtered_mcc_df



def plot_reporting_distribution(df, time_column, expected_reports):
    """
    Plots the distribution of reporting completeness across counties.
    
    :param df: DataFrame containing outage reports
    :param expected_reports: Expected number of reports per county
    """
    year = pd.to_datetime(df[time_column].iloc[0]).year


    report_counts = df.groupby('fips_code')[time_column].count()
    reporting_percentage = (report_counts / expected_reports) * 100

    plt.figure(figsize=(10, 4))
    plt.hist(reporting_percentage, bins=50, edgecolor='black')
    plt.xlim(0,100)
    plt.xlabel("Percentage of Expected Reports")
    plt.ylabel("Number of Counties")
    plt.title(f"Distribution of Reporting Completeness in {year}")
    plt.show()



def aggregate_daily_outages(df):
    """
    Aggregates outage data by day and county, computing the daily average of customers_out.
    
    :param df: DataFrame containing outage reports
    :return: Aggregated DataFrame with daily averages per county
    """
    df['run_start_time'] = pd.to_datetime(df['run_start_time'])
    df['date'] = df['run_start_time'].dt.date
    
    aggregated_df = df.groupby(["date", 'fips_code', 'county', 'state'])['customers_out'].mean().reset_index()
    
    return aggregated_df




def plot_removal_effect(df, time_column, expected_reports, thresholds):
    """
    Plots the percentage of counties removed as a function of the threshold.
    
    :param df: DataFrame containing outage reports
    :param time_column: Column name for timestamps
    :param expected_reports: Expected number of reports per county
    :param thresholds: List of threshold values to test
    """
    year = pd.to_datetime(df[time_column].iloc[0]).year

    report_counts = df.groupby('fips_code')[time_column].count()
    reporting_percentage = (report_counts / expected_reports) * 100
    
    removal_percentages = []
    total_counties = len(reporting_percentage)
    
    for threshold in thresholds:
        removed_counties = (reporting_percentage < threshold).sum()
        removal_percentages.append((removed_counties / total_counties) * 100)
    
    plt.figure(figsize=(10, 4))
    plt.plot(thresholds, removal_percentages, marker='o')
    plt.xlabel("Threshold (%)")
    plt.ylabel("Percentage of Counties Removed")
    plt.title(f"Effect of Threshold on County Removal in {year}")
    plt.grid()
    plt.show()




def filter_low_reporting_counties(df, time_column, expected_reports, threshold):
    """
    Removes counties that have reported below a given threshold of expected reports.
    
    :param df: DataFrame containing outage reports
    :param time_column: Column name for timestamps
    :param expected_reports: Expected number of reports per county
    :param threshold: Minimum percentage of expected reports required to keep a county
    :return: Filtered DataFrame
    """
    report_counts = df.groupby('fips_code')[time_column].count()
    reporting_percentage = (report_counts / expected_reports) * 100
    
    valid_fips = reporting_percentage[reporting_percentage >= threshold].index
    filtered_df = df[df['fips_code'].isin(valid_fips)]
    
    return filtered_df



def plot_reporting_distribution(df, time_column, expected_reports):
    """
    Plots the distribution of reporting completeness across counties.
    
    :param df: DataFrame containing outage reports
    :param expected_reports: Expected number of reports per county
    """
    year = pd.to_datetime(df[time_column].iloc[0]).year


    report_counts = df.groupby('fips_code')[time_column].count()
    reporting_percentage = (report_counts / expected_reports) * 100

    plt.figure(figsize=(10, 4))
    plt.hist(reporting_percentage, bins=50, edgecolor='black')
    plt.xlim(0,100)
    plt.xlabel("Percentage of Expected Reports")
    plt.ylabel("Number of Counties")
    plt.title(f"Distribution of Reporting Completeness in {year}")
    plt.show()



def add_valid_data_flag(df, time_col, expected_reports):
    """
    Adds a binary column `'valid_data_flag'`, which indicates whether a given `fips_code` has ≥90% data completeness.  

    **Parameters:**  
    - `df` (*pd.DataFrame*): The original DataFrame containing the data.  
    - `time_col` (*str*): The name of the column containing report timestamps.  
    - `expected_reports` (*int*): The expected number of reports for each `fips_code` per year.  

    **Returns:**  
    - *pd.DataFrame*: A copy of the DataFrame with the added `'valid_data_flag'` column.  
    """
    
    report_counts = df.groupby('fips_code')[time_col].count()
    reporting_percentage = (report_counts / expected_reports) * 100
    valid_data_flag = (reporting_percentage >= 90).astype(int)

    df = df.copy()
    df['valid_data_flag'] = df['fips_code'].map(valid_data_flag)

    return df



def fill_missing_dates(df):
    """
    Ensures that every 'fips_code' has a complete set of daily records for the given year.
    
    Parameters:
    df (pd.DataFrame): Original dataset with columns ['date', 'fips_code', 'county', 'state', 'customers_out', 'valid_data_flag'].

    Returns:
    pd.DataFrame: Updated DataFrame with missing dates filled and 'customers_out' set to 0 for new records.
    """
    year = pd.to_datetime(df['date'].iloc[0]).year

    all_dates = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31', freq='D')
    unique_fips = df[['fips_code', 'county', 'state', 'valid_data_flag']].drop_duplicates()
    
    full_date_fips = pd.DataFrame(product(unique_fips['fips_code'], all_dates), columns=['fips_code', 'date'])
    full_date_fips['date'] = pd.to_datetime(full_date_fips['date'])
    df['date'] = pd.to_datetime(df['date'])
    full_data = full_date_fips.merge(unique_fips, on='fips_code', how='left')
    
    df = full_data.merge(df, on=['fips_code', 'date', 'county', 'state', 'valid_data_flag'], how='left')
    df['customers_out'] = df['customers_out'].fillna(0)

    return df




def calculate_percent_customers_out(df, mcc):
    """
    Creates a new column `percent_customers_out` in DataFrame `df`, calculated as the ratio of 
    `customers_out` in `df` divided by `Customers` in DataFrame `mcc`, based on matching `fips_code` and `County_FIPS`.
    Any values in `percent_customers_out` greater than 100 are replaced by 100.
    
    Parameters:
    df (pd.DataFrame): The original DataFrame containing the column `customers_out` and `fips_code`.
    mcc (pd.DataFrame): The DataFrame containing the column `Customers` and `County_FIPS`.
    
    Returns:
    pd.DataFrame: The updated DataFrame `df` with the new column `percent_customers_out` and without unnecessary columns.
    """
    
    df['fips_code'] = df['fips_code'].astype(str)
    mcc['County_FIPS'] = mcc['County_FIPS'].astype(str)

    merged_df = df.merge(mcc[['County_FIPS', 'Customers']], left_on='fips_code', right_on='County_FIPS', how='left')
    merged_df['percent_customers_out'] = (merged_df['customers_out'] / merged_df['Customers']) * 100
    merged_df['percent_customers_out'] = merged_df['percent_customers_out'].clip(upper=100)
    merged_df = merged_df.drop(columns=['County_FIPS', 'Customers'])

    return merged_df




def estimate_customers_out(df, mcc):
    """
    Creates a new column `customers_out_estimate` in DataFrame `df`. If `customers_out` is zero,
    it estimates the value based on the mean `percent_customers_out` for the same `date`, `state`.
    The mean is multiplied by the number of `Customers` for the specific `fips_code` from the `mcc` DataFrame.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing `customers_out`, `percent_customers_out`, and other relevant columns.
    mcc (pd.DataFrame): The DataFrame containing the column `Customers` and `County_FIPS`.
    
    Returns:
    pd.DataFrame: The updated DataFrame `df` with the new column `customers_out_estimate`.
    """
    
    mean_percent = df[df['customers_out'] != 0].groupby(['date', 'state'])['percent_customers_out'].mean().reset_index()
    mean_percent = mean_percent.rename(columns={'percent_customers_out': 'mean_percent_customers_out'})
    
    df = df.merge(mean_percent, on=['date', 'state'], how='left')
    df = df.merge(mcc[['County_FIPS', 'Customers']], left_on='fips_code', right_on='County_FIPS', how='left')
    df['customers_out_estimate'] = df['customers_out']
    
    df.loc[df['customers_out'] == 0, 'customers_out_estimate'] = (
        df.loc[df['customers_out'] == 0, 'mean_percent_customers_out'] * df.loc[df['customers_out'] == 0, 'Customers'] / 100
    )

    df['customers_out_estimate'] = df['customers_out_estimate'].fillna(0)
    df = df.drop(columns=['mean_percent_customers_out', 'County_FIPS', 'Customers'])
    
    return df



def combine_dfs(dfs):
    """
    Combines a list of DataFrames into one large DataFrame by concatenating them vertically.
    
    Parameters:
    dfs (list of pd.DataFrame): A list containing DataFrames to be combined.
    
    Returns:
    pd.DataFrame: A single DataFrame containing all rows from the input DataFrames.
    """
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df