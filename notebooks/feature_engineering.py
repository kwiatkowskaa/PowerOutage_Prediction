import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

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


