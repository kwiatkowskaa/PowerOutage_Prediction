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

# Palette
custom_palette = ["#8ecae6", "#219ebc", "#023047", "#ffb703", "#fb8500"]
custom_cmap = ListedColormap(custom_palette)

def plot_event_counts(df):
    """
    Plots the most frequently reported weather disasters in a given year.

    Parameters:
    - df (DataFrame): A Pandas DataFrame containing storm event data for a single year.

    Output:
    - A bar chart displaying the number of reported occurrences for each disaster type.
    """
    
    # Count the occurrences of each event type
    event_counts = df['EVENT_TYPE'].value_counts()

    # Extract the year from the DataFrame
    year = int(df[['YEAR']].iloc[0, 0])

    # Plot the event counts
    plt.figure(figsize=(12, 6))
    sns.barplot(x=event_counts.index, y=event_counts.values, palette='viridis')

    # Formatting the plot
    plt.title(f'Most Frequently Reported Weather Disasters in {year}', fontsize=14)
    plt.xlabel('Event Type', fontsize=12)
    plt.ylabel('Number of Reports', fontsize=12)
    plt.xticks(rotation=90)

    # Display the plot
    plt.show()





def plot_event_trends(dfs):
    """
    Plots the trend of disaster event occurrences over multiple years.
    
    Parameters:
    - dfs (list of DataFrames): A list of Pandas DataFrames, each containing storm event data for a single year.
    
    Output:
    - A line plot showing the trend of the most common disaster types over multiple years.
    """
    
    # Combine all yearly DataFrames into one
    df_combined = pd.concat(dfs, ignore_index=True)
    
    # Count occurrences of each event type per year
    event_trends = df_combined.groupby(['YEAR', 'EVENT_TYPE']).size().reset_index(name='COUNT')

    # Select the top N most frequent disaster types
    top_events = event_trends.groupby("EVENT_TYPE")['COUNT'].sum().nlargest(5).index
    event_trends_filtered = event_trends[event_trends['EVENT_TYPE'].isin(top_events)]

    # Plot the trends
    plt.figure(figsize=(14, 7))
    sns.lineplot(data=event_trends_filtered, x='YEAR', y='COUNT', hue='EVENT_TYPE', marker='o', palette=custom_palette)

    plt.title('Trend of Most Frequent Weather Disasters Over the Years', fontsize=14)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Number of Reported Events', fontsize=12)
    plt.legend(title='Disaster Type')
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.show()




def plot_state_event_counts(df):
    """
    This function visualizes the number of reports for specific weather event types (Thunderstorm Wind, Hail, 
    Flash Flood, High Wind and Winter Weather) across different states. It filters the dataset for these events, groups it by state and event type,
    and creates a stacked bar plot to show the distribution of event reports per state.

    Parameters:
    df (pd.DataFrame): A pandas DataFrame containing the storm events data, including 'STATE' and 'EVENT_TYPE' columns.

    Returns:
    None: Displays a stacked bar plot showing the count of each event type per state.
    """
    # Extract the year from the DataFrame
    year = int(df[['YEAR']].iloc[0, 0])
    
    # Define the event types to analyze
    event_types = ['Thunderstorm Wind', 'Hail', 'Flash Flood', 'High Wind', 'Winter Weather']
    
    # Filter the DataFrame to include only the selected event types
    df_filtered = df[df['EVENT_TYPE'].isin(event_types)]

    state_counts = df_filtered.groupby(['STATE', 'EVENT_TYPE']).size().unstack().fillna(0)

    # Sum the counts of all event types for each state and sort states from highest to lowest
    state_counts['total'] = state_counts.sum(axis=1)
    state_counts_sorted = state_counts.sort_values(by='total', ascending=False)

    # Plotting the data
    plt.figure(figsize=(12, 6))
    
    # Create a stacked bar plot of sorted data
    state_counts_sorted.drop('total', axis=1).plot(kind='bar', stacked=True, figsize=(12, 6), colormap=custom_cmap)
    
    # Adding titles and labels
    plt.title(f'Number of Most Frequent Weather Disasters Reports by State in {year}', fontsize=14)
    plt.xlabel('State', fontsize=12)
    plt.ylabel('Number of Reports', fontsize=12)
    
    # Rotate x-axis labels for better visibility
    plt.xticks(rotation=90)
    
    # Add a legend to explain the event types
    plt.legend(title="Event Type")
    
    # Add gridlines for better readability
    plt.grid(axis='y')
    
    # Show the plot
    plt.show()




def plot_county_event_counts(df):
    """
    This function visualizes the number of reports for specific weather event types (Thunderstorm Wind, Hail, 
    Flash Flood, High Wind and Winter Weather) at the county level. It filters the dataset for these events, groups it by county and event type, 
    and creates a stacked bar plot to show the distribution of event reports for the top 20 counties.

    Parameters:
    df (pd.DataFrame): A pandas DataFrame containing the storm events data, including 'CZ_NAME' (county), 
                        'STATE', and 'EVENT_TYPE' columns.

    Returns:
    None: Displays a stacked bar plot showing the count of each event type for the top 20 counties.
    """

    # Extract the year from the DataFrame
    year = int(df[['YEAR']].iloc[0, 0])

    # Define the event types to analyze
    event_types = ['Thunderstorm Wind', 'Hail', 'Flash Flood', 'High Wind', 'Winter Weather']
    
    # Filter the DataFrame to include only the selected event types
    df_filtered = df[df['EVENT_TYPE'].isin(event_types)]

    # Create a new column that combines the county name and state
    df_filtered['COUNTY_STATE'] = df_filtered['CZ_NAME'] + " (" + df_filtered['STATE'] + ")"

    # Group by county-state and event type, then count the occurrences of each combination
    county_counts = df_filtered.groupby(['COUNTY_STATE', 'EVENT_TYPE']).size().unstack().fillna(0)

    # Sum the counts of all event types for each county-state and sort by total counts
    county_counts['total'] = county_counts.sum(axis=1)
    county_counts_sorted = county_counts.sort_values(by='total', ascending=False)

    # Plotting the data for the top 20 counties
    plt.figure(figsize=(12, 6))
    
    # Create a stacked bar plot for the top 20 counties sorted by total event counts
    county_counts_sorted.head(20).drop('total', axis=1).plot(kind='bar', stacked=True, figsize=(12, 6), colormap=custom_cmap)
    
    # Adding titles and labels
    plt.title(f'Number of Most Frequent Weather Disasters by County in {year} (Top 20)', fontsize=14)
    plt.xlabel('County (State)', fontsize=12)
    plt.ylabel('Number of Reports', fontsize=12)
    
    # Rotate x-axis labels for better visibility
    plt.xticks(rotation=90)
    
    # Add a legend to explain the event types
    plt.legend(title="Event Type")
    
    # Add gridlines for better readability
    plt.grid(axis='y')
    
    # Show the plot
    plt.show()




def plot_monthly_event_trends(dfs):
    """
    Plots the trend of disaster event occurrences by month for multiple years.
    
    Parameters:
    - dfs (list of DataFrames): A list of Pandas DataFrames, each containing storm event data for a single year.
    
    Output:
    - A line plot showing the trend of the number of disaster events reported for each month across multiple years.
    """

    # Combine all yearly DataFrames into one
    df_combined = pd.concat(dfs, ignore_index=True)
    
    # Extract month and year from the event data
    df_combined['MONTH'] = pd.to_datetime(df_combined['BEGIN_DATE_TIME']).dt.month
    df_combined['YEAR'] = pd.to_datetime(df_combined['BEGIN_DATE_TIME']).dt.year

    # Count occurrences of events for each year and month
    event_trends_monthly = df_combined.groupby(['YEAR', 'MONTH']).size().reset_index(name='COUNT')

    # Plot the trends
    plt.figure(figsize=(14, 7))
    sns.lineplot(data=event_trends_monthly, x='MONTH', y='COUNT', hue='YEAR', marker='o', palette='tab10', linewidth=2)

    plt.title('Monthly Trend of Storms Events Over the Years', fontsize=14)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Number of Reported Events', fontsize=12)
    plt.xticks(ticks=range(1, 13), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.legend(title='Year')
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.show()




def plot_state_event_seasonality(df):
    """
    Plots the seasonality of specific disaster events based on the month.
    
    Parameters:
    - df (DataFrame): A DataFrame containing storm event data, including the event type and start date.
    
    Output:
    - A box plot showing the distribution of disaster events by month.
    """
    # Extract the year from the DataFrame
    year = int(df[['YEAR']].iloc[0, 0])
    
    # Convert 'BEGIN_DATE_TIME' column to datetime
    df['BEGIN_DATE_TIME'] = pd.to_datetime(df['BEGIN_DATE_TIME'])
    
    # Extract month from the 'BEGIN_DATE_TIME' column
    df['MONTH'] = df['BEGIN_DATE_TIME'].dt.month
    
    # List of event types to analyze
    event_types = ['Thunderstorm Wind', 'Hail', 'Flash Flood', 'High Wind', 'Winter Weather']
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Filter data for the selected event types and create a boxplot
    sns.boxplot(x='MONTH', y='EVENT_TYPE', data=df[df['EVENT_TYPE'].isin(event_types)], palette=custom_palette)
    
    # Set the title and labels
    plt.title(f'Seasonality of Most Frequent Weather Disasters in {year}', fontsize=14)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Event Type', fontsize=12)
    
    # Set x-axis labels for months
    plt.xticks(range(12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    
    # Enable grid for better readability
    plt.grid(True)
    
    # Show the plot
    plt.show()



def plot_event_heatmap(df):
    """
    Plots a heatmap showing the monthly distribution of disaster events for the top 20 most affected counties.

    Parameters:
    - df (DataFrame): A Pandas DataFrame containing storm event data with 'STATE', 'BEGIN_DATE_TIME', and 'YEAR'.

    Output:
    - A heatmap visualizing the frequency of disaster events by month and county.
    """

    # Konwersja daty i ekstrakcja miesiąca
    df['BEGIN_DATE_TIME'] = pd.to_datetime(df['BEGIN_DATE_TIME'])
    df['MONTH'] = df['BEGIN_DATE_TIME'].dt.month

    # Zliczanie liczby zdarzeń dla każdego hrabstwa
    county_counts = df.groupby("STATE").size().nlargest(20).index
    df_filtered = df[df["STATE"].isin(county_counts)]

    # Grupowanie liczby zgłoszeń na miesiąc dla każdego hrabstwa
    heatmap_data = df_filtered.groupby(['STATE', 'MONTH']).size().unstack(fill_value=0)

    # Tworzenie wykresu heatmapy
    plt.figure(figsize=(14, 8))
    sns.heatmap(heatmap_data, cmap="rocket_r", linewidths=0.5, annot=True, fmt="d")

    # Ustawienie etykiet
    plt.title('Monthly Distribution of Storm Events in Top 20 States', fontsize=14)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('County (State)', fontsize=12)
    plt.xticks(ticks=range(12), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=45)
    plt.yticks(rotation=0)

    # Wyświetlenie wykresu
    plt.show()



def plot_top_damage_events(df, damage_type):
    """
    Plots the top 10 disaster event types that caused the highest property damage.

    Output:
    - A horizontal bar chart displaying the top 10 disaster categories with the highest total property damage.
    """
        
    # Extract the year from the DataFrame
    year = int(df[['YEAR']].iloc[0, 0])

    # Group by event type and sum the total property damage, then select the top 10
    top_damage = df.groupby('EVENT_TYPE')['DAMAGE_PROPERTY'].sum().sort_values(ascending=False).head(10)
    
    # Create the bar chart
    plt.figure(figsize=(12, 6))
    plt.barh(top_damage.index, top_damage.values, color='219ebc')
    
    # Set labels and title
    plt.xlabel('Property Damage ($)', fontsize=12)
    plt.ylabel('Event Type', fontsize=12)
    plt.title(f'Top 10 Disaster Categories with the Highest Property Damage in {year}', fontsize=14)
    
    # Invert the y-axis for better visualization (highest damage at the top)
    plt.gca().invert_yaxis()
    
    # Show the plot
    plt.show()



def plot_top_damage_events(df, damage_type='DAMAGE_PROPERTY'):
    """
    Plots the top 10 disaster event types that caused the highest financial losses.

    Parameters:
    - df (DataFrame): A Pandas DataFrame containing storm event data.
    - damage_type (str): Column name specifying which type of damage to analyze ('DAMAGE_PROPERTY' or 'DAMAGE_CROPS').

    Output:
    - A horizontal bar chart displaying the top 10 disaster categories with the highest total damage.
    """
    # Extract the year from the DataFrame
    year = int(df[['YEAR']].iloc[0, 0])
    
    # Validate damage_type input
    if damage_type not in df.columns:
        raise ValueError(f"Column '{damage_type}' not found in the DataFrame. Choose 'DAMAGE_PROPERTY' or 'DAMAGE_CROPS'.")

    # Group by event type and sum the total damage, then select the top 10
    top_damage = df.groupby('EVENT_TYPE')[damage_type].sum().sort_values(ascending=False).head(10)
    
    # Define plot title dynamically based on selected damage type
    damage_label = "Property Damage" if damage_type == 'DAMAGE_PROPERTY' else "Crop Damage"
    
    # Create the bar chart
    plt.figure(figsize=(12, 6))
    plt.barh(top_damage.index, top_damage.values, color='#219ebc')
    
    # Set labels and title
    plt.xlabel(f'{damage_label} ($)', fontsize=12)
    plt.ylabel('Event Type', fontsize=12)
    plt.title(f'Top 10 Disaster Categories with the Highest {damage_label} in {year}', fontsize=14)
    
    # Invert the y-axis for better visualization (highest damage at the top)
    plt.gca().invert_yaxis()
    
    # Show the plot
    plt.show()




def convert_damage(value):
    """
    Converts damage values from string format with suffixes ('K', 'M', 'B') to numerical format.
    
    Parameters:
    - value (str, float, or NaN): The damage value that may contain suffixes:
        - 'K' (thousands) → Multiplies by 1,000
        - 'M' (millions) → Multiplies by 1,000,000
        - 'B' (billions) → Multiplies by 1,000,000,000
        - If the value is already a number or does not contain a suffix, it is returned as a float.
    
    Returns:
    - float: The numerical equivalent of the damage value.
    """
    
    if pd.isna(value) or value == '':  # Handle missing or empty values
        return 0.0
    
    value = str(value).upper().strip()  # Convert to uppercase and remove spaces
    
    if value.endswith('K'):  # Thousands
        return float(value.replace('K', '')) * 1_000
    elif value.endswith('M'):  # Millions
        return float(value.replace('M', '')) * 1_000_000
    elif value.endswith('B'):  # Billions
        return float(value.replace('B', '')) * 1_000_000_000
    else:  # Assume it's already a numeric value
        return float(value)
    




def plot_monthly_damage(dfs, damage_type='DAMAGE_PROPERTY'):
    """
    Plots the monthly trend of financial damage caused by weather events over multiple years.
    
    Parameters:
    - dfs (list of DataFrames): A list of Pandas DataFrames, each containing storm event data for a single year.
    - damage_type (str): Specifies the type of damage to analyze ('DAMAGE_PROPERTY' or 'DAMAGE_CROPS').

    Output:
    - A line plot displaying monthly financial damage trends for each year.
    """
        # Combine all yearly DataFrames into one
    df_combined = pd.concat(dfs, ignore_index=True)

    # Convert date column to datetime format
    df_combined['BEGIN_DATE_TIME'] = pd.to_datetime(df_combined['BEGIN_DATE_TIME'])

    # Extract year and month
    df_combined['YEAR'] = df_combined['BEGIN_DATE_TIME'].dt.year
    df_combined['MONTH'] = df_combined['BEGIN_DATE_TIME'].dt.month

    # Ensure is numeric and handle missing values
    df_combined['damage_type'] = pd.to_numeric(df_combined[damage_type], errors='coerce').fillna(0)

    # Sum total damage for each month and year
    monthly_damage = df_combined.groupby(['YEAR', 'MONTH'])['damage_type'].sum().reset_index()

    # Define plot title dynamically based on selected damage type
    damage_label = "Property Damage" if damage_type == 'DAMAGE_PROPERTY' else "Crop Damage"

    # Plot the trends
    plt.figure(figsize=(14, 7))
    sns.lineplot(data=monthly_damage, x='MONTH', y='damage_type', hue='YEAR', marker='o', palette='tab10', linewidth=2)

    plt.title(f'Monthly {damage_label} Trends Over the Years', fontsize=14)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Total Property Damage ($)', fontsize=12)
    plt.xticks(ticks=range(1, 13), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.legend(title='Year')
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.show()






    # # Validate damage_type input
    # if damage_type not in dfs[0].columns:
    #     raise ValueError(f"Column '{damage_type}' not found in the DataFrame. Choose 'DAMAGE_PROPERTY' or 'DAMAGE_CROPS'.")
    
    # plt.figure(figsize=(12, 6))

    # # Iterate through each year's DataFrame
    # for df in dfs:
    #     df['MONTH'] = pd.to_datetime(df['BEGIN_DATE_TIME']).dt.month
    #     year = int(df['YEAR'].iloc[0])  # Extract the year from the DataFrame
        
    #     # Group by month and sum the damage
    #     monthly_damage = df.groupby('MONTH')[damage_type].sum()
        
    #     # Plot the data for each year
    #     plt.plot(monthly_damage.index, monthly_damage.values, label=f'{year}', marker='o')

    # # Define plot title dynamically based on selected damage type
    # damage_label = "Property Damage" if damage_type == 'DAMAGE_PROPERTY' else "Crop Damage"

    # # Set labels and title
    # plt.xlabel('Month', fontsize=12)
    # plt.ylabel(f'{damage_label} ($)', fontsize=12)
    # plt.title(f'Monthly {damage_label} Trends Over the Years', fontsize=14)

    # # Format x-axis with month labels
    # plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

    # # Add grid and legend
    # plt.grid(True, linestyle='--', alpha=0.7)
    # plt.legend(title='Year')

    # # Show the plot
    # plt.show()
