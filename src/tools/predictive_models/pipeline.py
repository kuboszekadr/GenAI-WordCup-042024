import pandas as pd

from src.tools.database.connectors import MockupConenctor
from src.constants import SENSORS_FIELDS

import pandas as pd
import numpy as np


def calculate_rolling_features(telemetry, fields, window_hours=3, rolling_window=24):
    # Calculate short-term (3-hour) rolling aggregates
    aggregates = {}
    for col in fields:
        resampled = telemetry.pivot_table(index='datetime', columns='machineID', values=col).resample(f'{window_hours}H', closed='left',
                                                                                                      label='right')
        aggregates[f'{col}mean_{window_hours}h'] = resampled.mean().unstack()
        aggregates[f'{col}sd_{window_hours}h'] = resampled.std().unstack()

        # Calculate long-term (24-hour) rolling aggregates based on the short-term ones
        rolling = resampled.mean().rolling(window=rolling_window // window_hours)
        aggregates[f'{col}mean_{rolling_window}h'] = rolling.mean().unstack()
        aggregates[f'{col}sd_{rolling_window}h'] = rolling.std().unstack()

    # Concatenate all temporary DataFrames into a comprehensive DataFrame
    feature_df = pd.concat(aggregates.values(), axis=1)
    feature_df.columns = aggregates.keys()
    feature_df.reset_index(inplace=True)
    return feature_df


def preprocess_errors(errors):
    # Create a column for each error type, and resample to count occurrences over time
    error_dummies = pd.get_dummies(errors.set_index('datetime')).reset_index()
    error_count = error_dummies.groupby(['machineID', 'datetime']).sum().reset_index()
    return error_count


def preprocess_maintenance(maintenance, telemetry):
    comp_rep = pd.get_dummies(maintenance.set_index('datetime')).reset_index()
    comp_rep = comp_rep.groupby(['machineID', 'datetime']).sum().reset_index()
    comp_rep = telemetry[['datetime', 'machineID']].merge(comp_rep, on=['datetime', 'machineID'], how='outer').fillna(0).sort_values(
        by=['machineID', 'datetime'])

    # Convert replacement indicators to dates and calculate days since last replacement
    for comp in ['comp1', 'comp2', 'comp3', 'comp4']:
        comp_rep.loc[comp_rep[comp] < 1, comp] = None
        comp_rep.loc[~comp_rep[comp].isnull(), comp] = comp_rep.loc[~comp_rep[comp].isnull(), 'datetime']
        comp_rep[comp] = comp_rep[comp].fillna(method='ffill')
        comp_rep[comp] = (comp_rep['datetime'] - comp_rep[comp]) / np.timedelta64(1, 'D')

    comp_rep = comp_rep.loc[comp_rep['datetime'] > pd.to_datetime('2015-01-01')]
    return comp_rep


def filter_data_for_date(dataframe, date, hours=24):
    """
    Filters the DataFrame for the 24-hour period ending on the specified date.

    Args:
    dataframe (pd.DataFrame): DataFrame containing a 'datetime' column.
    date (str): The end date in 'YYYY-MM-DD' format.
    hours (int): Number of hours before the end date to include in the filter.

    Returns:
    pd.DataFrame: Filtered DataFrame.
    """
    end_date = pd.to_datetime(date)
    start_date = end_date - pd.Timedelta(hours=hours)
    return dataframe[(dataframe['datetime'] >= start_date) & (dataframe['datetime'] <= end_date)]



def main_pipeline_with_date(telemetry, errors, maintenance, machines, target_date):
    """
    Processes all data related to a specific date to generate machine learning features.

    Args:
    telemetry, errors, maintenance, machines (pd.DataFrame): DataFrames containing the relevant data.
    target_date (str): The target date for which features are to be generated.

    Returns:
    pd.DataFrame: Final features DataFrame for the target date.
    """
    # Filter data for the 24 hours leading up to and including the target date
    telemetry_filtered = filter_data_for_date(telemetry, target_date, 24)
    errors_filtered = filter_data_for_date(errors, target_date, 24)
    maintenance_filtered = filter_data_for_date(maintenance, target_date, 24)

    # Generate features from the filtered data
    fields = SENSORS_FIELDS
    telemetry_features = calculate_rolling_features(telemetry_filtered, fields)

    # Process errors and maintenance records
    error_features = preprocess_errors(errors_filtered)
    maintenance_features = preprocess_maintenance(maintenance_filtered, telemetry_filtered)

    # Merge all features into a single DataFrame
    final_features = telemetry_features.merge(error_features, on=['datetime', 'machineID'], how='left')
    final_features = final_features.merge(maintenance_features, on=['datetime', 'machineID'], how='left')
    final_features = final_features.merge(machines, on='machineID', how='left').fillna(0)

    return final_features


class DataPipeline:

    def __init__(self):
        self.database = MockupConenctor()

    async def forward(self, date_query: str):

        telemetry = self.database.fetch("sensors")
        errors = self.database.fetch("errors")
        maintenance = self.database.fetch("maintenance")
        machines = self.database.fetch("machines")

        return main_pipeline_with_date(telemetry, errors, maintenance, machines, date_query)



