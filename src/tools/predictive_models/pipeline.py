
from src.tools.database.connectors import MockupConenctor
import pandas as pd
import numpy as np


def create_telemetry_features(telemetry):
    fields = ['volt', 'temperature', 'pressure', 'vibration']
    telemetry_features = []

    # Calculate 3-hour means and standard deviations
    for col in fields:
        resampled = pd.pivot_table(telemetry, index='datetime', columns='machineID', values=col).resample('3h', closed='left', label='right')
        mean_3h = resampled.mean().unstack()
        std_3h = resampled.std().unstack()

        telemetry_features.append(mean_3h)
        telemetry_features.append(std_3h)

        # Rolling 24-hour mean and standard deviation calculations
        rolling_mean_24h = resampled.mean().rolling(window=8).mean().unstack()  # 8 periods of 3 hours each
        rolling_std_24h = resampled.mean().rolling(window=8).std().unstack()

        telemetry_features.append(rolling_mean_24h)
        telemetry_features.append(rolling_std_24h)

    # Concatenate all feature series into a single DataFrame
    telemetry_df = pd.concat(telemetry_features, axis=1)
    cols = []
    for field in fields:
        cols.extend([
            f"{field}mean_3h", f"{field}sd_3h",
            f"{field}mean_24h", f"{field}sd_24h"
        ])
    telemetry_df.columns = cols
    telemetry_df.reset_index(inplace=True)

    # Drop rows that might have NaN values due to rolling window
    telemetry_df.dropna(inplace=True)

    return telemetry_df


def create_error_features(errors, telemetry):
    # Create dummy variables from the error ID
    error_dummies = pd.get_dummies(errors.set_index('datetime'), prefix='', prefix_sep='').reset_index()

    # Standardize column names if necessary (e.g., from 'errorID_error5' to 'error5')
    error_cols = {col: col.split('_')[-1] for col in error_dummies.columns if 'error' in col}
    error_dummies.rename(columns=error_cols, inplace=True)

    # Check for missing error columns and add them if missing
    expected_errors = ['error1', 'error2', 'error3', 'error4', 'error5']
    for error in expected_errors:
        if error not in error_dummies.columns:
            error_dummies[error] = False  # Add missing error column with all zeros

    # Ensure the order and naming of error columns
    error_dummies = error_dummies[['datetime', 'machineID'] + expected_errors]

    # Merge the error features with telemetry data
    error_features = telemetry[['datetime', 'machineID']].merge(error_dummies, on=['machineID', 'datetime'], how='left').fillna(0.0)

    temp = []
    for col in expected_errors:
        temp.append(pd.pivot_table(error_features,
                                   index='datetime',
                                   columns='machineID',
                                   values=col).resample('3h',
                                                        closed='left',
                                                        label='right',
                                                        ).first().unstack().rolling(window=24, center=False).sum())
    error_count = pd.concat(temp, axis=1)
    error_count.columns = [i + 'count' for i in expected_errors]
    error_count.reset_index(inplace=True)
    error_count = error_count.dropna()
    return error_count


def calculate_days_since_last_replacement(maint, telemetry):
    comp_rep = pd.get_dummies(maint.set_index('datetime')).reset_index()
    comp_rep.columns = ['datetime', 'machineID', 'comp1', 'comp2', 'comp3', 'comp4']
    comp_rep = telemetry[['datetime', 'machineID']].merge(comp_rep, on=['datetime', 'machineID'], how='outer').fillna(0).sort_values(
        by=['machineID', 'datetime'])
    for comp in ['comp1', 'comp2', 'comp3', 'comp4']:
        comp_rep.loc[comp_rep[comp] < 1, comp] = None
        comp_rep.loc[~comp_rep[comp].isnull(), comp] = comp_rep.loc[~comp_rep[comp].isnull(), 'datetime']
        comp_rep[comp] = comp_rep[comp].ffill()
        comp_rep[comp] = (comp_rep['datetime'] - comp_rep[comp]) / np.timedelta64(1, 'D')

    return comp_rep


def merge_features(telemetry_feat, error_count, comp_rep, machines):
    final_feat = telemetry_feat.merge(error_count, on=['datetime', 'machineID'], how='left')
    final_feat = final_feat.merge(comp_rep, on=['datetime', 'machineID'], how='left')
    final_feat = final_feat.merge(machines, on='machineID', how='left')
    return final_feat


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
    dataframe['datetime'] = pd.to_datetime(dataframe['datetime'])

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

    telemetry_feat = create_telemetry_features(telemetry_filtered)
    error_count = create_error_features(errors_filtered, telemetry_filtered)
    comp_rep = calculate_days_since_last_replacement(maintenance_filtered, telemetry_filtered)
    final_features = merge_features(telemetry_feat, error_count, comp_rep, machines)

    return final_features


class DataPipeline:

    def __init__(self):
        self.database = MockupConenctor()

    async def forward(self, date_query: str):

        telemetry = self.database.fetch("sensors")
        errors = self.database.fetch("errors")
        maintenance = self.database.fetch("maintenance")
        machines = self.database.fetch("machines")
        failures = self.database.fetch("failures")

        # Prepare telemetry data
        telemetry['temperature'] = telemetry['rotate']
        telemetry = telemetry.drop(columns='rotate')
        telemetry['datetime'] = pd.to_datetime(telemetry['datetime'], format="%Y-%m-%d %H:%M:%S")

        # Prepare error data
        errors['datetime'] = pd.to_datetime(errors['datetime'], format="%Y-%m-%d %H:%M:%S")
        errors['errorID'] = errors['errorID'].astype('object')

        # Prepare maintenance data
        maintenance['datetime'] = pd.to_datetime(maintenance['datetime'], format="%Y-%m-%d %H:%M:%S")
        maintenance['comp'] = maintenance['comp'].astype('object')

        # Prepare machine data
        machines['model'] = machines['model'].astype('object')

        # Prepare failures data
        failures['datetime'] = pd.to_datetime(failures['datetime'], format="%Y-%m-%d %H:%M:%S")
        failures['failure'] = failures['failure'].astype('object')

        final_features = main_pipeline_with_date(telemetry, errors, maintenance, machines, date_query)
        return final_features

    async def get_feature_names(self, date_query):
        telemetry = self.database.fetch("sensors")
        errors = self.database.fetch("errors")
        maintenance = self.database.fetch("maintenance")
        machines = self.database.fetch("machines")
        failures = self.database.fetch("failures")

        # Prepare telemetry data
        telemetry['temperature'] = telemetry['rotate']
        telemetry = telemetry.drop(columns='rotate')
        telemetry['datetime'] = pd.to_datetime(telemetry['datetime'], format="%Y-%m-%d %H:%M:%S")

        # Prepare error data
        errors['datetime'] = pd.to_datetime(errors['datetime'], format="%Y-%m-%d %H:%M:%S")
        errors['errorID'] = errors['errorID'].astype('object')

        # Prepare maintenance data
        maintenance['datetime'] = pd.to_datetime(maintenance['datetime'], format="%Y-%m-%d %H:%M:%S")
        maintenance['comp'] = maintenance['comp'].astype('object')

        # Prepare machine data
        machines['model'] = machines['model'].astype('object')

        # Prepare failures data
        failures['datetime'] = pd.to_datetime(failures['datetime'], format="%Y-%m-%d %H:%M:%S")
        failures['failure'] = failures['failure'].astype('object')

        final_features = main_pipeline_with_date(telemetry, errors, maintenance, machines, date_query)
        return final_features.columns.tolist()

