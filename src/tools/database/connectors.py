import pandas as pd
from src.constants import TABLES

class SQLDatabaseConnector:
    """
    Connector to the DBR sql database
    """
    def fetch(self, query):
        pass

class MockupConenctor:
    """
    Contain a mockup of historical data that can be processed and give to the model
    """

    def fetch(self, query) -> pd.DataFrame:
        for table in TABLES:
            table_name = table.name.split("/")[-1]
            if query in table_name:
                return pd.read_csv(table)

        raise FileNotFoundError(f"Did not find the correct Table name for query: {query}")

