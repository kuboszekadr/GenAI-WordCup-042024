import pandas as pd
from src.constants import TABLES
from src.config import config
from databricks import sql


class SQLDatabaseConnector:
    """
    Connector to the DBR sql database
    """

    # def __init__(self):
    #     self.connection = sql.connect(
    #         server_hostname=config.sql_server.server_hostname,
    #         http_path=config.sql_server.http_path,
    #         access_token=config.sql_server.access_token)

    def __del__(self):
        self.connection.close()

    def fetch(self, query):
        return pd.read_sql(query, self.connection)


class MockConnector:
    """
    Contain a mockup of historical data that can be processed and give to the model
    """

    def fetch(self, query) -> pd.DataFrame:
        for table in TABLES:
            table_name = table.name.split("/")[-1]
            if query in table_name:
                return pd.read_csv(table)

        raise FileNotFoundError(f"Did not find the correct Table name for query: {query}")
