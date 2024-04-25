from typing import Type, Optional

from langchain.agents.tools import BaseTool
from langchain.pydantic_v1 import BaseModel
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)


from src.tools.database.base import SQLQuery
from src.tools.database.connectors import MockConnector


class DatabaseRetriever(BaseTool):
    name = "database_retriever"
    description = """
    Useful for when you want to send a SQL query to the database and get results for analysis. The results will be returned as a dataframe. 
    Current SQL Tables you can query:
    1. errors - These are errors encountered by the machines while in operating condition. Since, these errors don't shut down the machines, these are not considered as failures. The error date and times are rounded to the closest hour since the telemetry data is collected at an hourly rate. Columns: "datetime","machineID","errorID". Example record: 2015-01-03 07:00:00,1,"error1"
    2. maintenance - Machine component replacements are recorded as either proactive maintenance during scheduled visits or reactive maintenance following a breakdown, with all data from 2014 and 2015 rounded to the nearest hour to match telemetry collection. Columns: "datetime","machineID","comp". Example record: 2014-06-01 06:00:00,1,"comp2" 
    3. sensors - It consists of hourly average of voltage, rotation, pressure, vibration collected from 100 machines for the year 2015. Columns: "datetime","machineID","volt","rotate","pressure","vibration". Example record: 2015-01-01 06:00:00,1,176.217853015625,418.504078221616,113.077935462083,45.0876857639276
    4. failures - Each record represents replacement of a component due to failure. This data is a subset of Maintenance data. This data is rounded to the closest hour since the sensors data is collected at an hourly rate. Columns: "datetime", "machineID", "failure". Example record: 2015-01-05 06:00:00,1,"comp4"
    5. machines - Model type & age of the Machines. Columns: "machineID","model","age". Example record: 1,"model3",18
 
    """
    args_schema: Type[BaseModel] = SQLQuery
    return_direct = True

    def _run(
            self,
            sql_query: str,
            run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        raise NotImplementedError("Use Only Async Run")

    async def _arun(self,
                    sql_query: str,
                    run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        return str(MockConnector().fetch(query=sql_query))