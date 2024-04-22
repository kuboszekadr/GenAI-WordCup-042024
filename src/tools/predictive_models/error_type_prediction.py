from typing import Type, Optional

import pandas as pd
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.pydantic_v1 import BaseModel
from langchain.agents.tools import BaseTool

from src.tools.predictive_models.base import SensorsData




class XGBoostErrorPredictionModel(BaseTool):

    name = "error_prediction_model"
    description = """
    Useful for when you have the sensors data: volt, rotate,pressure, vibration for some machineID and want to predict
    the possible error type or fault type for that machine using the Error Prediction Model. 
    """
    args_schema: Type[BaseModel] = SensorsData
    return_direct = True

    def _run(
            self,
            # TODO: what are the input should be?
            run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        raise NotImplementedError("Use Only Async Run")

    async def _arun(self,
                    # TODO: what are the input should be?
                    run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        pass

    async def _feature_extraction(self, voltage, rotate,pressure,vibration) -> pd.DataFrame:
        pass