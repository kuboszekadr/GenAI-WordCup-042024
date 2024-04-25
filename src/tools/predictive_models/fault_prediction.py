from typing import Type, Optional
import numpy as np
import pandas as pd
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.pydantic_v1 import BaseModel
from langchain.agents.tools import BaseTool

from xgboost import XGBClassifier

from src.tools.predictive_models.base import UserQuery
from src.tools.predictive_models.pipeline import DataPipeline
import src.constants as consts


class ErrorPredictionModel(BaseTool):
    name = "error_prediction_model"
    description = """
    Useful for when you want to predict if some machine will likely to fail in the next 24 hours due to a failure of a certain component.
    """
    args_schema: Type[BaseModel] = UserQuery
    return_direct = True
    data_pipeline: DataPipeline = DataPipeline()
    prediction_model = XGBClassifier()
    prediction_model.load_model(consts.MULTICLASS_XGB_MODEL)

    def _run(
            self,
            date: str,
            run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        raise NotImplementedError("Use Only Async Run")

    async def _arun(self, date: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        data = await self.data_pipeline.forward(date_query=date)

        machines = data['machineID'].values.tolist()
        features = pd.get_dummies(data.drop(['datetime', 'machineID'], axis=1))

        prediction = self.prediction_model.predict(features.values)

        dict_map = {0: 'no error',
                    1: 'error in comp1',
                    2: 'error in comp2',
                    3: 'error in comp3',
                    4: 'error in comp4'
                    }

        res = [{"error type:": dict_map[pred], "machine_to_fail": machines[index], "date": date_query}
               for index, pred in enumerate(prediction) if pred
               ]  # Only errors we care

        if res:
            return str(res)

        else:
            return "no errors"


if __name__ == '__main__':
    from asyncio import get_event_loop

    loop = get_event_loop()

    date_query = "2016-01-01"
    tool = ErrorPredictionModel()

    res = loop.run_until_complete(tool.arun(dict(date=date_query)))
    print(res)
