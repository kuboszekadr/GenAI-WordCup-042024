from typing import Type, Optional

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.pydantic_v1 import BaseModel
from langchain.agents.tools import BaseTool

from xgboost import XGBClassifier

import src.constants as consts
from src.tools.predictive_models.base import UserQuery
from src.tools.predictive_models.pipeline import DataPipeline


class ErrorPredictionModelFeatureImportance(BaseTool):
    name = "error_prediction_model_feature_importance"
    description = """
    Useful for when you have a prediction for a error on some machine by a error prediction model and want to know why did the model
    made that prediction. 
    The tool will return the feature importance of each feature according to the error prediction model
    """
    args_schema: Type[BaseModel] = UserQuery
    return_direct = True
    prediction_model = XGBClassifier()
    prediction_model.load_model(consts.MULTICLASS_XGB_MODEL)
    data_pipeline: DataPipeline = DataPipeline()


    def _run(self, date: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Use the tool."""
        raise NotImplementedError("Use Only Async Run")

    async def _arun(self, date: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        feature_names = await self.data_pipeline.get_feature_names(date_query=date)

        importances = self.prediction_model.feature_importances_

        feature_importance_dict = dict(zip(feature_names, importances))

        sorted_importance = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

        return str(sorted_importance)


if __name__ == '__main__':
    from asyncio import get_event_loop

    loop = get_event_loop()

    tool = ErrorPredictionModelFeatureImportance()
    date_query = "2016-01-01"

    res = loop.run_until_complete(tool.arun(dict(date=date_query)))
    print(res)