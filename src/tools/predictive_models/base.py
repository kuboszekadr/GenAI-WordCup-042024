from langchain.pydantic_v1 import BaseModel, Field

class MachineID(BaseModel):
    machine_id: int = Field("The machine ID, for example: 33")
class SensorsData(BaseModel):
    voltage: float = Field("The voltage data of the machine")
    rotate: float = Field("The rotate data of the machine")
    pressure: float = Field("The pressure data of the machine")
    vibration: float = Field("The vibration data of the machine")
class XGBoostFeautures(BaseModel):
    features: str = Field(description="The features to give as input to the XGBoostErrorPredictionModel")
