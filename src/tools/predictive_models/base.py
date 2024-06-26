from langchain.pydantic_v1 import BaseModel, Field

class UserQuery(BaseModel):
    date: str = Field(description="datetime of only day, month and year in the format YYYY-MM-DD")

