from langchain.pydantic_v1 import BaseModel, Field


class SQLQuery(BaseModel):
    sql_query: str = Field(description="SQL query")