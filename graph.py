from langgraph import StateGraph

from typing import TypedDict, List, Annotated
from operator import add

class AgentState(TypedDict):
    state: str
    messages: Annotated[List[str], add]


def response_generator_tool(state):
    pass


def context_retriever_tool(state):
    pass


if __name__ == '__main__':
    pass