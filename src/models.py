from src.config import config

from langchain_openai import AzureChatOpenAI

gpt_4 = AzureChatOpenAI(
            openai_api_base=config.openai.endpoint,
            openai_api_version=config.openai.version,
            deployment_name=config.openai.deployment_name,
            temperature=0
        )