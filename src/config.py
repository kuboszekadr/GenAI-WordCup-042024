import environ

class AppConfig:

    @environ.config(prefix='OPENAI')
    class OpenAI:
        endpoint = environ.var()
        key = environ.var()
        deployment_name = environ.var()

config = AppConfig.from_environ()
