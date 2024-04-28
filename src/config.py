import environ


@environ.config(prefix="APP")
class AppConfig:
    @environ.config(prefix='OPENAI')
    class OpenAI:
        endpoint = environ.var()
        key = environ.var()
        deployment_name = environ.var()

    @environ.config(prefix='SQLSERVER')
    class SQLServer:
        server_hostname = environ.var()
        http_path = environ.var()
        access_token = environ.var()

    sql_server = environ.group(SQLServer)


config = AppConfig.from_environ()
