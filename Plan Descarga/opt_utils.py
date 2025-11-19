def get_env_file_to_load(argument):
    env_file = '.env'
    db_environment_to_use = argument
    if db_environment_to_use == 'prod':
        env_file = '.env.prod'
    elif db_environment_to_use == 'dev':
        env_file = '.env.dev'
    elif db_environment_to_use == 'dumped':
        env_file = '.env.dev'
    elif db_environment_to_use == 'saved-data-dev':
        env_file = '.env.saved-data-dev'
    return env_file