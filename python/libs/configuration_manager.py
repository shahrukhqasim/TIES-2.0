import configparser as cp

config_manager_instance = "none"


class ConfigurationManager:
    def __init__(self, config_file_path, config_name):
        config_file = cp.ConfigParser()
        config_file.read(config_file_path)
        self.config = config_file[config_name]

    @staticmethod
    def init(config_file_path, config_name):
        global config_manager_instance
        if config_manager_instance!="none":
            raise Exception("Config manager already initialized")
        config_manager_instance = ConfigurationManager(config_file_path, config_name)

    @staticmethod
    def get_config_param(key, type="str"):
        global config_manager_instance
        if config_manager_instance=="none":
            raise Exception("Config manager not initialized")

        return_value = None
        if type=="str":
            return_value = str(config_manager_instance.config[key])
        elif type=="float":
            return_value = float(config_manager_instance.config[key])
        elif type=="int":
            return_value = int(config_manager_instance.config[key])
        else:
            raise Exception("Unknown config type")

        return return_value

    @staticmethod
    def set_config_param(key, value):
        global config_manager_instance
        if config_manager_instance=="none":
            raise Exception("Config manager not initialized")

        config_manager_instance.config[key] = value
