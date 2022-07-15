from resources.crosswalk_config import crosswalk_config


class Config:
    def __init__(self, environment):
        """
        Parameters
        ----------
        environment : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.env = environment

    def get_config(self):
        return crosswalk_config[env]
