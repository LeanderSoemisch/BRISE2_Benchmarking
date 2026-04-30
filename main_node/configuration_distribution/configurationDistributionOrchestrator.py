from configuration_distribution.distribution_abs import AbstractDistribution
from tools.reflective_class_import import reflective_class_import
import logging


class ConfigurationDistributionOrchestrator:

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def get_distribution(self, distribution_description: dict) -> AbstractDistribution:
        """
        Returns an instance of a distribution algorithm based on the provided configuration.

        :param distribution_description: experiment config (dict) with one key, 
                                         whose value is a dict including a "Type" field.
        :return: an instantiated AbstractDistribution subclass.
        """

        try:
            distribution = distribution_description["DistributionMode"]
            dist_name = [key for key in distribution 
                        if isinstance(distribution[key], dict) 
                        and "Type" in distribution[key]][0]
                
        except:
            self.logger.info("No valid 'distributionMode' key found in Configuration description.")
            self.logger.info("Asynchronous Distribution is selected as default.")

            # fall back to Asynchronous Distribution
            distribution = {
                "AsynchronousDistribution": {
                    "Type": "AsynchronousDistribution"
                }
            }
            dist_name = "AsynchronousDistribution"

        distribution_type = distribution[dist_name]["Type"]

        distribution_class = reflective_class_import(
            class_name=distribution_type,
            folder_path="configuration_distribution"
        )
        
        return distribution_class(distribution)