import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class ConfigDetector:
    """Detects configuration type and available configuration files"""

    DEFAULT_CONFIG_PATHS = [
        "./configuration.json",
        "./configs/benchmark_templates/comparative_benchmark_template.json",
        "./configs/benchmark_templates/benchmark_template_with_grouped_testcases.json"
    ]

    @staticmethod
    def detect_config_file() -> Optional[str]:
        """
        Detect which configuration file to use.

        Returns:
            Path to the configuration file to use, or None if none found
        """
        for config_path in ConfigDetector.DEFAULT_CONFIG_PATHS:
            if Path(config_path).exists():
                logger.info(f"Found configuration file: {config_path}")
                return config_path

        logger.warning("No configuration file found in default locations")
        return None

    @staticmethod
    def load_config(config_path: str) -> Optional[Dict[str, Any]]:
        """
        Load configuration from JSON file.

        Args:
            config_path: Path to configuration file

        Returns:
            Dictionary containing configuration, or None if load failed
        """
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {config_path}: {e}")
        except Exception as e:
            logger.error(f"Error loading {config_path}: {e}")
        return None

    @staticmethod
    def has_comparative_metrics(config_dict: Dict[str, Any]) -> bool:
        """
        Check if configuration includes comparative metrics.

        Args:
            config_dict: Configuration dictionary

        Returns:
            True if comparative metrics are configured
        """
        comp_metrics = config_dict.get("Benchmark", {}).get("ComparativeMetrics", {})
        if not comp_metrics:
            return False
        enabled = [
            m for m in ["RegretAnalysis", "NormalizedImprovement", "PerformanceProfile", "ComparativeTable"]
            if comp_metrics.get(m) and isinstance(comp_metrics[m], dict)
        ]
        if enabled:
            logger.info(f"Comparative metrics detected: {', '.join(enabled)}")
        return bool(enabled)
