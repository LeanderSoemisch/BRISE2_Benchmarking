import re
from typing import Any, Dict


class ExperimentParser:
    """Parses experiment naming and extracts features"""

    PATTERN_WITH_TEST_CASE = re.compile(
        r'^exp_([^_]+)_([^_]+)_([^_]+)_([^_]+)_([^_]+)_test_case_(\d+)(?:_(.+?))?(?:_(\d+))?$')
    PATTERN_WITHOUT_TEST_CASE = re.compile(r'^exp_([^_]+)_([^_]+)_([^_]+)_([^_]+)_([^_]+?)(?:_(\d+))?$')
    TEST_CASE_PATTERN = re.compile(r'(test_case_\d+(?:_[a-zA-Z_]+)?)')

    @staticmethod
    def get_name(exp: Any) -> str:
        """Get experiment name or ID"""
        return getattr(exp, 'name', None) or getattr(exp, 'ed_id', None) or 'experiment'

    def parse_features(self, raw_name: str) -> Dict[str, Any]:
        """Parse features from experiment naming pattern

        Expected patterns:
        - exp_<task>_<model>_<sampler>_<configStrategy>_<stopCondition>_test_case_<N>[_<descriptor>][_<idx>]
        - exp_<task>_<model>_<sampler>_<configStrategy>_<stopCondition>[_<idx>]
        """
        features = {'Task': None, 'Model': None, 'Sampler': None, 'ConfigurationStrategy': None, 'StopCondition': None,
            'TestCase': None, 'Index': None, 'ExperimentName': None}

        match = self.PATTERN_WITH_TEST_CASE.match(raw_name)
        if match:
            features['Task'] = match.group(1)
            features['Model'] = match.group(2)
            features['Sampler'] = match.group(3)
            features['ConfigurationStrategy'] = match.group(4)
            features['StopCondition'] = match.group(5)
            features['TestCase'] = int(match.group(6))

            middle_part = match.group(7)
            last_part = match.group(8)

            if middle_part and last_part:
                features['Index'] = int(last_part)
            elif last_part and not middle_part:
                features['Index'] = int(last_part)
            elif middle_part and not last_part and middle_part.isdigit():
                features['Index'] = int(middle_part)

            return features

        match = self.PATTERN_WITHOUT_TEST_CASE.match(raw_name)
        if match:
            features['Task'] = match.group(1)
            features['Model'] = match.group(2)
            features['Sampler'] = match.group(3)
            features['ConfigurationStrategy'] = match.group(4)
            features['StopCondition'] = match.group(5)
            if match.group(6):
                features['Index'] = int(match.group(6))

        return features

    def build_display_name(self, raw_name: str) -> str:
        """Build display name from raw experiment name

        Examples:
        - exp_test_gpr_sobol_quantitybased_timebased_test_case_0 -> test_case_0
        - exp_test_gpr_sobol_quantitybased_timebased_test_case_2_wo_dch_2 -> test_case_2_wo_dch
        """
        match = self.TEST_CASE_PATTERN.search(raw_name)
        return match.group(1) if match else raw_name
