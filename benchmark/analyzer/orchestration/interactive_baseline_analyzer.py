import logging
import time
import webbrowser
from pathlib import Path
from typing import List, Any, Optional

from analyzer.comparison import BaselineSelector, SelectableExperiment
from analyzer.config import BenchmarkConfig
from analyzer.orchestration import BenchmarkAnalyzer
from analyzer.util.baseline_selection_server import BaselineSelectionServer
from template.template_loader import TemplateLoader

logger = logging.getLogger(__name__)


class InteractiveBaselineAnalyzer:
    """Manages interactive baseline selection workflow"""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.baseline_selector = BaselineSelector(Path(config.results_folder))
        self.template_loader = TemplateLoader()
        self.server = BaselineSelectionServer(port=8765)
        self.selected_baselines: Optional[List[str]] = None
        self.analyzer = BenchmarkAnalyzer(config)

    def analyze_with_baseline_selection(self, output_html: str, output_csv: str):
        if self.baseline_selector.has_selection():
            selected = self.baseline_selector.load_selection()
            if selected:
                logger.info(f"Using previously selected baselines: {selected}")
                self._analyze_with_baselines(selected, output_html, output_csv)
                return

        experiments = self._load_all_experiments()
        selectable = self.baseline_selector.get_selectable_experiments(experiments)

        if not selectable:
            logger.error("No experiments available for baseline selection")
            logger.info("Falling back to standard analysis without baselines")
            self.analyzer.analyze(output_html, output_csv)
            return

        self._serve_baseline_selection_ui(selectable, output_html, output_csv)

    def _load_all_experiments(self) -> List[Any]:
        from analyzer.data_pipeline import ExperimentLoader
        return ExperimentLoader(self.config.results_folder).load_all_experiments()

    def _serve_baseline_selection_ui(self, experiments: List[SelectableExperiment], output_html: str, output_csv: str):
        logger.info(f"Generating baseline selection UI for {len(experiments)} experiments")

        html_content = self.template_loader.render_jinja_template(
            'baseline_selection.html',
            {'experiment_name': self.config.experiment_name, 'experiments': experiments}
        )

        analyzer = self

        def on_selection_complete(selected):
            logger.info(f"User selected {len(selected)} baseline(s): {selected}")
            analyzer.baseline_selector.save_selection(selected)
            analyzer.selected_baselines = selected
            analyzer._analyze_with_baselines(selected, output_html, output_csv)

        url = self.server.start(html_content, output_html, on_selection_complete)
        logger.info(f"Opening baseline selection interface at {url}")
        webbrowser.open(url)

        logger.info("Waiting for baseline selection... (Press Ctrl+C to cancel)")
        try:
            while self.selected_baselines is None:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Baseline selection cancelled")
            self.server.stop()
            raise

    def _analyze_with_baselines(self, selected_baselines: List[str], output_html: str, output_csv: str):
        logger.info(f"Starting analysis with {len(selected_baselines)} baseline(s)")

        all_experiments = self._load_all_experiments()

        baseline_experiments = {
            exp_name: exp
            for exp in all_experiments
            for exp_name in [getattr(exp, 'name', None) or getattr(exp, 'ed_id', '')]
            if exp_name in selected_baselines
        }

        if not baseline_experiments:
            logger.error("Could not find selected baseline experiments - falling back to standard analysis")
            self.analyzer.analyze(output_html, output_csv)
            return

        if not self.analyzer.comparative_orchestrator or not self.analyzer.comparative_orchestrator.baseline_manager:
            logger.error("Comparative orchestrator not initialized - falling back to standard analysis")
            self.analyzer.analyze(output_html, output_csv)
            return

        for baseline_name, exp in baseline_experiments.items():
            self.analyzer.comparative_orchestrator.baseline_manager.register_user_baseline(exp, baseline_name)

        logger.info("Running analysis with selected baselines...")
        self.analyzer.analyze(output_html, output_csv)

        if self.server.running:
            self.server.stop()

        logger.info(f"Analysis complete. Report generated: {output_html}")

    def clear_baseline_selection(self):
        self.baseline_selector.clear_selection()
        logger.info("Cleared baseline selection")

