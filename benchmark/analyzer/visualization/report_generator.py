from datetime import datetime
from typing import Dict, List, Any, Optional

import plotly.graph_objs as go

from analyzer.config.benchmark_config import BenchmarkConfig
from analyzer.visualization.table_generator import TableGenerator
from template.template_loader import TemplateLoader


class ReportGenerator:
    def __init__(self, config: BenchmarkConfig, table_generator: TableGenerator):
        self.config = config
        self.table_generator = table_generator
        self.template_loader = TemplateLoader()

    def generate(self, objective_plots: Dict[str, List[go.Figure]], table_data: Dict[str, List[Dict[str, Any]]],
                 csv_files: Dict[str, str], zip_file: Optional[str]) -> str:
        """Generate complete HTML report"""
        tab_buttons = []
        tab_contents = []

        for idx, (objective, figures) in enumerate(objective_plots.items()):
            tab_id = f'obj_tab_{idx}'
            active_class = 'active' if idx == 0 else ''

            tab_buttons.append(self._create_tab_button(objective, tab_id, active_class))
            tab_contents.append(self._create_tab_content(objective, figures, table_data, csv_files, tab_id, idx == 0))

        global_download = self._create_global_download(zip_file)
        tabs_section = self._build_tabs_section(tab_buttons, tab_contents, global_download)

        context = {'generated_time': datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
                   'experiment_name': self.config.experiment_name,
                   'experiment_description': self.config.experiment_description,
                   'objectives': ', '.join(objective_plots.keys()), 'tabs_section': tabs_section, }

        return self.template_loader.render_template('report_template.html', context=context, inline_assets=True,
                                                    css_files=['report.css'], js_files=['report.js'])

    @staticmethod
    def _create_tab_button(objective: str, tab_id: str, active_class: str) -> str:
        """Create HTML for tab button"""
        return f"<button class='tab-btn {active_class}' onclick=showTab('{tab_id}',this)>{objective}</button>"

    def _create_tab_content(self, objective: str, figures: List[go.Figure], table_data: Dict[str, List[Dict[str, Any]]],
                            csv_files: Dict[str, str], tab_id: str, is_active: bool) -> str:
        """Create HTML for tab content"""
        plots_html = self._create_plots_html(objective, figures, tab_id)
        table_html = self._create_table_html(objective, table_data)
        download_link = self._create_csv_download(objective, csv_files)

        display_style = 'block' if is_active else 'none'
        return (f"<div id='{tab_id}' class='tab-content' style='display:{display_style}'>"
                f"{plots_html}"
                f"{download_link}"
                f"<div class='table-wrapper'>{table_html}</div>"
                f"</div>")

    @staticmethod
    def _create_plots_html(objective: str, figures: List[go.Figure], tab_id: str) -> str:
        """Create HTML for all plots in a tab"""
        plots_html_parts = []
        for plot_idx, fig in enumerate(figures):
            plot_id = f"plot_{tab_id}_{plot_idx}"
            plot_type = "improvement" if plot_idx == 0 else f"plot_{plot_idx}"

            plots_html_parts.append(f"<div class='plot-container'>"
                                    f"<div class='plot-wrapper' id='{plot_id}'>"
                                    f"{fig.to_html(include_plotlyjs=False, full_html=False)}"
                                    f"</div>"
                                    f"<button class='export-btn' onclick='exportPlotAsSVG(\"{plot_id}\", \"{objective}_{plot_type}\")'>Export as SVG</button>"
                                    f"</div>")
        return ''.join(plots_html_parts)

    def _create_table_html(self, objective: str, table_data: Dict[str, List[Dict[str, Any]]]) -> str:
        """Create HTML for table"""
        table_rows = table_data.get(objective, [])
        return self.table_generator.format_table(table_rows, self.config.table_config)

    @staticmethod
    def _create_csv_download(objective: str, csv_files: Dict[str, str]) -> str:
        """Create CSV download link"""
        csv_filename = csv_files.get(objective, '')
        if not csv_filename:
            return ''
        return (f"<div class='download-row'>"
                f"<a class='download-link' href='{csv_filename}' download>Download {objective} CSV</a>"
                f"</div>")

    @staticmethod
    def _create_global_download(zip_file: Optional[str]) -> str:
        """Create global download link for all CSVs"""
        if not zip_file:
            return ''
        return (f"<div class='global-download'>"
                f"<a class='download-link all' href='{zip_file}' download>Download all tables (.zip)</a>"
                f"</div>")

    @staticmethod
    def _build_tabs_section(tab_buttons: List[str], tab_contents: List[str], global_download: str) -> str:
        """Build complete tabs section HTML"""
        return (f"<div class='tabs'>"
                f"{global_download}"
                f"<div class='tab-buttons'>{''.join(tab_buttons)}</div>"
                f"{''.join(tab_contents)}"
                f"</div>")
