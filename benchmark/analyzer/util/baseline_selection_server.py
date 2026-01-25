import json
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from threading import Thread
from typing import List, Callable

logger = logging.getLogger(__name__)


class BaselineSelectionHandler(BaseHTTPRequestHandler):

    selection_callback: Callable[[List[str]], None] = None
    html_content: str = ""
    report_path: str = ""

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path in ('/', '/select'):
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(BaselineSelectionHandler.html_content.encode())
        elif self.path == '/report':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            report_path = Path(BaselineSelectionHandler.report_path)
            if report_path.exists():
                self.wfile.write(report_path.read_text().encode())
            else:
                self.wfile.write(b"<html><body><h1>Report not yet generated</h1></body></html>")
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == '/select_baselines':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            try:
                data = json.loads(post_data.decode('utf-8'))
                selected = data.get('selected_baselines', [])

                if selected and BaselineSelectionHandler.selection_callback:
                    BaselineSelectionHandler.selection_callback(selected)
                    response = {'success': True, 'message': f'Selected {len(selected)} baseline(s)', 'report_url': '/report'}
                else:
                    response = {'success': False, 'message': 'No baselines selected'}

                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())
            except Exception as e:
                logger.error(f"Error processing baseline selection: {e}")
                self.send_error(500, str(e))
        else:
            self.send_error(404)


class BaselineSelectionServer:
    """HTTP server for baseline selection UI"""

    def __init__(self, port: int = 8765):
        self.port = port
        self.server = None
        self.thread = None
        self.running = False

    def start(self, html_content: str, report_path: str, selection_callback: Callable[[List[str]], None]) -> str:
        BaselineSelectionHandler.html_content = html_content
        BaselineSelectionHandler.report_path = report_path
        BaselineSelectionHandler.selection_callback = selection_callback

        self.server = HTTPServer(('localhost', self.port), BaselineSelectionHandler)
        self.running = True
        self.thread = Thread(target=self._run_server, daemon=True)
        self.thread.start()

        url = f"http://localhost:{self.port}"
        logger.info(f"Baseline selection server started at {url}")
        return url

    def _run_server(self):
        while self.running:
            self.server.handle_request()

    def stop(self):
        self.running = False
        if self.server:
            self.server.server_close()
        logger.info("Baseline selection server stopped")
