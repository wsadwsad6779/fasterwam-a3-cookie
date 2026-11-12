import functools
import http.server
import os

ROOT = os.path.dirname(os.path.abspath(__file__))


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Private-Network", "true")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def translate_path(self, path):
        return os.path.join(ROOT, path.split("?")[0].split("#")[0].lstrip("/"))

    def log_message(self, fmt, *args):
        print(fmt % args, flush=True)


with http.server.ThreadingHTTPServer(("127.0.0.1", 8765), functools.partial(Handler, directory=ROOT)) as srv:
    srv.serve_forever()
