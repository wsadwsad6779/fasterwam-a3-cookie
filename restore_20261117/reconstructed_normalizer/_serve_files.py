import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, unquote

ROOT = Path(r"C:\Users\ASUS\Documents\Qoder\2026-10-31\b462d38d\_clone_default\restore_20261117\reconstructed_normalizer")


class H(BaseHTTPRequestHandler):
    def cors(self):
        self.send_header('Access-Control-Allow-Origin', 'https://jupyter.anruicloud.com')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Access-Control-Request-Private-Network')
        self.send_header('Access-Control-Allow-Private-Network', 'true')
        self.send_header('Access-Control-Expose-Headers', 'X-Sha256')

    def do_OPTIONS(self):
        self.send_response(204)
        self.cors()
        self.end_headers()

    def do_GET(self):
        name = unquote(urlparse(self.path).path.lstrip('/'))
        target = ROOT / name
        if not name or target.parent != ROOT or not target.is_file():
            self.send_error(404)
            return
        data = target.read_bytes()
        self.send_response(200)
        self.cors()
        self.send_header('Content-Type', 'application/octet-stream')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('X-Sha256', hashlib.sha256(data).hexdigest())
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


print('serving', ROOT, flush=True)
ThreadingHTTPServer(('127.0.0.1', 18949), H).serve_forever()
