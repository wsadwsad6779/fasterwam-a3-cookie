import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

FILE = Path(r"C:\Users\ASUS\Documents\Qoder\2026-10-31\b462d38d\_clone_default\restore_20261117\reconstructed_normalizer\final_gpu_mechanism_worker_v6_20261118.py")
DATA = FILE.read_bytes()
SHA = hashlib.sha256(DATA).hexdigest()


class H(BaseHTTPRequestHandler):
    def cors(self):
        self.send_header('Access-Control-Allow-Origin', 'https://jupyter.anruicloud.com')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Access-Control-Request-Private-Network')
        self.send_header('Access-Control-Allow-Private-Network', 'true')

    def do_OPTIONS(self):
        self.send_response(204)
        self.cors()
        self.end_headers()

    def do_GET(self):
        self.send_response(200)
        self.cors()
        self.send_header('Content-Type', 'application/octet-stream')
        self.send_header('Content-Length', str(len(DATA)))
        self.send_header('X-Sha256', SHA)
        self.end_headers()
        self.wfile.write(DATA)

    def log_message(self, *a):
        pass


print('serving', len(DATA), SHA, flush=True)
ThreadingHTTPServer(('127.0.0.1', 18948), H).serve_forever()
