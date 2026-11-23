import http.server
import os
import socketserver
import urllib.parse

ROOT = r'C:\Users\ASUS\Documents\Qoder\2026-10-31\b462d38d\_clone_default'


class H(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=ROOT, **k)

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Private-Network', 'true')
        self.send_header('Access-Control-Allow-Headers',
                         'Content-Type, Access-Control-Request-Private-Network')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS, POST')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def do_POST(self):
        q = urllib.parse.urlparse(self.path)
        if q.path != '/save':
            self.send_response(404)
            self.end_headers()
            return
        rel = urllib.parse.parse_qs(q.query).get('rel', [''])[0]
        n = int(self.headers.get('Content-Length') or 0)
        body = self.rfile.read(n)
        dst = os.path.join(ROOT, rel.replace('/', os.sep))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, 'wb') as f:
            f.write(body)
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(('{"saved":%d}' % len(body)).encode())

    def log_message(self, *a):
        pass


socketserver.ThreadingTCPServer.allow_reuse_address = True
with socketserver.ThreadingTCPServer(('127.0.0.1', 18767), H) as httpd:
    httpd.serve_forever()