import http.server
import socketserver

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

    def log_message(self, *a):
        pass


socketserver.ThreadingTCPServer.allow_reuse_address = True
with socketserver.ThreadingTCPServer(('127.0.0.1', 18767), H) as httpd:
    httpd.serve_forever()