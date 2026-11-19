import base64
import hashlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).parent / 'forensics_download'
ROOT.mkdir(exist_ok=True)

class Handler(BaseHTTPRequestHandler):
    def cors(self):
        self.send_header('Access-Control-Allow-Origin', 'https://jupyter.anruicloud.com')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Access-Control-Request-Private-Network')
        self.send_header('Access-Control-Allow-Private-Network', 'true')

    def do_OPTIONS(self):
        self.send_response(204)
        self.cors()
        self.end_headers()

    def do_POST(self):
        name = parse_qs(urlparse(self.path).query).get('name', [''])[0]
        if not name or Path(name).name != name or not name.endswith(('.json', '.npz', '.mp4', '.py', '.log')):
            self.send_error(400)
            return
        data = base64.b64decode(self.rfile.read(int(self.headers['Content-Length'])))
        target = ROOT / name
        with target.open('xb') as stream:
            stream.write(data)
        result = json.dumps({'name': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}).encode()
        self.send_response(200)
        self.cors()
        self.end_headers()
        self.wfile.write(result)

ThreadingHTTPServer(('127.0.0.1', 18947), Handler).serve_forever()
