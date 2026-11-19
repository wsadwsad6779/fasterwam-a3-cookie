from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
import functools
import os


class H(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Private-Network', 'true')
        self.send_header('Access-Control-Allow-Headers',
                         'Content-Type, Access-Control-Request-Private-Network')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def log_message(self, *a):
        pass


if __name__ == '__main__':
    root = os.path.dirname(os.path.abspath(__file__))
    handler = functools.partial(H, directory=root)
    ThreadingHTTPServer(('127.0.0.1', 18963), handler).serve_forever()
