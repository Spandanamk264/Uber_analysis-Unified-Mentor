from http.server import BaseHTTPRequestHandler
import os

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type','text/plain')
        self.end_headers()
        self.wfile.write('Vercel works, but Streamlit requires a persistent server (WebSockets), which Vercel Serverless does not support. Please use Streamlit Cloud or Render.'.encode('utf-8'))
        return
