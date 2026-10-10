"""A tiny OpenAI-compatible HTTP server that replays scripted chat-completion responses.
Lets tests drive the REAL openai client + our retry/tool-calling code with no network and no key."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer


def tool_call(name, args, call_id="call_1"):
    return {"role": "assistant", "content": None, "tool_calls": [
        {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}


def text(content):
    return {"role": "assistant", "content": content}


class FakeOpenAI:
    def __init__(self, script, status=200):
        self.script, self.status, self.requests = list(script), status, []

    def __enter__(self):
        outer = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append(body)
                if outer.status != 200:
                    data = json.dumps({"error": {"message": "boom", "type": "server_error"}}).encode()
                    self.send_response(outer.status)
                else:
                    msg = outer.script.pop(0) if outer.script else text("(script exhausted)")
                    data = json.dumps({"id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
                                       "choices": [{"index": 0, "finish_reason": "stop", "message": msg}]}).encode()
                    self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass

        self.srv = HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_port}/v1"
        return self

    def __exit__(self, *a):
        self.srv.shutdown()
