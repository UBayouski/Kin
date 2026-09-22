"""Where Vector's own ears meet Kin's brain.

Saying "Hey Vector" or pressing his back button reaches Vector's built-in voice system, run by wire-pod.
wire-pod tries his built-in commands first ("I love you", "give me a fist bump", "go to sleep"), and those
stay entirely his. For speech it does not recognise, wire-pod can ask an OpenAI-compatible endpoint what
to say, and then Vector says it in his own voice with his own talking animation.

This is that endpoint. It answers at once with a short phrase ("On it!") so Vector acknowledges the
request as himself, and puts what he heard in a durable queue. The service picks it up a moment later
and Kin does the real work. One brain, whichever way the person chooses to talk to him.

Bound to 127.0.0.1: only wire-pod on the same machine can reach it.
"""

import json
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from kin import config, wirepod
from kin.memory import Memory

_last_ack = ""
_lock = threading.Lock()


def pick_ack() -> str:
    """A random acknowledgement, never the same one twice running."""
    global _last_ack
    with _lock:
        choices = [p for p in config.ack_phrases() if p != _last_ack] or list(config.ack_phrases())
        _last_ack = random.choice(choices)
        return _last_ack


def transcript_of(body: dict) -> str:
    """What the person said: the last user message of the chat request."""
    for message in reversed(body.get("messages") or []):
        if message.get("role") == "user":
            content = message.get("content") or ""
            if isinstance(content, list):  # the multi-part form
                content = " ".join(part.get("text", "") for part in content if isinstance(part, dict))
            return " ".join(str(content).split())
    return ""


def _queue(heard: str) -> str:
    """Put what Vector heard in the durable queue for the service. Returns what he should say meanwhile."""
    if not heard:
        return "Sorry, I did not catch that."
    ack = pick_ack()
    memory = Memory()  # this request thread's own connection
    try:
        memory.heard_add(heard, ack)
    finally:
        memory.close()
    return ack


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:  # keep the service log for Kin's own lines
        pass

    def do_GET(self) -> None:
        if self.path.rstrip("/").endswith("/health"):
            self._send(200, "application/json", b'{"status": "ok"}')
        elif self.path.rstrip("/").endswith("/ack"):
            self._send(200, "text/plain; charset=utf-8", pick_ack().encode())
        elif self.path.rstrip("/").endswith("/handover"):
            # wire-pod's takeover script (see kin/wirepod.py): a built-in command Kin answers better, weather
            # first. The script cannot pass the words along, so they are read from wire-pod's log.
            try:
                heard = wirepod.heard_just_now()
            except Exception:
                heard = ""
            ack = _queue(heard)
            self._send(200, "text/plain; charset=utf-8", (ack if heard else "").encode())
        else:
            self._send(404, "text/plain", b"not found")

    def do_POST(self) -> None:
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self._send(404, "text/plain", b"not found")
            return
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        except ValueError:
            self._send(400, "text/plain", b"bad json")
            return

        ack = _queue(transcript_of(body))

        if body.get("stream"):
            self._stream(ack, body.get("model") or "kin")
        else:
            self._send(200, "application/json", json.dumps(_completion(ack, body.get("model") or "kin")).encode())

    def _stream(self, text: str, model: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        for delta, finish in (({"role": "assistant", "content": ""}, None), ({"content": text}, None), ({}, "stop")):
            chunk = {"id": "kin-ack", "object": "chat.completion.chunk", "created": int(time.time()), "model": model,
                     "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _send(self, status: int, content_type: str, payload: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def _completion(text: str, model: str) -> dict:
    return {"id": "kin-ack", "object": "chat.completion", "created": int(time.time()), "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]}


def start(port: int | None = None) -> ThreadingHTTPServer:
    """Serve in a daemon thread. Returns the server (its `server_address[1]` is the port actually bound)."""
    server = ThreadingHTTPServer(("127.0.0.1", config.gateway_port() if port is None else port), _Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True, name="Kin gateway").start()
    return server
