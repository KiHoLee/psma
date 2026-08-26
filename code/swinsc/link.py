"""Minimal transport between a standalone transmitter process and receiver process.

Frame = 8-byte big-endian length + torch.save() bytes of {"t_x": float16 tensor, "meta": dict}.
Works over a file (offline hand-off) or a TCP socket (live link). The physical channel
is emulated by whoever owns the link (see scripts/run_rx.py --channel / --snr).
"""
import io
import socket
import struct
import torch


def encode_frame(t_x, meta=None):
    buf = io.BytesIO()
    torch.save({"t_x": t_x.detach().cpu().half(), "meta": meta or {}}, buf)
    payload = buf.getvalue()
    return struct.pack(">Q", len(payload)) + payload


def decode_frame(payload):
    d = torch.load(io.BytesIO(payload), map_location="cpu")
    return d["t_x"].float(), d["meta"]


def write_file(path, t_x, meta=None):
    with open(path, "wb") as f:
        f.write(encode_frame(t_x, meta))


def read_file(path):
    with open(path, "rb") as f:
        n = struct.unpack(">Q", f.read(8))[0]
        return decode_frame(f.read(n))


def _recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(1 << 20, n - len(buf)))
        if not chunk:
            raise ConnectionError("socket closed")
        buf += chunk
    return bytes(buf)


def send_tcp(host, port, t_x, meta=None):
    with socket.create_connection((host, port)) as s:
        s.sendall(encode_frame(t_x, meta))
        ack = _recv_exact(s, 8)
        return struct.unpack(">Q", ack)[0]


def serve_tcp(host, port, handler):
    """handler(t_x, meta) -> int status; blocks forever."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((host, port)); srv.listen(1)
        print(f"[rx] listening on {host}:{port}", flush=True)
        while True:
            conn, addr = srv.accept()
            with conn:
                n = struct.unpack(">Q", _recv_exact(conn, 8))[0]
                t_x, meta = decode_frame(_recv_exact(conn, n))
                status = handler(t_x, meta)
                conn.sendall(struct.pack(">Q", int(status)))
