"""Publish escapepod.local over IPv4 mDNS so Vector can find wire-pod.

Workaround: on some Windows machines wire-pod's own mDNS broadcaster binds
IPv6 only, so the robot cannot resolve escapepod.local and activation fails
with "Error logging in". Keep this running while Vector is in use.

Some routers (seen on an ASUS AiMesh) drop multicast from 5 GHz clients to
2.4 GHz clients, so the normal multicast answer never reaches the robot.
Pass --unicast <vector-ip> to also send the answer straight to the robot.

usage: python scripts/escapepod_mdns.py [--ip <this-pc-ip>] [--unicast <vector-ip>]
"""

import argparse
import socket
import time

from zeroconf import DNSAddress, DNSOutgoing, IPVersion, ServiceInfo, Zeroconf
from zeroconf.const import _CLASS_IN, _CLASS_UNIQUE, _FLAGS_AA, _FLAGS_QR_RESPONSE, _TYPE_A

MDNS_PORT = 5353
HOSTNAME = "escapepod.local."


def lan_ip() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("192.168.255.255", 1))
        return s.getsockname()[0]


def answer_packet(ip: str) -> bytes:
    out = DNSOutgoing(_FLAGS_QR_RESPONSE | _FLAGS_AA)
    record = DNSAddress(HOSTNAME, _TYPE_A, _CLASS_IN | _CLASS_UNIQUE, 120, socket.inet_aton(ip))
    out.add_answer_at_time(record, 0)
    return out.packets()[0]


parser = argparse.ArgumentParser()
parser.add_argument("--ip", default=None, help="this PC's LAN address (auto-detected)")
parser.add_argument("--unicast", default=None, help="Vector's IP; also send answers directly to it")
args = parser.parse_args()

ip = args.ip or lan_ip()
info = ServiceInfo(
    "_app-proto._tcp.local.",
    "escapepod._app-proto._tcp.local.",
    addresses=[socket.inet_aton(ip)],
    port=8084,
    server=HOSTNAME,
)

zc = Zeroconf(interfaces=[ip], ip_version=IPVersion.V4Only)
zc.register_service(info)
print(f"Publishing escapepod.local -> {ip}. Ctrl+C to stop.")

direct = None
if args.unicast:
    direct = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    direct.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    direct.bind((ip, MDNS_PORT))
    print(f"Also sending unicast answers to {args.unicast} every 2 s.")

try:
    while True:
        if direct:
            direct.sendto(answer_packet(ip), (args.unicast, MDNS_PORT))
            time.sleep(2)
        else:
            time.sleep(3600)
except KeyboardInterrupt:
    pass
finally:
    if direct:
        direct.close()
    zc.unregister_service(info)
    zc.close()
