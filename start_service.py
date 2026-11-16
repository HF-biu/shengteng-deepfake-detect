"""Ascend socket service using the original Content-Length/Input-Type framing."""
import argparse
import json
import socket
import tempfile
from pathlib import Path

import cv2
import numpy as np

from ascend_infer import load_config, infer_frames, infer_path
from ais_infer import Inference

DELIMITER = b'\n\n\n\n'
MAX_HEADER = 8192
MAX_PAYLOAD = 256 * 1024 * 1024


def read_request(client):
    buffer = b''
    while DELIMITER not in buffer:
        packet = client.recv(4096)
        if not packet:
            raise ValueError('Connection closed before request header')
        buffer += packet
        if len(buffer.split(DELIMITER, 1)[0]) > MAX_HEADER:
            raise ValueError('Request header too large')
    header, data = buffer.split(DELIMITER, 1)
    fields = dict(line.split(':', 1) for line in header.decode('utf-8').splitlines())
    length = int(fields['Content-Length'].strip())
    input_type = fields['Input-Type'].strip()
    if not 0 < length <= MAX_PAYLOAD:
        raise ValueError('Content-Length out of range')
    if input_type not in ('image', 'video'):
        raise ValueError('Input-Type must be image or video')
    data = bytearray(data)
    if len(data) > length:
        raise ValueError('Payload exceeds Content-Length')
    while len(data) < length:
        packet = client.recv(min(65536, length - len(data)))
        if not packet:
            raise ValueError('Connection closed before complete payload')
        data.extend(packet)
    return bytes(data), input_type


def handle_client_connection(client, engine):
    try:
        client.settimeout(60)
        data, input_type = read_request(client)
        if input_type == 'image':
            frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError('Cannot decode image')
            results = infer_frames(engine, [frame])
        else:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'input.mp4'
                path.write_bytes(data)
                results = infer_path(engine, path, 'video')
        # Preserve numeric responses for existing clients.
        response = str(max(item['score'] for item in results))
    except Exception as exc:
        response = json.dumps({'error': str(exc)}, ensure_ascii=False)
    try:
        client.sendall(response.encode('utf-8'))
    except OSError:
        pass  # A disconnected client must not stop the service.
    finally:
        client.close()


def start_server(host='0.0.0.0', port=54321, config=None):
    # ACL sessions stay on their creating thread; requests are serialized.
    with Inference(config or load_config()) as engine, socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen(10)
        print('Ascend service listening on {}:{}'.format(host, port))
        while True:
            client, _ = server.accept()
            handle_client_connection(client, engine)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=54321)
    parser.add_argument('--config')
    parser.add_argument('--device-id', type=int)
    args = parser.parse_args()
    start_server(args.host, args.port, load_config(args.config, args.device_id))
