import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from utils.ascend_runtime import StaticOMSession, SAMPLE_SHAPES
from ascend_infer import load_config, infer_frames, infer_path, smoke_test
from ais_infer import Inference, FaceStd
from start_service import read_request, handle_client_connection


class FakeSession:
    instances = []
    predictions = 'scalar'
    detect_face = False

    def __init__(self, device_id, model_path):
        self.device_id = device_id
        self.name = {'FaceDetector.om': 'retina', 'pscc.om': 'pscc',
                     'refine_128.om': 'deepfake'}[Path(model_path).name]
        self.shape = (2,) + SAMPLE_SHAPES[self.name]
        self.calls = []
        self.freed = 0
        self.instances.append(self)

    def get_inputs(self):
        return [SimpleNamespace(shape=self.shape, datatype=np.float32)]

    def infer(self, feeds, mode):
        assert mode == 'static'
        data = feeds[0]
        assert data.shape == self.shape and data.dtype == np.float32
        self.calls.append(data.copy())
        if self.name == 'retina':
            loc = np.zeros((2, 16800, 4), np.float32)
            conf = np.zeros((2, 16800, 2), np.float32)
            landmarks = np.zeros((2, 16800, 10), np.float32)
            return [loc, conf, landmarks]
        if self.name == 'pscc':
            return [np.zeros((2, 1, size, size), np.float32) for size in (256, 128, 64, 32)] + [
                np.array([[0., 1.], [1., 0.]], np.float32)]
        if self.predictions == 'logits':
            return [np.array([[1000., 1001.], [1001., 1000.]], np.float32)]
        if self.predictions == 'column':
            return [np.array([[0.2], [0.8]], np.float32)]
        return [np.array([0.2, 0.8], np.float32)]

    def free_resource(self):
        self.freed += 1


@pytest.fixture
def config():
    cfg = load_config()
    cfg['batch_size'] = dict(retina=2, pscc=2, deepfake=2)
    cfg['ascend']['device_id'] = 3
    FakeSession.instances = []
    FakeSession.predictions = 'scalar'
    return cfg


@pytest.fixture
def engine(config):
    with patch('ais_infer.StaticOMSession',
               side_effect=lambda cfg, name: StaticOMSession(cfg, name, FakeSession)):
        with Inference(config) as instance:
            yield instance
    assert all(session.freed == 1 for session in FakeSession.instances)


@pytest.mark.parametrize('count', [0, 1, 2, 3, 5])
def test_padding_order_and_trimming(config, count):
    session = StaticOMSession(config, 'deepfake', FakeSession)
    samples = np.stack([np.full(SAMPLE_SHAPES['deepfake'], i + 1, np.float32)
                        for i in range(count)]) if count else []
    outputs = list(session.iter_batches(samples))
    assert sum(len(output[0]) for output in outputs) == count
    observed = [batch[:, 0, 0, 0] for batch in session.session.calls]
    if count:
        np.testing.assert_equal(np.concatenate(observed)[:count], np.arange(1, count + 1))
    if count % 2:
        assert not session.session.calls[-1][1].any()
    session.close()
    session.close()
    assert FakeSession.instances[-1].freed == 1


@pytest.mark.parametrize('device', [-1, '0', True])
def test_invalid_device(config, device):
    config['ascend']['device_id'] = device
    with pytest.raises(ValueError):
        StaticOMSession(config, 'deepfake', FakeSession)


@pytest.mark.parametrize('batch', [0, -1, 3, True])
def test_invalid_batch_releases_session(config, batch):
    config['batch_size']['deepfake'] = batch
    with pytest.raises(ValueError):
        StaticOMSession(config, 'deepfake', FakeSession)
    assert FakeSession.instances[0].freed == 1


def test_missing_model(config):
    config['ascend']['models']['deepfake'] = 'missing-model.om'
    with pytest.raises(FileNotFoundError):
        StaticOMSession(config, 'deepfake', FakeSession)
    assert not FakeSession.instances


def test_aclruntime_enum_datatype(config):
    from enum import Enum
    dtype = Enum('dtype', ['float32', 'float16'])
    with patch.object(FakeSession, 'get_inputs', return_value=[SimpleNamespace(
            shape=(2, 3, 224, 224), datatype=dtype.float32)]):
        session = StaticOMSession(config, 'deepfake', FakeSession)
    session.close()


def test_compiler_preserves_original_onnx_source():
    from compile_ascend import ONNX_MODELS, make_command
    assert ONNX_MODELS['deepfake'] == 'cfl_128.onnx'
    command = make_command('atc', Path('input model.onnx'), Path('output model'),
                           'Ascend310P3', 'input0', (64, 3, 640, 640))
    assert '--input_shape=input0:64,3,640,640' in command
    assert '--model=input model.onnx' in command
    assert '--soc_version=Ascend310P3' in command


@pytest.mark.parametrize('shape,dtype', [((-1, 3, 224, 224), np.float32),
                                       ((2, 224, 224, 3), np.float32),
                                       ((2, 3, 224, 224), np.float16)])
def test_incompatible_om(config, shape, dtype):
    with patch.object(FakeSession, 'get_inputs', return_value=[SimpleNamespace(shape=shape, datatype=dtype)]):
        with pytest.raises(ValueError):
            StaticOMSession(config, 'deepfake', FakeSession)
    assert FakeSession.instances[0].freed == 1


def test_partial_initialization_releases_models(config):
    def factory(cfg, name):
        if name == 'deepfake':
            raise RuntimeError('Cannot load model')
        return StaticOMSession(cfg, name, FakeSession)
    with patch('ais_infer.StaticOMSession', side_effect=factory):
        with pytest.raises(RuntimeError):
            Inference(config)
    assert FakeSession.instances[0].freed == 1


def test_pipeline_empty_and_no_faces(engine):
    assert infer_frames(engine, []) == []
    results = infer_frames(engine, [np.zeros((300, 400, 3), np.uint8)] * 3)
    assert len(results) == 3
    assert all(result['score'] == 0 for result in results)
    assert {session.device_id for session in FakeSession.instances} == {3}
    assert all(len(session.calls) == 2 for session in FakeSession.instances)
    json.dumps(results)


@pytest.mark.parametrize('form', ['scalar', 'column', 'logits'])
def test_deepfake_output_shapes(engine, form):
    FakeSession.predictions = form
    outputs = engine.fake_infer([FaceStd(i, np.ones((224, 224, 3), np.uint8)) for i in range(3)])
    expected = 1 / (1 + np.exp(-1)) if form == 'logits' else 0.2
    assert outputs[0] == pytest.approx(expected)
    assert outputs[2] == pytest.approx(expected)
    assert len(outputs) == 3


def test_face_pipeline_uses_configured_threshold(engine):
    engine.processers.threshold = 0.1
    face = np.ones((224, 224, 3), np.float32) * 80
    with patch.object(engine.processers, 'get_face', return_value=([face], [[20, 20, 220, 220]])):
        results = infer_frames(engine, [np.ones((300, 300, 3), np.uint8) * 80] * 3,
                               'video', frame_offset=7)
    assert [result['frame_id'] for result in results] == [7, 8, 9]
    assert all(result['state'] == 'Fake' for result in results)
    assert all(result['score'] > 0 for result in results)


def test_video_batches_and_release(engine):
    frames = [np.ones((300, 300, 3), np.uint8)] * 5
    capture = SimpleNamespace(isOpened=lambda: True, release=lambda: None)
    with patch('ascend_infer.cv2.VideoCapture', return_value=capture), \
         patch.object(capture, 'release') as released:
        capture.read = lambda: (True, frames.pop()) if frames else (False, None)
        results = infer_path(engine, Path(__file__), 'video')
    assert [item['frame_id'] for item in results] == list(range(5))
    released.assert_called_once()


class Client:
    def __init__(self, packets):
        self.packets = list(packets)
        self.sent = b''
        self.closed = False
    def recv(self, size):
        return self.packets.pop(0) if self.packets else b''
    def settimeout(self, value):
        pass
    def sendall(self, data):
        self.sent += data
    def close(self):
        self.closed = True


def test_fragmented_socket_request():
    client = Client([b'Content-Length: 3\nInput-Type: image\n\n', b'\n\na', b'bc'])
    assert read_request(client) == (b'abc', 'image')


@pytest.mark.parametrize('packets', [[], [b'Content-Length: 3\nInput-Type: image\n\n\n\na'],
                                    [b'Content-Length: -1\nInput-Type: image\n\n\n\n']])
def test_disconnected_or_invalid_client(packets):
    with pytest.raises(ValueError):
        read_request(Client(packets))


def test_service_image(engine):
    ok, data = cv2.imencode('.png', np.zeros((300, 300, 3), np.uint8))
    assert ok
    payload = data.tobytes()
    header = 'Content-Length: {}\nInput-Type: image\n\n\n\n'.format(len(payload)).encode()
    client = Client([header, payload])
    handle_client_connection(client, engine)
    assert float(client.sent) == 0
    assert client.closed


def test_smoke_test_runs_every_model(config):
    with patch('ascend_infer.StaticOMSession',
               side_effect=lambda cfg, name: StaticOMSession(cfg, name, FakeSession)):
        report = smoke_test(config)
    assert set(report) == {'retina', 'pscc', 'deepfake'}
    assert all(session.freed == 1 for session in FakeSession.instances)
