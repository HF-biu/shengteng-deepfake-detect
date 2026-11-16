"""Static Ascend OM inference, with lazy runtime imports and explicit ownership."""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS = {"retina": "FaceDetector.om", "pscc": "pscc.om",
                  "deepfake": "refine_128.om"}
SAMPLE_SHAPES = {"retina": (3, 640, 640), "pscc": (3, 256, 256),
                 "deepfake": (3, 224, 224)}


def model_path(config, name):
    path = Path(config.get("ascend", {}).get("models", {}).get(name, DEFAULT_MODELS[name]))
    return path if path.is_absolute() else ROOT / path


def device_id(config):
    value = config.get("ascend", {}).get("device_id", 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("ascend.device_id must be a non-negative integer")
    return value


class StaticOMSession:
    """One static NCHW float32 model. Pad only the last batch; trim every output."""

    def __init__(self, config, name, session_factory=None):
        self.session = None
        self.name = name
        path = model_path(config, name)
        selected_device = device_id(config)
        if not path.is_file():
            raise FileNotFoundError("Missing Ascend model: {}".format(path))
        if session_factory is None:
            try:
                from ais_bench.infer.interface import InferSession
            except ImportError as exc:
                raise RuntimeError("Ascend inference requires CANN, aclruntime and ais_bench; "
                                   "see README.md for installation.") from exc
            session_factory = InferSession
        try:
            self.session = session_factory(device_id=selected_device, model_path=str(path))
            inputs = self.session.get_inputs()
            if len(inputs) != 1:
                raise ValueError("{} must have exactly one input".format(name))
            self.shape = tuple(int(n) for n in inputs[0].shape)
            if len(self.shape) != 4 or any(n <= 0 for n in self.shape):
                raise ValueError("{} requires a static NCHW OM, got {}".format(name, self.shape))
            if self.shape[1:] != SAMPLE_SHAPES[name]:
                raise ValueError("{} input shape mismatch: {}".format(name, self.shape))
            # aclruntime exposes a pybind enum, not a NumPy dtype object.
            raw_dtype = inputs[0].datatype
            try:
                dtype = np.dtype(raw_dtype)
            except TypeError:
                dtype = np.dtype(getattr(raw_dtype, 'name', str(raw_dtype).rsplit('.', 1)[-1]))
            if dtype != np.dtype("float32"):
                raise ValueError("{} requires float32 input, got {}".format(name, dtype))
            self.batch_size = self.shape[0]
            self.chunk_size = config.get("batch_size", {}).get(name, self.batch_size)
            if (isinstance(self.chunk_size, bool) or not isinstance(self.chunk_size, int)
                    or not 0 < self.chunk_size <= self.batch_size):
                raise ValueError("batch_size.{} must be between 1 and OM batch {}".format(
                    name, self.batch_size))
        except Exception:
            self.close()
            raise

    def infer_batch(self, samples):
        if self.session is None:
            raise RuntimeError("Session is closed")
        data = np.asarray(samples, dtype=np.float32)
        count = len(data)
        if not 0 < count <= self.batch_size or data.shape[1:] != self.shape[1:]:
            raise ValueError("{} expected 1..{} samples of {}, got {}".format(
                self.name, self.batch_size, self.shape[1:], data.shape))
        feeds = np.zeros(self.shape, dtype=np.float32)
        feeds[:count] = data
        outputs = self.session.infer(feeds=[feeds], mode="static")
        trimmed = []
        for output in outputs:
            output = np.asarray(output)
            if output.ndim == 0 or output.shape[0] != self.batch_size:
                raise ValueError("{} output batch does not match input".format(self.name))
            trimmed.append(output[:count].copy())
        return trimmed

    def iter_batches(self, samples):
        for start in range(0, len(samples), self.chunk_size):
            yield self.infer_batch(samples[start:start + self.chunk_size])

    def close(self):
        session, self.session = self.session, None
        if session is not None:
            session.free_resource()

