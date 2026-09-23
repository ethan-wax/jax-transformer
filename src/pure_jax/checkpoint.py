import pathlib

import flax.serialization as serialization


def save_checkpoint(params: dict, path: str | pathlib.Path) -> None:
    """Serialize params to disk."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(serialization.to_bytes(params))


def load_checkpoint(template_params: dict, path: str | pathlib.Path) -> dict:
    """Deserialize params from disk, restoring into the structure of template_params."""
    path = pathlib.Path(path)
    return serialization.from_bytes(template_params, path.read_bytes())
