from pathlib import Path
import yaml

try:
    import orjson
    def _dumps(obj): return orjson.dumps(obj, option=orjson.OPT_INDENT_2)
except Exception:
    import json
    def _dumps(obj): return json.dumps(obj, indent=2).encode("utf-8")

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}

def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(_dumps(obj))

def resolve_config(args, settings, retr_conf):
    return {
        "models": settings.get("models", {}),
        "decoding": settings.get("decoding", {}),
        "retriever": retr_conf.get("retriever", {}),
        "flags": {
            "k": getattr(args, "k", None),
            "embed_model": getattr(args, "embed_model", None),
            "chat_model": getattr(args, "chat_model", None),
            "labelset": getattr(args, "labelset", None),
        },
    }
