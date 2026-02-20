import json
import platform
from datetime import datetime
from pathlib import Path
import sklearn

def save_run_info(params: dict, metrics: dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp_utc": datetime.utcnow().isoformat(),
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "params": params,
        "metrics": metrics,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")