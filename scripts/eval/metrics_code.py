
import os
import subprocess
import tempfile


def run(code, tests, setup=""):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write((setup or "") + "\n" + (code or "") + "\n")
        for t in tests:
            f.write(f"assert {t}\n")
        path = f.name
    try:
        r = subprocess.run(
            ["python3", path],
            timeout=10,
            capture_output=True,
        )
        ok = r.returncode == 0
    except Exception:
        ok = False
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    return ok


def evaluate_code(preds, data):
    passed = 0
    total = len(data)
    for ex in data:
        ex_id = str(ex["task_id"])
        if run(preds.get(ex_id, ""), ex.get("test_list", []), ex.get("test_setup_code", "")):
            passed += 1
    return {"pass_rate": passed / total if total else 0.0}
