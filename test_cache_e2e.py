"""End-to-end cache persistence test for all local models.

For each model:
1. Clear disk cache
2. Start server (with TurboQuant for Qwen models)
3. Cold request via opencode run (cache miss)
4. Follow-up request (in-memory cache hit)
5. Restart server
6. Follow-up request (disk-loaded cache hit)
7. Verify all responses are coherent and cache hit logs are correct
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

MODELS_DIR = Path.home() / ".lmstudio/models/mlx-community"
CACHE_DIR = Path.home() / ".mlx_vlm/cache"
SERVER_PORT = 8080
SERVER_URL = f"http://127.0.0.1:{SERVER_PORT}/v1/chat/completions"

MODELS = sorted(MODELS_DIR.iterdir())


def cache_dir_for_model(model_path: Path) -> Path:
    return CACHE_DIR / str(model_path).replace("/", "_")


def is_qwen(model_path: Path) -> bool:
    return "qwen" in model_path.name.lower()


def server_args(model_path: Path) -> list[str]:
    args = [
        sys.executable, "-m", "mlx_vlm.server",
        "--model", str(model_path),
        "--port", str(SERVER_PORT),
    ]
    if is_qwen(model_path):
        args += ["--kv-bits", "3.5"]
    return args


def start_server(model_path: Path) -> subprocess.Popen:
    proc = subprocess.Popen(
        server_args(model_path),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(Path.home() / "projects/mlx-vlm"),
    )
    # Wait for server to be ready
    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            import urllib.request
            urllib.request.urlopen(f"http://127.0.0.1:{SERVER_PORT}/v1/models", timeout=2)
            return proc
        except Exception:
            time.sleep(1)
            if proc.poll() is not None:
                out = proc.stdout.read()
                raise RuntimeError(f"Server exited early:\n{out}")
    raise TimeoutError("Server did not start within 60s")


def stop_server(proc: subprocess.Popen) -> str:
    """Stop server and return its stdout."""
    proc.send_signal(signal.SIGINT)
    try:
        out, _ = proc.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, _ = proc.communicate()
    return out


def send_request(model_path: Path, messages: list[dict], max_tokens: int = 30) -> tuple[str, dict]:
    """Send a streaming chat request via curl, return (response_text, last_usage)."""
    import urllib.request

    payload = json.dumps({
        "model": str(model_path),
        "messages": messages,
        "stream": True,
        "max_tokens": max_tokens,
    }).encode()

    req = urllib.request.Request(
        SERVER_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=120)
    text_parts = []
    usage = {}
    for line in resp:
        line = line.decode().strip()
        if line.startswith("data: {"):
            data = json.loads(line[6:])
            content = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
            if content:
                text_parts.append(content)
            if "usage" in data:
                usage = data["usage"]
    return "".join(text_parts), usage


def opencode_run(model_path: Path, message: str, continue_session: bool = False) -> str:
    """Run opencode CLI and return the response."""
    model_id = f"mlx-vlm/{model_path}"
    cmd = ["opencode", "run", "-m", model_id, message]
    if continue_session:
        cmd.insert(2, "-c")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=180,
        cwd=str(Path.home() / "projects/mlx-vlm"),
    )
    if result.returncode != 0:
        raise RuntimeError(f"opencode failed: {result.stderr}")
    return result.stdout


def extract_cache_lines(server_output: str) -> list[str]:
    """Extract cache-related log lines."""
    keywords = ["Cache HIT", "Cache MISS", "generate_step", "Cache: loaded", "Cache: saved"]
    return [l for l in server_output.splitlines() if any(k in l for k in keywords)]


def run_test(model_path: Path):
    model_name = model_path.name
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"{'='*60}")

    # Step 0: Clear disk cache
    cd = cache_dir_for_model(model_path)
    if cd.exists():
        import shutil
        shutil.rmtree(cd)
    print(f"  Cleared disk cache: {cd}")

    # Step 1: Start server (cold)
    print(f"  Starting server...")
    proc = start_server(model_path)
    print(f"  Server started (pid={proc.pid})")

    try:
        # Step 2: First opencode request (cache miss)
        print(f"  Request 1 (cold, via opencode run)...")
        resp1 = opencode_run(model_path, "say hello")
        print(f"    Response: {resp1.strip()[-200:]}")

        # Step 3: Second opencode request (in-memory hit)
        print(f"  Request 2 (in-memory hit, via opencode run -c)...")
        resp2 = opencode_run(model_path, "What is the capital of France?", continue_session=True)
        print(f"    Response: {resp2.strip()[-200:]}")

    finally:
        # Step 4: Stop server (triggers disk save)
        print(f"  Stopping server (triggers disk save)...")
        out1 = stop_server(proc)

    cache_lines1 = extract_cache_lines(out1)
    for line in cache_lines1:
        print(f"    {line.strip()}")

    # Step 5: Restart server (loads from disk)
    print(f"  Restarting server...")
    proc = start_server(model_path)
    print(f"  Server restarted (pid={proc.pid})")

    try:
        # Step 6: Third opencode request (disk-loaded hit)
        print(f"  Request 3 (disk-loaded hit, via opencode run -c)...")
        resp3 = opencode_run(model_path, "Name three colors of the rainbow.", continue_session=True)
        print(f"    Response: {resp3.strip()[-200:]}")

    finally:
        print(f"  Stopping server...")
        out2 = stop_server(proc)

    cache_lines2 = extract_cache_lines(out2)
    for line in cache_lines2:
        print(f"    {line.strip()}")

    # Verify
    has_miss = any("Cache MISS" in l for l in cache_lines1)
    has_mem_hit = any("source=memory" in l for l in cache_lines1)
    has_disk_hit = any("source=disk" in l for l in cache_lines2)
    has_disk_save = any("Cache: saved" in l for l in cache_lines1)

    print(f"\n  Results:")
    print(f"    Cold miss:       {'PASS' if has_miss else 'FAIL'}")
    print(f"    In-memory hit:   {'PASS' if has_mem_hit else 'FAIL'}")
    print(f"    Disk save:       {'PASS' if has_disk_save else 'FAIL'}")
    print(f"    Disk-loaded hit: {'PASS' if has_disk_hit else 'FAIL'}")

    return has_miss and has_mem_hit and has_disk_save and has_disk_hit


def main():
    print(f"Found {len(MODELS)} models:")
    for m in MODELS:
        print(f"  - {m.name}")

    results = {}
    for model_path in MODELS:
        try:
            passed = run_test(model_path)
            results[model_path.name] = "PASS" if passed else "FAIL"
        except Exception as e:
            print(f"  ERROR: {e}")
            results[model_path.name] = f"ERROR: {e}"

    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'='*60}")
    for name, status in results.items():
        print(f"  {name}: {status}")


if __name__ == "__main__":
    main()
