# app.py
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import json
import uuid
import traceback
import threading
from typing import Dict, Any, List
import os
from api import run_mind_evolution


app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# 保存参数
RUNS: Dict[str, Dict[str, Any]] = {}

# 日志缓冲（批量轮询）
LOGS: Dict[str, Dict[str, Any]] = {}
LOGS_LOCK = threading.Lock()

# 定义历史记录和输出文件的目录
HISTORY_FILE = "history.json"
OUTPUT_DIR = "output_history"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (IOError, json.JSONDecodeError):
        return []

def save_history(history: List[Dict[str, Any]]):
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=4)

def _init_log_channel(run_id: str):
    with LOGS_LOCK:
        if run_id not in LOGS:
            LOGS[run_id] = {"items": [], "done": False}

def _append_log(run_id: str, pretty_line: str, step: Dict[str, Any]):
    try:
        filepath = os.path.join(OUTPUT_DIR, f"{run_id}.txt")
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(pretty_line + '\n')
    except IOError as e:
        print(f"Error: Could not write to log file for run_id {run_id}: {e}")

    with LOGS_LOCK:
        chan = LOGS.get(run_id)
        if not chan:
            chan = {"items": [], "done": False}
            LOGS[run_id] = chan
        seq = len(chan["items"])
        record = {"seq": seq, "pretty": pretty_line, "step": step}
        chan["items"].append(record)
    print(pretty_line, flush=True)

def _mark_done(run_id: str):
    with LOGS_LOCK:
        if run_id in LOGS:
            LOGS[run_id]["done"] = True


# --- MODIFIED FUNCTION START ---
def _format_pretty_line(run_id: str, step: Dict[str, Any]) -> str:
    """
    修改后的版本：会检查并包含 best_plan
    """
    phase = step.get("phase")
    gen_idx = step.get("gen")
    msg = step.get("message", "")
    best = step.get("best_fitness")
    best_plan = step.get("best_plan")  # <-- 新增：获取方案

    line = (
        f"[RUN {run_id[:8]}] phase={phase}"
        + (f", gen={gen_idx}" if gen_idx is not None else "")
        + (f", best={best:.2f}" if isinstance(best, (int, float)) else "")
        + (f" | {msg}" if msg else "")
    )
    
    # 如果 best_plan 存在，将其格式化为紧凑的 JSON 字符串并附加
    if best_plan:
        plan_str = json.dumps(best_plan, ensure_ascii=False, separators=(',', ':'))
        line += f" | Plan: {plan_str}"
        
    return line
# --- MODIFIED FUNCTION END ---


def start_background_run(run_id: str, params: dict):
    """后台线程：消费 api 生成器，每步打印+入缓冲。"""
    try:
        head = f"===== 🧠 MindEvolution started (run_id={run_id}) ====="
        _append_log(run_id, head, {"phase": "meta", "message": "started", "run_id": run_id})

        gen = run_mind_evolution(
            prompt=params["prompt"],
            population_size=params["population_size"],
            generations=params["generations"],
            num_parents=params["num_parents"],
            tournament_size=params["tournament_size"],
            max_retries=params["max_retries"],
        )
        for step in gen:
            pretty = _format_pretty_line(run_id, step)
            _append_log(run_id, pretty, step)

        tail = f"===== ✅ MindEvolution completed (run_id={run_id}) ====="
        _append_log(run_id, tail, {"phase": "final", "message": "completed", "run_id": run_id})
    except Exception as e:
        err = f"===== ❌ MindEvolution error (run_id={run_id}): {e} ====="
        _append_log(run_id, err, {"phase": "error", "message": str(e), "run_id": run_id})
    finally:
        _mark_done(run_id)

@app.route('/', methods=['GET'])
def index():
    history = load_history()
    return render_template('index.html', history=history)

@app.route('/session/<run_id>', methods=['GET'])
def session_page(run_id):
    prompt = request.args.get('prompt', '')
    history = load_history()
    filepath = os.path.join(OUTPUT_DIR, f"{run_id}.txt")
    if run_id not in RUNS and not os.path.exists(filepath):
        return render_template('chat.html', run_id=run_id, prompt=prompt, invalid=True, history=history)
    return render_template('chat.html', run_id=run_id, prompt=prompt, invalid=False, history=history)

@app.route('/healthz', methods=['GET'])
def healthz():
    return jsonify({"status": "ok"})

@app.route('/solve', methods=['POST'])
def solve_vrp():
    try:
        prompt = request.form.get('vrp_problem', '').strip()
        def to_int(name, default):
            v = request.form.get(name, default)
            try: return int(v)
            except Exception: return default

        params = {
            "prompt": prompt,
            "population_size": to_int('population_size', 10),
            "generations": to_int('generations', 6),
            "num_parents": to_int('parents', 5),
            "tournament_size": to_int('tournament_size', 5),
            "max_retries": to_int('retries', 3),
        }

        run_id = str(uuid.uuid4())
        RUNS[run_id] = params
        _init_log_channel(run_id)

        if prompt:
            history = load_history()
            new_entry = {"run_id": run_id, "prompt": prompt}
            history.insert(0, new_entry)
            save_history(history)

        print("\n" + "="*60)
        print("✅ Received /solve")
        for k, v in params.items():
            if k != "prompt": print(f"  - {k}: {v}")
        print(f"  - prompt: {params['prompt'][:120]}..." if params['prompt'] else "  - prompt: <EMPTY>")
        print(f"➡️  run_id: {run_id}")
        print("="*60 + "\n")

        threading.Thread(target=start_background_run, args=(run_id, dict(params)), daemon=True).start()
        return jsonify({"status": "accepted", "run_id": run_id})
    except Exception as e:
        print("❌ /solve error:", e, flush=True)
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/logs/<run_id>', methods=['GET'])
def fetch_logs(run_id: str):
    try:
        cursor = int(request.args.get('cursor', 0))
    except Exception:
        cursor = 0

    with LOGS_LOCK:
        chan = LOGS.get(run_id)
        if chan:
            items: List[Dict[str, Any]] = chan["items"]
            done: bool = chan["done"]
            batch = items[cursor:cursor + 200]
            next_cursor = cursor + len(batch)
            payload = [{"seq": rec["seq"], "pretty": rec["pretty"]} for rec in batch]
            return jsonify({"status": "ok", "logs": payload, "next_cursor": next_cursor, "done": done})

    filepath = os.path.join(OUTPUT_DIR, f"{run_id}.txt")
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f.readlines()]
            if cursor < len(lines):
                batch_lines = lines[cursor:]
                payload = [{"seq": cursor + i, "pretty": line} for i, line in enumerate(batch_lines)]
                next_cursor = len(lines)
                return jsonify({"status": "ok", "logs": payload, "next_cursor": next_cursor, "done": True})
            else:
                return jsonify({"status": "ok", "logs": [], "next_cursor": cursor, "done": True})
        except IOError as e:
            return jsonify({"status": "error", "message": f"Could not read log file: {e}"}), 500
    
    return jsonify({"status": "error", "message": "Invalid run_id"}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)