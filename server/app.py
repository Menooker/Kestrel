import argparse
from flask import Flask, render_template, request, jsonify, make_response
import subprocess
import os
import threading
import re
import signal
import time
import unicodedata
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


def _set_python_encoding_env(envr):
    """Inject Python UTF-8 encoding variables into an env dict."""
    envr["PYTHONIOENCODING"] = "utf-8"
    envr["PYTHONUTF8"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument('--path-env', type=str, required=True, help='Path enviroment variable')
parser.add_argument('--llama-server', type=str, required=True, help='Path to llama-server executable')
parser.add_argument('--model', type=str, required=True, help='Path to model GGUF file')
parser.add_argument('--proxy', type=str, default="http://127.0.0.1:8010", help='Proxy for API requests')
args, unknown = parser.parse_known_args()

app = Flask(__name__)
app.secret_key = 'your_secret_key'

LLAMA_SERVER_CMD = [
    args.llama_server,
    "-m", args.model,
    "-c", "12800",
    "-fa", "on",
    "-t", "20",
    "--jinja",
    "-ctk", "q8_0",
    "-ctv", "q8_0",
    "--parallel", "1",
    "--host", "0.0.0.0",
    "--fit", "on",
    "--fit-target", "256",
    "--no-mmap",
    "-cram", "0",
    "--spec-type", "draft-mtp",
    "--spec-draft-n-max", "3",
    "--chat-template-kwargs", "{\"enable_thinking\":false}",
]
LLAMA_BASE_URL = "http://127.0.0.1:8080/v1/"

process_info = {
    'proc': None,
    'step': None,
    'log': []
}

def is_working():
    return process_info['proc'] is not None or process_info.get('step') is not None

def sanitize_filename(name: str) -> str:
    """
    Allow Unicode characters (including Chinese). Only remove/replace characters
    that are illegal on Windows file systems and control chars. Also strip trailing
    spaces and dots.
    """
    if not name:
        return "file"
    # Normalize to composed form
    name = unicodedata.normalize('NFKC', name)
    # Replace characters illegal on Windows and control chars with underscore
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name)
    # Collapse multiple underscores
    name = re.sub(r'_+', '_', name)
    # Strip trailing spaces and dots (invalid on Windows filenames)
    name = name.rstrip(' .')
    # Fallback if empty after cleaning
    if not name:
        return "file"

    return name

def start_llama_server():
    creationflags = 0
    if os.name == 'nt':
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
    return subprocess.Popen(
        LLAMA_SERVER_CMD,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
    )

def wait_for_llama_server(proc, timeout=120):
    deadline = time.time() + timeout
    last_error = None
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited early with code {proc.returncode}")
        try:
            with urlopen(LLAMA_BASE_URL + "models", timeout=2):
                return
        except (URLError, HTTPError, OSError) as e:
            last_error = e
            time.sleep(1)
    raise RuntimeError(f"llama-server not ready: {last_error}")

def stop_llama_server(proc):
    if proc is None:
        print("No llama-server process to stop.")
        return
    try:
        print("Sent terminate signal to llama-server.")
        proc.terminate()
    except Exception:
        pass
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        print("llama-server did not terminate in time, killing...")
        try:
            proc.kill()
        except Exception:
            pass
    print("tried kill")
    

def run_task(video_path, context_file, batchsize, hint, output_file):
    base_dir = os.path.dirname(video_path)
    filename = os.path.splitext(os.path.basename(video_path))[0]
    video_ext = os.path.splitext(video_path)[1][1:] if '.' in os.path.basename(video_path) else 'mp4'
    process_info['log'] = []
    # Step 1: transcribe_subsai.py，传递参数
    cmd1 = [
        'python', '-u', 'transcribe_subsai.py',
        '--base', base_dir,
        '--files', filename,
        '--video_ext', video_ext
    ]
    process_info['step'] = 'transcribe'
    envr = os.environ.copy()
    envr["PATH"] = args.path_env + os.pathsep + envr.get("PATH", "")
    _set_python_encoding_env(envr)
    envr["HF_HUB_OFFLINE"] = "1"  # 禁止 transformers 从 Hugging Face Hub 下载模型
    proc1 = subprocess.Popen(cmd1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, universal_newlines=True, env=envr, encoding='utf-8')
    process_info['proc'] = proc1
    for line in proc1.stdout:
        process_info['log'].append('[转录] ' + line)
    proc1.wait()
    process_info['proc'] = None

    envr = os.environ.copy()
    envr["HTTP_PROXY"] = args.proxy
    envr["HTTPS_PROXY"] = args.proxy
    envr["http_proxy"] = args.proxy
    envr["https_proxy"] = args.proxy
    _set_python_encoding_env(envr)
    llama_proc = start_llama_server()
    process_info['step'] = 'translate'
    try:
        wait_for_llama_server(llama_proc)
        cmd2 = [
            'python', '-u', 'translate_ollama.py',
            '--base', base_dir,
            '--api', 'openai',
            '--base-url', LLAMA_BASE_URL,
            '--window', '15',
            '--batchsize', str(batchsize),
            '-l', filename,
            '--hint', hint
        ]
        if context_file:
            cmd2 += ['--context', context_file]
        proc2 = subprocess.Popen(cmd2, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, bufsize=1, universal_newlines=True,
                                 env=envr, encoding='utf-8')
        process_info['proc'] = proc2
        for line in proc2.stdout:
            process_info['log'].append('[翻译] ' + line)
        proc2.wait()
        process_info['proc'] = None
    finally:
        stop_llama_server(llama_proc)
        process_info['step'] = None

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start():
    if is_working():
        return jsonify({'error': '已有任务在运行，请稍后再试。'}), 400
    video_path = request.form['video_path']
    context_file = request.form.get("context_file", "").strip()
    batchsize = request.form['batchsize']
    hint = request.form['hint']
    process_info['step'] = 'starting'
    process_info['log'] = []

    thread = threading.Thread(target=run_task, args=(video_path, context_file, batchsize, hint, None))
    thread.start()
    return jsonify({'status': 'started'})

@app.route('/download', methods=['POST'])
def download():
    if is_working():
        return jsonify({'error': '已有任务在运行，请稍后再试。'}), 400

    url = request.form.get('download_url', '').strip()
    filename = request.form.get('download_filename', '').strip()
    if not url or not filename:
        return jsonify({'error': '请提供 URL 和 文件名'}), 400

    filename = sanitize_filename(filename)
    dest_dir = r"C:\shared"
    try:
        os.makedirs(dest_dir, exist_ok=True)
    except Exception as e:
        return jsonify({'error': f'无法创建目录 {dest_dir}: {e}'}), 500

    dest_path = os.path.join(dest_dir, filename)

    # build you-get command using -O as requested
    youget_cmd = ['you-get', '-O', dest_path, url]

    # prepare env (include path env if provided)
    envr = os.environ.copy()
    envr["PATH"] = args.path_env + os.pathsep + envr.get("PATH", "")
    _set_python_encoding_env(envr)

    # clear previous log and set step
    process_info['log'] = []
    process_info['step'] = 'download'

    def worker():
        try:
            process_info['proc'] = subprocess.Popen(youget_cmd, stdout=subprocess.PIPE,
                                                    stderr=subprocess.STDOUT, universal_newlines=True,
                                                    bufsize=1, env=envr, encoding='utf-8')
            proc = process_info['proc']
            process_info['log'].append(f"> {' '.join(youget_cmd)}\n")
            for line in proc.stdout:
                process_info['log'].append('[下载] ' + line)
            proc.wait()
            process_info['log'].append(f"\n=== you-get exited with code {proc.returncode} ===\n")
            # 合并音频视频为mp4
            if os.path.exists(dest_path+"[00].mp4") and os.path.exists(dest_path+"[01].mp4"):
                subprocess.run(["ffmpeg", "-i", dest_path+"[00].mp4", "-i", dest_path+"[01].mp4", "-c:v", "copy", "-c:a", "aac", dest_path + ".mp4"],
                               env=envr, encoding='utf-8')
        except Exception as e:
            process_info['log'].append(f"\n*** 下载异常: {e} ***\n")
        finally:
            process_info['proc'] = None
            process_info['step'] = None

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return jsonify({'status': 'download_started'})

@app.route('/translate', methods=['POST'])
def translate_only():
    # prevent starting a new job while another proc is active
    if process_info.get('proc') is not None:
        return jsonify({'error': '已有任务在运行，请稍后再试。'}), 400

    video_path = request.form.get('video_path', '').strip()
    context_file = request.form.get("context_file", "").strip()
    batchsize = request.form.get('batchsize', '50').strip()
    hint = request.form.get('hint', '').strip()

    if not video_path:
        return jsonify({'error': '未提供视频路径'}), 400

    base_dir = os.path.dirname(video_path)
    filename = os.path.splitext(os.path.basename(video_path))[0]

    # prepare env for translate (reuse proxy handling)
    envr = os.environ.copy()
    envr["HTTP_PROXY"] = args.proxy
    envr["HTTPS_PROXY"] = args.proxy
    envr["http_proxy"] = args.proxy
    envr["https_proxy"] = args.proxy
    # make sure PATH from args.path_env is included if provided
    envr["PATH"] = args.path_env + os.pathsep + envr.get("PATH", "")
    _set_python_encoding_env(envr)
    # clear previous log and set step
    process_info['step'] = 'starting'
    process_info['step'] = 'translate'

    def worker():
        llama_proc = None
        try:
            cmd = [
                'python', '-u', 'translate_ollama.py',
                '--base', base_dir,
                '--api', 'openai',
                '--window', '15',
                '--base-url', LLAMA_BASE_URL,
                '--batchsize', str(batchsize),
                '-l', filename,
                '--hint', hint
            ]
            if context_file:
                cmd += ['--context', context_file]
            llama_proc = start_llama_server()
            wait_for_llama_server(llama_proc)
            process_info['proc'] = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                                    universal_newlines=True, bufsize=1, env=envr, encoding='utf-8')
            proc = process_info['proc']
            process_info['log'].append(f"> {' '.join(cmd)}\n")
            for line in proc.stdout:
                process_info['log'].append('[翻译] ' + line)
            proc.wait()
            process_info['log'].append(f"\n=== translate_ollama.py exited with code {proc.returncode} ===\n")
        except Exception as e:
            process_info['log'].append(f"\n*** 翻译异常: {e} ***\n")
        finally:
            process_info['proc'] = None
            process_info['step'] = None
            stop_llama_server(llama_proc)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return jsonify({'status': 'translate_started'})

@app.route('/progress')
def progress():
    log = process_info.get('log', [])
    content = ''.join(log)
    running = process_info['proc'] is not None or process_info.get('step') is not None
    step = process_info.get('step')
    resp = make_response(jsonify({'output': content, 'running': running, 'step': step}))
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    resp.headers['Content-Type'] = 'application/json; charset=utf-8'
    return resp

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
