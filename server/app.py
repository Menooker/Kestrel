import argparse
from flask import Flask, render_template, request, jsonify
import subprocess
import os
import threading
import re

parser = argparse.ArgumentParser()
parser.add_argument('--api-key', type=str, required=True, help='Google Gemini API Key')
parser.add_argument('--path-env', type=str, required=True, help='Path enviroment variable')
parser.add_argument('--proxy', type=str, default="http://127.0.0.1:8010", help='Proxy for API requests')
args, unknown = parser.parse_known_args()

app = Flask(__name__)
app.secret_key = 'your_secret_key'

process_info = {
    'proc': None,
    'step': None,
    'log': []
}

def sanitize_filename(name: str) -> str:
    # keep letters, numbers, space, dot, underscore, hyphen; replace others with underscore
    return re.sub(r'[^A-Za-z0-9 _\.\-]', '_', name).strip()

def run_task(video_path, api_key, batchsize, hint, output_file):
    base_dir = os.path.dirname(video_path)
    filename = os.path.splitext(os.path.basename(video_path))[0]
    video_ext = os.path.splitext(video_path)[1][1:] if '.' in os.path.basename(video_path) else 'mp4'
    process_info['log'] = []
    # Step 1: transcribe_subsai.py，传递参数
    cmd1 = [
        'python', 'transcribe_subsai.py',
        '--base', base_dir,
        '--files', filename,
        '--video_ext', video_ext
    ]
    process_info['step'] = 'transcribe'
    envr = os.environ.copy()
    envr["PATH"] = args.path_env + os.pathsep + envr.get("PATH", "")
    proc1 = subprocess.Popen(cmd1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, universal_newlines=True, env=envr)
    process_info['proc'] = proc1
    for line in proc1.stdout:
        process_info['log'].append('[转录] ' + line)
    proc1.wait()
    process_info['proc'] = None
    api_key = api_key if api_key else args.api_key
    envr = os.environ.copy()
    envr["HTTP_PROXY"] = args.proxy
    envr["HTTPS_PROXY"] = args.proxy
    envr['http_proxy'] = args.proxy
    envr['https_proxy'] = args.proxy
    # Step 2: translate.py
    cmd2 = [
        'python', 'translate.py',
        '--base', base_dir,
        '--key', api_key,
        '--batchsize', str(batchsize),
        '-l', filename,
        '--hint', hint
    ]
    process_info['step'] = 'translate'
    proc2 = subprocess.Popen(cmd2, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, universal_newlines=True, env=envr)
    process_info['proc'] = proc2
    for line in proc2.stdout:
        process_info['log'].append('[翻译] ' + line)
    proc2.wait()
    process_info['proc'] = None
    process_info['step'] = None

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start():
    video_path = request.form['video_path']
    api_key = request.form['api_key']
    batchsize = request.form['batchsize']
    hint = request.form['hint']
    filename = os.path.splitext(os.path.basename(video_path))[0]
    process_info['step'] = None
    process_info['log'] = []

    thread = threading.Thread(target=run_task, args=(video_path, api_key, batchsize, hint, None))
    thread.start()
    return jsonify({'status': 'started'})

@app.route('/download', methods=['POST'])
def download():
    # prevent starting a new job while another proc is active
    if process_info.get('proc') is not None:
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

    # clear previous log and set step
    process_info['log'] = []
    process_info['step'] = 'download'

    def worker():
        try:
            process_info['proc'] = subprocess.Popen(youget_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, bufsize=1, env=envr, encoding='utf-8')
            proc = process_info['proc']
            process_info['log'].append(f"> {' '.join(youget_cmd)}\n")
            for line in proc.stdout:
                process_info['log'].append('[下载] ' + line)
            proc.wait()
            process_info['log'].append(f"\n=== you-get exited with code {proc.returncode} ===\n")
            # 合并音频视频为mp4
            if os.path.exists(dest_path+"[00].mp4") and os.path.exists(dest_path+"[01].mp4"):
                subprocess.run(["ffmpeg", "-i", dest_path+"[00].mp4", "-i", dest_path+"[01].mp4", "-c:v", "copy", "-c:a", "aac", dest_path + ".mp4"], env=envr)
        except Exception as e:
            process_info['log'].append(f"\n*** 下载异常: {e} ***\n")
        finally:
            process_info['proc'] = None
            process_info['step'] = None

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return jsonify({'status': 'download_started'})

@app.route('/progress')
def progress():
    log = process_info.get('log', [])
    content = ''.join(log)
    running = process_info['proc'] is not None or process_info['step'] is not None
    step = process_info.get('step')
    return jsonify({'output': content, 'running': running, 'step': step})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')