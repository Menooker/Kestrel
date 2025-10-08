
from flask import Flask, render_template, request, jsonify
import subprocess
import os
import threading

app = Flask(__name__)
app.secret_key = 'your_secret_key'

process_info = {
    'proc': None,
    'step': None,
    'log': []
}

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
    proc1 = subprocess.Popen(cmd1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, universal_newlines=True)
    process_info['proc'] = proc1
    for line in proc1.stdout:
        process_info['log'].append('[转录] ' + line)
    proc1.wait()
    process_info['proc'] = None

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
    proc2 = subprocess.Popen(cmd2, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, universal_newlines=True)
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

@app.route('/progress')
def progress():
    log = process_info.get('log', [])
    content = ''.join(log)
    running = process_info['proc'] is not None or process_info['step'] is not None
    step = process_info.get('step')
    return jsonify({'output': content, 'running': running, 'step': step})

if __name__ == '__main__':
    app.run(debug=True)