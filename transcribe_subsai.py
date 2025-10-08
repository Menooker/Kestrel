
import argparse
from subsai import SubsAI

parser = argparse.ArgumentParser()
parser.add_argument('--base', type=str, default='D:\\Temp', help='视频文件所在目录')
parser.add_argument('--files', nargs='+', required=True, help='视频文件名列表（不带扩展名）')
parser.add_argument('--video_ext', type=str, default='mp4', help='视频扩展名')
args = parser.parse_args()

subs_ai = SubsAI()
base = args.base
files = args.files
video_ext = args.video_ext


def transcribe(name):
    file = f'{base}\\{name}.{video_ext}'
    print(name, "=======================")
    model = subs_ai.create_model('guillaumekln/faster-whisper', {'model_size_or_path': 'large-v3', 'device': "cuda", "compute_type": "float16", "language": "ja", "suppress_blank": True, "vad_filter": True,
                                                                 "vad_parameters":  {
                                                                     "min_speech_duration_ms": 250,
                                                                     "max_speech_duration_s": float("inf"),
                                                                     "min_silence_duration_ms": 2000,
                                                                     "speech_pad_ms": 400,
                                                                 },
                                                                 })  # ,
    subs = subs_ai.transcribe(file, model)
    subs.save(f'{base}\\{name}.jp.srt')


for n in files:
    transcribe(n)
