from subsai import SubsAI

subs_ai = SubsAI()

base = 'D:\\Temp' #D:\\Temp'
files = [
	'VIBY_1548 Title 1'
]

video_ext = 'mp4'

def transcribe(name):
	file = f'{base}\\{name}.{video_ext}'
	print(name, "=======================")
	model = subs_ai.create_model('guillaumekln/faster-whisper', {'model_size_or_path': 'large-v3', 'device': "cpu", "cpu_threads": 16}) #, "compute_type": "int16"
	subs = subs_ai.transcribe(file, model)
	subs.save(f'{base}\\{name}.jp.srt')

for n in files:
	transcribe(n)