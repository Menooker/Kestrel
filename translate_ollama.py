"""
At the command line, only need to run once to install the package via pip:

$ pip install google-generativeai
"""


import ollama
import tqdm
import time
import argparse
import os
import json
from pydantic import BaseModel

# old_init = requests.Session.request
# def newrequest()

# requests.Session.request

parser = argparse.ArgumentParser()
parser.add_argument("--base", type=str, required=True)
parser.add_argument('-l','--list', nargs='+', help='file names', default=[], required=True)
parser.add_argument('--resume', type=int, default=0)
parser.add_argument('--batchsize', type=int, default=50)
parser.add_argument('--in-lang', type=str, default="jp")
parser.add_argument('--out-lang', type=str, default="zh-cn")
parser.add_argument('--hint', type=str, default="")
parser.add_argument('--model', type=str, default="gpt-oss:20b")

args = parser.parse_args()

base = args.base
files = args.list
resume = args.resume
in_lang = args.in_lang
out_lang = args.out_lang
hint = args.hint

language_map = {"zh-cn" : "chinese", "jp" : "japanese", "ja" : "japanese", "en" : "english"}
target_language = language_map.get(out_lang, out_lang)
source_language = language_map.get(in_lang, in_lang)

client = ollama.Client(host = "http://kun:11434")

class TranslatedMessage(BaseModel):
  id: int
  translated: str

class TranslatedResponse(BaseModel):
  data: list[TranslatedMessage]

schema = TranslatedResponse.model_json_schema()

def remove_think(data: str) -> str:
  think_tag = "</think>"
  if think_tag in data:
    data = data[data.find(think_tag) + len(think_tag):]
  return data.strip()

system_instruction = f'''Translate a subtitle. You only need to translate the contents.
Translate to {target_language}.
The input will be in JSON format like below:
[
  {{"id": 123, "content": "content1"}},
  {{"id": 124, "content": "content2"}}
]

You only need to translate the contents. Keep the id unchanged in output.

Do not miss any of the content. Strictly align the id number with the translated output and original input. 
Do not output extra words other than the translated content.
Use native terms and expressions of the target language.
The input for you is a transcription of an audio. In some cases, it may be incorrect which messes up some words with similar pronunciation. You may need to infer the correct meaning by the context.

如果出现谐音哏、双关语、文化背景等情况，请结合上下文在括号中进行补充说明.
用户给出的“content”是从连续的对话中截取的，相邻的content可能可以组成一句连续的话。翻译时考虑语序。允许重新排列临近的句子保证翻译语序自然。
给你的文字是从录音中听写下来的，可能会有听写错误，如果发现原文语义不正确，试着结合上下文找出原来意思并翻译，通过括号说明
{hint}
The user will provide the context of the conversions. It is only for your reference to understand the context. You only need to translate the latest input.

Translate from {source_language} to {target_language}.
'''

print(base, files)

for filename in files:
  with open(os.path.join(base, f'{filename}.{in_lang}.srt'), encoding="utf-8-sig") as f:
    lines = f.readlines()
  print("GEN")
  line_count = 0
  contents=[]
  for idx,l in enumerate(lines):
    if " --> " in l:
      line_count+=1
      continue
    l=l.strip()
    if not l.isdigit() and len(l):
      contents.append((idx, l))

  outf = open(os.path.join(base, f'{filename}.{out_lang}.srt'), 'w' if resume == 0 else 'a', encoding="utf-8-sig")
  if resume == 0:
    progress = 0
  else:
    progress = lines.index(str(resume+1) + "\n")
    if progress < 0:
      raise RuntimeError("Cannot find resume point")
  translated = lines[:]
  bar = tqdm.tqdm(total=line_count)
  batchsize = args.batchsize
  bar.update(resume)
  system_prompt = [{'role': 'system', 'content': system_instruction}]
  history = []
  # for start in range(resume, len(contents), batchsize):
  start = resume
  while start < len(contents):
    history = history[-3:]
    prompt_parts = []
    def make_promp():
      content_slice = contents[start:start+batchsize]
      request = [{"id": c[0], "content": c[1]} for c in content_slice]
      request_json = json.dumps(request, ensure_ascii=False)
      # promp = [f"[[{c[0]}::{c[1]}" for c in content_slice]
      # print("range:", content_slice[0][0], content_slice[-1][0])
      # print(promp)
      history_str = ("history for reference: " + '\n'.join(history)) if history else ""
      prompt_parts.append({"role":"user", "content": f"{history_str}\nTranslate this from {source_language} to {target_language}:\n{request_json}\nTranslate this to {target_language}。 翻译到中文"})
      return content_slice, request
    content_slice, promp = make_promp()
    resp = dict()
    while True:
      time.sleep(1)
      response = client.chat(args.model, messages=system_prompt + prompt_parts, options={'think': False}, format=schema)
      data = response.message.content
      # prompt_parts.append({'role': 'assistant', 'content': data})
      # data = remove_think(data)
      try:
        raw: TranslatedResponse = TranslatedResponse.model_validate_json(data)
        # if len(resp.data) != len(promp):
        for m in raw.data:
          resp[m.id] = m.translated
        missing = []
        for r in promp:
          if r["id"] not in resp:
            missing.append(r["id"])
        if missing:
          request = [{"id": c[0], "content": c[1]} for c in content_slice if c[0] in missing]
          request_json = json.dumps(request, ensure_ascii=False)
          prompt_parts[-1]={"role":"user", "content": request_json}
          print("Missing:", missing, "Retrying...")
          continue
        history.append("\n".join([resp[p['id']] for p in promp]))
        break
      except Exception as e:
        print("Error:", e)
        print("Response:", data)
        print("Retrying...")
        prompt_parts.append({"role":"user", "content":f"The previous output is not valid JSON format. error: {str(e)}"})
        continue
    bar.update(len(promp))
    # print(outtxt)
    for idx, con in resp.items():
      con = con.strip()
      if idx < content_slice[0][0] or idx > content_slice[-1][0]:
        #print("Bad line", outline)
        continue
      translated[idx] = con + "\n"
    end_idx = content_slice[-1][0]
    for idx in range(progress, end_idx+1):
      outf.write(translated[idx])
    outf.flush()
    progress = end_idx+1
    start += batchsize
  outf.flush()
  outf.close()