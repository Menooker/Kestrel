"""
At the command line, only need to run once to install the package via pip:

$ pip install google-generativeai
"""


import ollama
import tqdm
import time
import argparse
import os

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

args = parser.parse_args()

base = args.base
files = args.list
resume = args.resume
in_lang = args.in_lang
out_lang = args.out_lang
hint = args.hint

language_map = {"zh-cn" : "chinese", "jp" : "japanese", "ja" : "japanese", "en" : "english"}
target_language = language_map.get(out_lang, out_lang)

client = ollama.Client(host = "http://kun:11434")


def remove_think(data: str) -> str:
  think_tag = "</think>"
  if think_tag in data:
    data = data[data.find(think_tag) + len(think_tag):]
  return data.strip()

system_instruction = f'''Translate a subtitle file.
You only need to translate the contents of the file.
Translate to {target_language}. Output "ENDENDEND" when you reach the end of the input
The input will be like

[[123::content1
[[124::content2

You only need to translate the contents. The content between "[[" and "::" should be ignored and unchanged in the output.

For the example above, output
[[123::内容1
[[124::内容2

Do not miss any of the content. Strictly align the id number with the translated output and original input. 
The format above is important. Do not output extract words other than the translated content. No "]]" necessary at the end.
Use native terms and expressions of the target language.
The input for you is a transcription of an audio. In some cases, it may be incorrect which messes up some words with similar pronunciation. You may need to infer the correct meaning by the context.

{hint}
Translate to {target_language}.
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
  prompt_parts = []
  # for start in range(resume, len(contents), batchsize):
  start = resume
  while start < len(contents):
    prompt_parts = prompt_parts[-3*2:]
    def make_promp():
      content_slice = contents[start:start+batchsize]
      promp = [f"[[{c[0]}::{c[1]}" for c in content_slice]
      # print("range:", content_slice[0][0], content_slice[-1][0])
      # print(promp)
      prompt_parts.append({"role":"user", "content":"\n".join(promp)})
      return content_slice, promp
    content_slice, promp = make_promp()
    done = False
    outtxt = ""
    while not done:
      time.sleep(1)
      response = client.chat('deepseek-r1:14b', messages=system_prompt + prompt_parts, options={'think': False})
      data = response.message.content
      prompt_parts.append({'role': 'assistant', 'content': data})
      data = remove_think(data)
      outtxt+=data
      if "ENDENDEND" in outtxt or outtxt.count("[[") >= len(content_slice):
        done = True
        outtxt = outtxt.replace("ENDENDEND", "")
      if done:
        break
      prompt_parts.append({"role":"user", "content":"continue"})
    bar.update(len(promp))
    # print(outtxt)
    outstr = outtxt.split("[[")
    for outline in outstr:
      outline = outline.strip()
      if not outline:
        continue
      spl = outline.split("::")
      idx = int(spl[0])
      con = spl[1]
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