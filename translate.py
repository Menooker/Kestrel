"""
At the command line, only need to run once to install the package via pip:

$ pip install google-generativeai
"""

from copy import Error
import google.generativeai as genai
import sys
import tqdm
import time
import argparse
import os

# old_init = requests.Session.request
# def newrequest()

# requests.Session.request

parser = argparse.ArgumentParser()
parser.add_argument("--key", type=str, required=True)
parser.add_argument("--base", type=str, required=True)
parser.add_argument('-l','--list', nargs='+', help='file names', default=[], required=True)
parser.add_argument('--resume', type=int, default=0)
parser.add_argument('--batchsize', type=int, default=200)
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

genai.configure(api_key=args.key,  transport="rest")

# Set up the model
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 0,
  "max_output_tokens": 1048576,
}

safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_NONE"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_NONE"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_NONE"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_NONE"
  },
]

system_instruction = f'''Translate a subtitle file.
You only need to translate the contents of the file.
Translate to {target_language}. Output "ENDENDEND" when you reach the end of the input
The input will be like

[[123::content1
[[124::content2

You only need to translate the contents. The content between "[[" and "::" should be ignored and unchanged in the output.
Translate to {target_language}. Translate to {target_language}. Translate to {target_language}.
Ignore the unnecessary spaces in the output sentences.
Ignore the unnecessary spaces in the output sentences.
For example, don't output "[[124::我的 早饭 是 饭团". Instead, output "[[124::我的早饭是饭团".
{hint}
'''

model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest",
                              generation_config=generation_config,
                              system_instruction=system_instruction,
                              safety_settings=safety_settings)

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
      prompt_parts.append({"role":"user", "parts":"\n".join(promp)})
      return content_slice, promp
    content_slice, promp = make_promp()
    done = False
    outtxt = ""
    while not done:
      for retries in range(6):
        try:
          response = model.generate_content(prompt_parts, stream=False)
          time.sleep(15)
          break
        except Exception as e:
          if retries == 5:
            raise e
          print(f"Error!!!!!!!!!!!!!!{e}\\nsleeping")
          is_internal_err = "An internal error has occurred." in str(e)
          time.sleep(20 if is_internal_err else 60)
          if 'Remote end closed connection without response' in str(e):
            batchsize //= 2
            if batchsize == 0:
              batchsize = 1
            prompt_parts.pop()
            content_slice, promp = make_promp()
            print("Retry with BS=", batchsize)
      for part in response:
        outtxt+=part.text
        if "ENDENDEND" in outtxt or outtxt.count("[[") >= len(content_slice):
          done = True
          outtxt = outtxt.replace("ENDENDEND", "")
      prompt_parts.append(response.candidates[0].content)
      if done:
        break
      prompt_parts.append({"role":"user", "parts":"continue"})
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