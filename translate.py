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
parser.add_argument('-l','--list', nargs='+', help='<Required> Set flag', required=True)
parser.add_argument('--resume', type=int, default=0)
parser.add_argument('--batchsize', type=int, default=200)

args = parser.parse_args()

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

system_instruction = '''Translate a subtitle file.
You only need to translate the contents of the file.
Translate to chinese. Output "ENDENDEND" when you reach the end of the input
The input will be like

[[123::content1
[[124::content2

You only need to translate the contents. The content between "[[" and "::" should be ignored and unchanged in the output.
Translate to chinese. Translate to chinese. Translate to chinese.
'''

model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest",
                              generation_config=generation_config,
                              system_instruction=system_instruction,
                              safety_settings=safety_settings)
base = args.base
files = args.list
resume = args.resume

print(base, files)

for filename in files:
  with open(os.path.join(base, f'{filename}.jp.srt'), encoding="utf-8") as f:
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

  outf = open(os.path.join(base, f'{filename}.zh-cn.srt'), 'w', encoding="utf-8")
  progress = 0
  translated = lines[:]
  bar = tqdm.tqdm(total=line_count)
  batchsize = args.batchsize
  bar.update(resume)
  prompt_parts = []
  for start in range(resume, len(contents), batchsize):
    content_slice = contents[start:start+batchsize]
    promp = [f"[[{c[0]}::{c[1]}" for c in content_slice]
    # print("range:", content_slice[0][0], content_slice[-1][0])
    # print(promp)
    prompt_parts.append({"role":"user", "parts":"\n".join(promp)})
    done = False
    outtxt = ""
    while not done:
      for retries in range(3):
        try:
          response = model.generate_content(prompt_parts, stream=False)
          time.sleep(15)
          break
        except Exception as e:
          print(f"Error!!!!!!!!!!!!!!{e}\\nsleeping")
          time.sleep(60)
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
    progress = end_idx+1
  outf.flush()
  outf.close()