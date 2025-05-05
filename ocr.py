import time
from PIL import Image
import os
from pathlib import Path
import google.generativeai as genai
from tqdm import tqdm
import argparse

def get_model(key: str):
    # generation_config = {
    #     "temperature": 1,
    #     "top_p": 0.95,
    #     "top_k": 0,
    #     "max_output_tokens": 1048576,
    # }

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

    system_instruction = '''Output the contents of this article. Only the contents texts are needed.
Before outputing, remove the line breaks at the edge of the book, or in the middle of a sentence. Remove them.
Ignore the book page number.'''
    genai.configure(api_key=key,  transport="rest")
    model = genai.GenerativeModel(model_name="gemini-2.0-flash",
                                  system_instruction=system_instruction,
                                  safety_settings=safety_settings)
    return model

allowed_end = {'」', '。', '>', '，', '、'}
def clean_line_breaks(s: str) -> str:
    spl = s.split("\n")
    newlines = []
    for idx, line in enumerate(spl):
        strp = line.strip()
        if strp == "" or strp[-1] in allowed_end:
            newlines.append(line)
            newlines.append("\n")
            continue
        newlines.append(line)
    return "".join(newlines)
        

        

def do_ocr(model, rotate: bool, base: str, resume: int, outfile):
    files = [f for f in os.listdir(base) if f.lower().endswith(".jpg") and not f.startswith("compressed_") and os.path.isfile(os.path.join(base, f))]
    files = sorted(files)
    bar = tqdm(total=len(files), initial=resume)
    for idx, f in enumerate(files):
        if idx < resume:
            continue
        src = os.path.join(base, f)
        image = Image.open(src)
        x,y = image.size
        if x < y and rotate:
            image = image.rotate(90, Image.NEAREST, expand = 1)
        # new_fn = "compressed_" + Path(f).stem
        # print(src, "->", new_fn)
        image = image.resize((1024, 768), Image.Resampling.LANCZOS)
        prompt_parts=[image]
        for retries in range(6):
            try:
                response = model.generate_content(prompt_parts, request_options={"timeout": 600})
                cleaned = clean_line_breaks(response.text)
                outfile.write(cleaned)
                outfile.write("\n\n")
                outfile.flush()
                time.sleep(3)
                break
            except Exception as e:
                if retries == 5:
                    raise e
                print(f"Error!!!!!!!!!!!!!!{e}\\nsleeping")
                if " An internal error has occurred." in str(e):
                    time.sleep(20)
                else:
                    time.sleep(60)
        bar.update(1)
    # image.save(os.path.join(base, new_fn) + ".jpg", optimize=True)

def main(key: str, rotate: bool, base: str, out: str, resume: int):
    model = get_model(key)
    with open(out, 'a', encoding="utf-8-sig") as f:
        do_ocr(model, rotate, base, resume, f)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--key", type=str, required=True)
    parser.add_argument("--base", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--no-rotate", action="store_true", default=True)
    parser.add_argument('--resume', type=int, default=0)

    args = parser.parse_args()
    main(args.key, not args.no_rotate, args.base, args.out, args.resume)