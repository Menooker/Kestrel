import time
import os
import google.generativeai as genai
from tqdm import tqdm
import argparse

def get_model(key: str, hint: str):
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

    system_instruction = '''翻译这篇文章。只需要翻译内容本身。如果你认为有一些需要补充的译注可以在文中通过括号（）插入，例如汉语语境当中较难翻译的地方。'''
    if hint:
        system_instruction += f"\n上下文: [{hint}]"
    genai.configure(api_key=key,  transport="rest")
    model = genai.GenerativeModel(model_name="gemini-2.0-flash",
                                  system_instruction=system_instruction,
                                  safety_settings=safety_settings)
    return model

       

def translate(model, base: str, resume: int, outfile):
    with open(base, encoding="utf-8-sig") as f:
        lines = f.readlines()
    bar = tqdm(total=len(lines), initial=resume)
    idx = resume
    batch = 70
    prompt_parts = []
    while idx < len(lines):
        batch_data = lines[idx: idx+batch]
        prompt_parts.append(
            {"role":"user", "parts":"".join(batch_data)})
        for retries in range(6):
            try:
                response = model.generate_content(prompt_parts, request_options={"timeout": 600})
                cleaned = response.text
                outfile.write(cleaned)
                if cleaned[-1] != "\n":
                    outfile.write("\n")
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
        bar.update(len(batch_data))
        idx += len(batch_data)
        prompt_parts.append(response.candidates[0].content)
        prompt_parts = prompt_parts[-2:]
    # image.save(os.path.join(base, new_fn) + ".jpg", optimize=True)

def main(key: str, base: str, out: str, resume: int, hint: str):
    model = get_model(key, hint)
    with open(out, 'a', encoding="utf-8-sig") as f:
        translate(model, base, resume, f)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--key", type=str, required=True)
    parser.add_argument("--base", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--hint", type=str, default="")
    parser.add_argument('--resume', type=int, default=0)

    args = parser.parse_args()
    main(args.key, args.base, args.out, args.resume, args.hint)