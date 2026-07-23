"""
OCR tool supporting Ollama and OpenAI-compatible APIs.

Usage examples:
  python ocr.py --base /path/to/images --out output.txt
  python ocr.py --base /path/to/images --out output.txt --api openai --base-url https://api.openai.com/v1 --key sk-xxx
  python ocr.py --base /path/to/images --out output.txt --resume 5
  python ocr.py --base /path/to/images --out output.txt --no-rotate
"""

import time
import argparse
import os
import io
from PIL import Image
import tqdm

SYSTEM_INSTRUCTION = (
    "You are an OCR engine. Extract all readable text from the image.\n"
    "Rules:\n"
    "  1. Output linear prose only - do NOT preserve line breaks from the original image, except for rule 3 below. Original layout of the image is irrelevant.\n"
    "  2. Do NOT include page numbers, headers, footers, watermarks, or any unrelated metadata. Footnotes should be included if present.\n"
    "  3. Only insert a newline when there is a semantic paragraph break or a new section/chapter.\n"
    "  4. Join broken words and sentences that were split across image lines.\n"
    "  5. Keep the original language of the text; do not translate.\n"
    "  6. If the image contains no readable text, output nothing.\n"
)

# Mutable accumulator: [input_tokens, cached_tokens, output_tokens, total_tokens]
_tokens = [0, 0, 0, 0]


def record_ollama_tokens(resp):
    """Extract token counts from an Ollama chat response."""
    _tokens[0] += resp.prompt_tokens
    _tokens[1] += resp.prompt_token_details.cached_tokens
    _tokens[2] += resp.completion_tokens
    _tokens[3] += resp.total_tokens


def record_openai_tokens(resp):
    """Extract token counts from an OpenAI chat completion response."""
    usage = resp.usage
    _tokens[0] += usage.prompt_tokens
    _tokens[1] += usage.prompt_tokens_details.cached_tokens
    _tokens[2] += usage.completion_tokens
    _tokens[3] += usage.total_tokens


def get_client(api, model, key, base_url):
    """Return a client and a chat() function for the chosen backend."""
    if api == "ollama":
        import ollama
        client = ollama.Client(host="http://127.0.0.1:11434")

        def chat(messages, image=None):
            kwargs = {"model": model, "messages": messages, "options": {"think": False}}
            if image is not None:
                kwargs["images"] = [image]
            resp = client.chat(**kwargs)
            record_ollama_tokens(resp)
            return resp.message.content

        return client, chat

    elif api == "openai":
        from openai import OpenAI
        client = OpenAI(base_url=base_url, api_key=key if key else "sk-placeholder")

        def chat(messages, image=None):
            parts = []
            import base64
            for m in messages:
                if m["role"] == "user":
                    content = m["content"]
                    if isinstance(content, str):
                        content = [{"type": "text", "text": content}]
                    if image is not None:
                        b64 = base64.b64encode(image).decode("utf-8")
                        content.insert(0, {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"}
                        })
                    parts.append({"role": m["role"], "content": content})
                else:
                    parts.append(m)
            completion = client.chat.completions.create(
                model=model,
                messages=parts,
                max_tokens=8192,
            )
            record_openai_tokens(completion)
            return completion.choices[0].message.content

        return client, chat

    else:
        raise ValueError(f"Unsupported API: {api}. Use 'ollama' or 'openai'.")


def prepare_image(src, rotate):
    """Load image, optionally rotate, resize, and return as JPEG bytes."""
    image = Image.open(src)
    x, y = image.size
    if x < y and rotate:
        image = image.rotate(90, Image.Resampling.BICUBIC, expand=True)
    image = image.resize((1024, 768), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    return buf.read()


def do_ocr(chat_fn, rotate, base, resume, outfile):
    """Process JPG/PNG files through the vision model sequentially."""
    files = sorted(
        f for f in os.listdir(base)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
        and not f.startswith("compressed_")
        and os.path.isfile(os.path.join(base, f))
    )
    total = len(files)
    if total == 0:
        print(f"No image files found in {base}")
        return

    bar = tqdm.tqdm(total=total, initial=resume, desc="OCR")
    user_prompt = "Extract all readable text from this image."

    for idx, f in enumerate(files):
        if idx < resume:
            bar.update(1)
            continue

        src = os.path.join(base, f)
        image_bytes = prepare_image(src, rotate)

        for attempt in range(6):
            try:
                resp = chat_fn(
                    [
                        {"role": "system", "content": SYSTEM_INSTRUCTION},
                        {"role": "user", "content": user_prompt},
                    ],
                    image=image_bytes,
                )
                outfile.write(resp)
                outfile.write("\n\n")
                outfile.flush()
                time.sleep(2)
                break
            except Exception as e:
                if attempt == 5:
                    raise
                print(f"\n[{f}] Error (attempt {attempt + 1}/6): {e}")
                wait = 10 * (attempt + 1)
                time.sleep(wait)
                print(f"  Retrying in {wait}s...")
        bar.update(1)


def main():
    parser = argparse.ArgumentParser(
        description="OCR for image files, supporting Ollama and OpenAI APIs."
    )
    parser.add_argument("--base", type=str, required=True, help="Directory containing image files")
    parser.add_argument("--out", type=str, required=True, help="Output text file path")
    parser.add_argument(
        "--api", type=str, default="ollama", choices=["ollama", "openai"],
        help="API backend (default: ollama)",
    )
    parser.add_argument("--model", type=str, default="qwen2.5vl:7b", help="Model name")
    parser.add_argument("--key", type=str, default="", help="OpenAI API key (only for --api openai)")
    parser.add_argument(
        "--base-url", type=str, default="http://127.0.0.1:8080/v1/",
        help="OpenAI-compatible API base URL",
    )
    parser.add_argument("--no-rotate", action="store_true", default=False, help="Disable auto-rotation")
    parser.add_argument("--resume", type=int, default=0, help="Skip first N files")

    args = parser.parse_args()
    _, chat_fn = get_client(args.api, args.model, args.key, args.base_url)

    with open(args.out, "a", encoding="utf-8") as f:
        do_ocr(chat_fn, rotate=not args.no_rotate, base=args.base, resume=args.resume, outfile=f)

    # Print token usage summary
    inp, cached, outp, total = _tokens
    print(f"Input={inp // 1000}k Cached={cached // 1000}k Output={outp // 1000}k")


if __name__ == "__main__":
    main()