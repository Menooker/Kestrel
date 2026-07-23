"""
Article translation tool supporting Ollama and OpenAI-compatible APIs.

Usage examples:
  # Ollama (default)
  python article_translate.py --base source.txt --out translated.txt

  # With context file and hint
  python article_translate.py --base source.txt --out translated.txt ^
      --context context.txt --hint "This is a technical article about ML"

  # Resume from line 200
  python article_translate.py --base source.txt --out translated.txt --resume 200
"""

import time
import os
import argparse
import sys
import tqdm

# --- language helpers (copied from translate_ollama.py) ---

language_map = {
    "zh-cn": "chinese", "zh": "chinese",
    "jp": "japanese", "ja": "japanese",
    "en": "english",
}

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


# --- API clients ---

def get_client(api, model, key, base_url):
    """Return a chat() function for the chosen backend."""
    if api == "ollama":
        import ollama
        client = ollama.Client(host="http://127.0.0.1:11434")

        def chat_fn(messages):
            resp = client.chat(
                model=model,
                messages=messages,
                options={"think": False},
            )
            record_ollama_tokens(resp)
            return resp.message.content

        return chat_fn

    elif api == "openai":
        from openai import OpenAI
        client = OpenAI(base_url=base_url, api_key=key if key else "sk-placeholder")

        def chat_fn(messages):
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=8192,
            )
            record_openai_tokens(completion)
            return completion.choices[0].message.content

        return chat_fn

    else:
        raise ValueError(f"Unsupported API: {api}. Use 'ollama' or 'openai'.")


# --- translation core (same batching pattern as article_translate_google.py) ---

def translate(chat_fn, system_prompt, base, resume, outfile):
    with open(base, encoding="utf-8-sig") as f:
        lines = f.readlines()

    total = len(lines)
    bar = tqdm.tqdm(total=total, initial=resume, desc="Translate")
    idx = resume
    batch_size = 70
    history = []  # assistant responses for context window

    while idx < total:
        batch = lines[idx : idx + batch_size]
        user_msg = "".join(batch)

        messages = [
            {"role": "system", "content": system_prompt},
        ] + history + [
            {"role": "user", "content": user_msg},
        ]

        for attempt in range(6):
            try:
                cleaned = chat_fn(messages)
                outfile.write(cleaned)
                if cleaned and cleaned[-1] != "\n":
                    outfile.write("\n")
                outfile.flush()
                time.sleep(3)
                break
            except Exception as e:
                if attempt == 5:
                    raise
                print(f"\nError (attempt {attempt + 1}/6): {e}")
                wait = 10 * (attempt + 1)
                time.sleep(wait)
                print(f"  Retrying in {wait}s...")

        # Append assistant response to history (keep last 2 turns)
        history.append({"role": "assistant", "content": cleaned})
        history = history[-10:]

        bar.update(len(batch))
        idx += len(batch)


def build_system_prompt(out_lang, hint, context_file):
    """Build the system prompt following translate_ollama.py's style.

    Adapts translate_ollama.py's quality instructions to article translation,
    keeping the core instruction from article_translate_google.py.
    """
    target_language = language_map.get(out_lang, out_lang)

    parts = []
    # Core instruction (from article_translate_google.py)
    parts.append(
        "翻译这篇文章，我会一段段给你。只需要翻译最新给你的段落内容本身。"
        "如果你认为有一些需要补充的译注可以在文中通过括号（译注：……）插入，例如汉语语境当中较难翻译的地方。"
    )
    parts.append(f"请翻译到 {target_language}。")

    # Quality rules (adapted from translate_ollama.py)
    parts.append(
        "不要遗漏任何内容，严格对齐原文和译文。\n"
        "只输出翻译后的内容，不要输出额外说明。\n"
        "使用目标语言的母语表达和习惯用语。\n"
    )

    # Handling ambiguities / cultural context (from translate_ollama.py)
    parts.append(
        "如果出现谐音哏、双关语、文化背景等情况，请结合上下文在括号中进行补充说明。\n"
        "给你的文字可能存在语义不明确的情况，试着结合上下文推断正确含义并翻译。\n"
    )

    # Context file injection (same pattern as translate_ollama.py)
    custom_context = ""
    if context_file and os.path.isfile(context_file):
        with open(context_file, encoding="utf-8") as f:
            custom_context = f.read()
    if custom_context:
        parts.append(f"关于内容的额外信息：\n{custom_context}")

    # Hint injection (same pattern as article_translate_google.py)
    if hint:
        parts.append(f"上下文: [{hint}]")

    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(
        description="Article translation with Ollama / OpenAI API support."
    )
    parser.add_argument("--base", type=str, required=True, help="Input text file path")
    parser.add_argument("--out", type=str, required=True, help="Output translation file path")
    parser.add_argument(
        "--api", type=str, default="ollama", choices=["ollama", "openai"],
        help="API backend (default: ollama)",
    )
    parser.add_argument("--model", type=str, default="qwen2.5:14b", help="Model name")
    parser.add_argument("--key", type=str, default="", help="OpenAI API key (only for --api openai)")
    parser.add_argument(
        "--base-url", type=str, default="http://127.0.0.1:8080/v1/",
        help="OpenAI-compatible API base URL",
    )
    parser.add_argument("--in-lang", type=str, default="auto", help="Source language code (e.g. en, ja)")
    parser.add_argument("--out-lang", type=str, default="zh-cn", help="Target language code (e.g. zh-cn, en)")
    parser.add_argument("--hint", type=str, default="", help="Additional context / hint text")
    parser.add_argument("--context", type=str, default="", help="Path to a context file (injected as extra info)")
    parser.add_argument("--resume", type=int, default=0, help="Resume from line N")

    args = parser.parse_args()

    system_prompt = build_system_prompt(
        out_lang=args.out_lang,
        hint=args.hint,
        context_file=args.context,
    )
    print("System prompt preview:", system_prompt[:120], "..." if len(system_prompt) > 120 else "")

    chat_fn = get_client(args.api, args.model, args.key, args.base_url)

    with open(args.out, "a", encoding="utf-8-sig") as f:
        translate(chat_fn, system_prompt, args.base, args.resume, f)

    # Print token usage summary
    inp, cached, outp, total = _tokens
    print(f"Input={inp // 1000}k Cached={cached // 1000}k Output={outp // 1000}k")


if __name__ == "__main__":
    main()