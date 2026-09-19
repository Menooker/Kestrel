"""
Article translation tool supporting Ollama and OpenAI-compatible APIs.

Usage examples:
  # Ollama (default)
  python article_translate.py --base source.txt --out translated.txt

  # With context file and hint
  python article_translate.py --base source.txt --out translated.txt ^
      --context context.txt --hint "This is a technical article about ML"

  # Control batch size and history window (with automatic note compression)
  python article_translate.py --base source.txt --out translated.txt --batch 40 --history 8

  # Resume from line 200
  python article_translate.py --base source.txt --out translated.txt --resume 200

  # Preserve reasoning/thinking across turns (on by default for --api openai;
  # disable with --no-preserve-thinking)
  python article_translate.py --base source.txt --out translated.txt --api openai
"""

import argparse
import json
import os
import sys
import time

import tqdm

from pydantic import BaseModel, Field, ValidationError, model_validator

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

def get_client(api, model, key, base_url, preserve_thinking=False, max_thinking_tokens=5120):
    """Return a chat() function for the chosen backend.

    chat_fn(messages) returns a (content, reasoning) tuple; reasoning is the
    model's thinking trace (llama.cpp: message.reasoning_content), or "" if
    the model produced none.
    """
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
            return resp.message.content, ""

        return chat_fn

    elif api == "openai":
        from openai import OpenAI
        client = OpenAI(base_url=base_url, api_key=key if key else "sk-placeholder")

        def chat_fn(messages):
            extra_body = {}
            if preserve_thinking:
                extra_body["chat_template_kwargs"] = {"preserve_thinking": True}
            extra_body["thinking_budget_tokens"] = max_thinking_tokens
            extra_body["reasoning_budget_tokens"] = max_thinking_tokens
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                extra_body=extra_body or None,
            )
            record_openai_tokens(completion)
            msg = completion.choices[0].message
            return msg.content, getattr(msg, "reasoning_content", None) or ""

        return chat_fn

    else:
        raise ValueError(f"Unsupported API: {api}. Use 'ollama' or 'openai'.")


# --- history compression (translation notes) ---

class CompressionError(Exception):
    """Raised when the compression task exceeds its retry limit."""


def build_semantic_errors(names, notes, max_notes_len=100):
    """Return a list of human-readable error strings; empty means OK."""
    errs = []
    if not notes or not notes.strip():
        errs.append("notes 为空")
    elif len(notes) > max_notes_len + 50:
        errs.append(f"notes 长度 {len(notes)} 超过 {max_notes_len} 字，请缩短notes")
    for name, translated in names.items():
        if not name or not name.strip():
            errs.append("names 中存在空的 key")
        elif not translated or not translated.strip():
            errs.append(f"names 中 {name} 的值为空，缺少目标语言译文")
    return errs


class CompressionNote(BaseModel):
    """Compression result: name correspondence table + brief summary."""
    names: dict[str, str] = Field(
        default_factory=dict,
        description="原文语言到目标语言的名字对照表（人物、地名、事物等）",
    )
    notes: str = Field(
        description="当前可见内容的简要梗概（目标语言，100字以内）",
    )

    @model_validator(mode="after")
    def _semantic_check(self):
        errs = build_semantic_errors(self.names, self.notes)
        if errs:
            raise ValueError("\n".join(errs))
        return self


def build_compression_prompt(out_lang, max_notes_len=100):
    """Build the compression task prompt (placed inside the compression message)."""
    target_language = language_map.get(out_lang, out_lang)
    return f"""你现在执行"翻译笔记"压缩任务，不要执行翻译任务。
仔细阅读你当前能看到的所有内容（包括 system 消息中已有的翻译笔记），生成一份翻译笔记。
忽略其它 system 消息中关于翻译任务的指令，只执行本压缩任务。
“压缩”任务的执行目的是为了翻译一致性，需要你总结出你当前可见的所有内容当中，出现的名称（人物、地名、事物等）在原文语言和目标语言之间的对应关系，并且生成一份简要的翻译笔记，帮助后续翻译。
需要你排除掉 system 消息中已有的翻译笔记内容，只记录之前未曾出现的新内容。

只输出如下格式的 JSON，不要输出任何其它文字，不要使用 markdown 代码块：
{{"names": {{"<原文语言的名字>": "<目标语言的名字>"}}, "notes": "<目标语言写的笔记>"}}

"names"：原文语言中的人物、地名、事物等值得在后续翻译中保持一致译名的名字。key 必须使用原文语言，value 使用目标语言（{target_language}）。
"notes"：当前可见内容的简要翻译笔记，包含内容梗概等有助于之后翻译的信息，使用目标语言（{target_language}），{max_notes_len} 字以内。

重要：system 消息中已有翻译笔记记录过的内容，不要在本任务的"names"/"notes"中重复出现；只记录之前未曾出现的新内容。

JSON schema:
{CompressionNote.model_json_schema()}
"""


def parse_compression_response(content):
    """Parse the LLM response into a CompressionNote, raising on bad output."""
    text = content.strip()
    if text.startswith("```"):
        idx = text.find("\n")
        if idx == -1:
            raise json.JSONDecodeError("fence without newline", text, 0)
        text = text[idx + 1:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    obj = json.loads(text)
    return CompressionNote.model_validate(obj)


def compress(chat_fn, conversation, out_lang, max_retries=3):
    """Condense the visible conversation into translation notes.

    Returns a CompressionNote. Retries up to max_retries times, feeding back
    concrete error messages via extra system messages.
    """
    base_messages = conversation + [{"role": "user", "content": build_compression_prompt(out_lang)}]
    error_messages = []
    for attempt in range(1, max_retries + 1):
        messages = base_messages + error_messages
        content, _ = chat_fn(messages)
        try:
            return parse_compression_response(content)
        except (json.JSONDecodeError, ValidationError) as e:
            if isinstance(e, json.JSONDecodeError):
                print(content)
                reason = f"JSON 解析失败：{e.msg}（位置 {e.pos}）。只输出合法的 JSON 对象，不要包含其它文字或代码块。"
            else:
                errors = "; ".join(str(item.get("msg")) for item in e.errors())
                reason = f"JSON 结构或语义不符合要求：{errors}"
        except Exception as e:
            reason = f"无法解析输出：{e}"
        print(f"[compress] Attempt {attempt} failed: {reason}")
        error_messages.append({
            "role": "system",
            "content": (
                f"上一次的笔记输出不符合要求：{reason}\n"
                "请修正后重新输出完整的 JSON，包含全部需要记录的 names 与 notes。"
            ),
        })
        time.sleep(2)
    raise CompressionError(f"压缩任务在 {max_retries} 次重试后仍然失败")


def format_note_block(note):
    """Render a CompressionNote as readable text for the system prompt."""
    lines = ["\n--- 翻译笔记 ---"]
    if note.names:
        lines.append("名字对照表：")
        for name, translated in note.names.items():
            lines.append(f"- {name} => {translated}")
    lines.append(f"笔记：{note.notes}")
    return "\n".join(lines)


# --- translation core (same batching pattern as article_translate_google.py) ---

def translate(chat_fn, out_lang, hint, custom_context, base, resume, outfile, batch_size, num_history, note_file):
    with open(base, encoding="utf-8-sig") as f:
        lines = f.readlines()
    note_string = ""
    if note_file and os.path.isfile(note_file):
        with open(note_file, "r", encoding="utf-8-sig") as nf:
            note_string = nf.read()
    total = len(lines)
    bar = tqdm.tqdm(total=total, initial=resume, desc="Translate")
    idx = resume
    system_prompt = build_system_prompt(out_lang, hint, custom_context, note_string)
    messages = [{"role": "system", "content": system_prompt}]
    input_count = 0

    while idx < total:
        batch = lines[idx : idx + batch_size]
        user_msg = "".join(batch)
        messages.append({"role": "user", "content": user_msg})

        cleaned, reasoning = chat_fn(messages)
        outfile.write(cleaned)
        if cleaned and cleaned[-1] != "\n":
            outfile.write("\n")
        outfile.flush()
        time.sleep(0.5)

        assistant_msg = {"role": "assistant", "content": cleaned}
        if reasoning:
            assistant_msg["reasoning_content"] = reasoning
        messages.append(assistant_msg)
        input_count += 1

        if input_count >= num_history:
            # set text of the bar
            bar.set_description_str("compressing")
            note = compress(chat_fn, messages, out_lang)
            bar.set_description_str("Translate")
            note_string += format_note_block(note)
            system_prompt = build_system_prompt(out_lang, hint, custom_context, note_string)
            messages = [{"role": "system", "content": system_prompt}]
            input_count = 0
            with open(note_file, "a", encoding="utf-8-sig") as nf:
                nf.write(format_note_block(note) + "\n")

        bar.update(len(batch))
        idx += len(batch)


def build_system_prompt(out_lang, hint, custom_context, note_string=""):
    """Build the system prompt following translate_ollama.py's style.

    Adapts translate_ollama.py's quality instructions to article translation,
    keeping the core instruction from article_translate_google.py. The
    accumulated note_string is appended verbatim when provided.
    """
    target_language = language_map.get(out_lang, out_lang)

    parts = []
    # Core instruction (from article_translate_google.py)
    system_instruction = f'''翻译这篇文章，我会一段段给你。只需要翻译最新给你的段落内容本身。每一段输入内容是前后关联的。
如果你认为有一些需要补充的译注可以在文中通过括号（译注：……）插入，例如汉语语境当中较难翻译的地方。
译注内不要放入常识性内容、不要放入你对原文的理解。
请翻译到 {target_language}。
不要遗漏任何内容，严格对齐原文和译文。
只输出翻译后的内容，不要输出额外说明。
使用目标语言的母语表达、语序和习惯用语。
如果出现谐音哏、双关语、文化背景等情况，请结合上下文在括号中进行补充说明。
给你的文字可能存在语义不明确的情况，试着结合上下文推断正确含义并翻译。如果你并不明确原文的意思，可以在译注当中额外说明。

除了文章内容，我可能会给你“压缩”命令，会以【你现在执行"翻译笔记"压缩任务】开头来出现，如果你收到这样的命令，你需要按照命令的要求进行格式化输出。否则执行翻译任务。你不需要对“压缩”命令进行翻译，只需要执行翻译笔记的压缩任务。
'''
    parts.append(system_instruction)
    # Custom context injection (same pattern as translate_ollama.py)
    if custom_context:
        parts.append(f"=======\n关于内容的额外信息：\n{custom_context}\n=======")

    # Hint injection (same pattern as article_translate_google.py)
    if hint:
        parts.append(f"上下文: [{hint}]")

    # Accumulated translation notes from previous compression rounds
    if note_string:
        parts.append("以下是之前翻译内容的笔记，你需要在翻译时候保持和笔记内翻译名称的一致性：" + note_string)

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
    parser.add_argument("--batch", type=int, default=70, help="Batch size in lines per request")
    parser.add_argument(
        "--num-history", dest="num_history", type=int, default=8,
        help="Max user inputs kept in history before triggering note compression",
    )
    parser.add_argument("--note-file", type=str, default="", help="The note file path to save translation notes (optional)")
    parser.add_argument(
        "--no-preserve-thinking", dest="preserve_thinking", action="store_false",
        help="Disable preserving the model's reasoning/thinking across turns (llama.cpp chat_template_kwargs; enabled by default for --api openai)",
    )
    parser.set_defaults(preserve_thinking=True)
    parser.add_argument(
        "--max-thinking-tokens", type=int, default=5120,
        help="Thinking token budget passed to the sampler (extra_body thinking_budget_tokens / reasoning_budget_tokens; --api openai)",
    )

    args = parser.parse_args()

    chat_fn = get_client(args.api, args.model, args.key, args.base_url, args.preserve_thinking, args.max_thinking_tokens)

    custom_context = ""
    if args.context and os.path.isfile(args.context):
        with open(args.context, encoding="utf-8") as f:
            custom_context = f.read()
        print(f"[context] Loaded {len(custom_context)} characters from {args.context}")
    if not args.note_file:
        args.note_file = args.out + ".notes.txt"
    with open(args.out, "a", encoding="utf-8-sig") as f:
        translate(
            chat_fn,
            out_lang=args.out_lang,
            hint=args.hint,
            custom_context=custom_context,
            base=args.base,
            resume=args.resume,
            outfile=f,
            batch_size=args.batch,
            num_history=args.num_history,
            note_file=args.note_file
        )

    # Print token usage summary
    inp, cached, outp, total = _tokens
    print(f"Input={inp // 1000}k Cached={cached // 1000}k Output={outp // 1000}k")


if __name__ == "__main__":
    main()