"""
Knowledge pre-fetch tool to run before article_translate.py.

For each chunk of the article the model reads the text, studies what the user
is unlikely to already know (see the user profile passed via command line),
verifies candidate knowledge points by calling the search engine and opening
web pages through Chrome MCP, and finally emits a "result" call with an array
of KnowledgeObject {name, description, url}.

Usage examples:
  # Require a Chrome DevTools port (see chrome_tools/chrome_browser.py):
  #   chrome --remote-debugging-port=19222 (separate terminal, once)

  # Fetch hidden knowledge before translating, written as plain text
  python fetch_knowledge.py --base source.txt ^
      --profile "中文读者，金融从业者，熟悉美股基本概念" ^
      --base-url http://127.0.0.1:8080/v1/ --model qwen2.5:14b

  # Explicit output path, 4 function calls per chunk, language 中文
  python fetch_knowledge.py --base source.txt --out knowledge.txt ^
      --profile-file profile.txt --out-lang zh-cn --max-calls 4

  # Resume from line 200; loads knowledge.txt as already-fetched knowledge
  # and appends new knowledge points to it
  python fetch_knowledge.py --base source.txt --out knowledge.txt --resume 200

The out file is plain text, one ``name：description`` per line (URLs dropped);
``--resume`` parses it back into the model's "already fetched" section.
The LLM speaks JSON: each turn it returns either one function call
(search / open_urls) or the final result call. Parse errors and schema
violations are fixed with retries (mirroring article_translate.py).
"""

import argparse
import json
import os
import sys
import time
from typing import Annotated, Literal, Union

import tqdm
from pydantic import BaseModel, Field, TypeAdapter, ValidationError, model_validator

# --- language helpers (copied from article_translate.py) ---

language_map = {
    "zh-cn": "chinese", "zh": "chinese",
    "jp": "japanese", "ja": "japanese",
    "en": "english",
}

# Mutable accumulator: [input_tokens, cached_tokens, output_tokens, total_tokens]
_tokens = [0, 0, 0, 0]


def record_openai_tokens(resp):
    """Extract token counts from an OpenAI chat completion response."""
    usage = resp.usage
    _tokens[0] += usage.prompt_tokens
    _tokens[1] += usage.prompt_tokens_details.cached_tokens
    _tokens[2] += usage.completion_tokens
    _tokens[3] += usage.total_tokens


# --- API client ---

def get_client(model, key, base_url, preserve_thinking=False, max_thinking_tokens=5120):
    """Return a chat() function talking to an OpenAI-compatible backend (llama.cpp).

    chat_fn(messages) returns a (content, reasoning) tuple; reasoning is the
    model's thinking trace (llama.cpp: message.reasoning_content), or "" if
    the model produced none.
    """
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


# --- pydantic schemas (mapped to Python classes + validators) ---

def _is_http_url(url: str) -> bool:
    return url.startswith("http://") or url.startswith("https://")


def _visible_len(text: str) -> int:
    """Count visible (non-whitespace) characters, mirroring NightHeron models.py."""
    return len("".join(text.split()))


class KnowledgeObject(BaseModel):
    """One fetched knowledge point."""
    name: str = Field(description="知识点的名称（人名、地名、机构、事件等）")
    description: str = Field(description="用目标语言编写的简介，不超过50字")
    url: str = Field(description="该知识点的参考来源URL（必须是你实际打开并核实过页面内容的真实链接）")

    @model_validator(mode="after")
    def _semantic_check(self):
        self.name = self.name.strip()
        self.description = " ".join(self.description.split())
        self.url = self.url.strip()
        errs = []
        if not self.name:
            errs.append("name 不能为空")
        if not self.description:
            errs.append("description 不能为空")
        elif _visible_len(self.description) > 100:
            errs.append(f"description 长度 {_visible_len(self.description)} 超过上限，请压缩简介")
        if not self.url or not _is_http_url(self.url):
            errs.append("url 必须是有效的 http/https 链接")
        if errs:
            raise ValueError("；".join(errs))
        return self


class SearchCall(BaseModel):
    """Call the configured search engine."""
    tool: Literal["search"] = "search"
    query: str = Field(description="搜索关键词（读者不了解的人名、地名、机构或事件的名称）")

    @model_validator(mode="after")
    def _semantic_check(self):
        self.query = self.query.strip()
        if not self.query:
            raise ValueError("query 不能为空")
        return self


class OpenUrlsCall(BaseModel):
    """Open concrete web pages to verify facts."""
    tool: Literal["open_urls"] = "open_urls"
    urls: list[str] = Field(description="需要打开并核实的网页URL列表（1-5个）")

    @model_validator(mode="after")
    def _semantic_check(self):
        cleaned: list[str] = []
        seen: set[str] = set()
        for url in self.urls:
            url = (url or "").strip()
            if not url or url in seen:
                continue
            seen.add(url)
            cleaned.append(url)
        self.urls = cleaned
        errs = []
        if not cleaned:
            errs.append("urls 不能为空")
        elif len(cleaned) > 5:
            errs.append(f"urls 一次最多5个，当前 {len(cleaned)} 个")
        bad = [url for url in cleaned if not _is_http_url(url)]
        if bad:
            errs.append(f"存在非法URL：{bad}")
        if errs:
            raise ValueError("；".join(errs))
        return self


class ResultCall(BaseModel):
    """Final result of the current article chunk."""
    tool: Literal["result"] = "result"
    knowledge_points: list[KnowledgeObject] = Field(
        default_factory=list,
        description="本段文章新核实到的知识点列表（可以为空）",
    )

    @model_validator(mode="after")
    def _semantic_check(self):
        seen: set[str] = set()
        dedup: list[KnowledgeObject] = []
        for item in self.knowledge_points:
            key = item.name.casefold()
            if key in seen:
                continue
            seen.add(key)
            dedup.append(item)
        self.knowledge_points = dedup
        return self


Step = Annotated[Union[SearchCall, OpenUrlsCall, ResultCall], Field(discriminator="tool")]
StepAdapter = TypeAdapter(Step)


def parse_step(content: str) -> BaseModel:
    """Parse one LLM turn into a validated Step (SearchCall/OpenUrlsCall/ResultCall)."""
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
    if not isinstance(obj, dict):
        raise ValueError("输出必须是JSON对象，例如 {\"tool\": \"search\", \"query\": \"...\"}")
    return StepAdapter.validate_python(obj)


class FetchError(Exception):
    """Raised when an LLM step cannot be recovered after retries."""


# --- retry helper (mirrors article_translate.py's compress()) ---

def _parse_error_feedback(e: Exception) -> str:
    if isinstance(e, ValidationError):
        errors = "; ".join(str(item.get("msg")) for item in e.errors())
        return f"JSON 结构或语义不符合要求：{errors}"
    if isinstance(e, json.JSONDecodeError):
        return f"JSON 解析失败：{e.msg}（位置 {e.pos}）。只输出合法的 JSON 对象，不要包含其它文字或代码块。"
    return f"无法解析输出：{e}"


def run_step(chat_fn, messages, max_retries=3):
    """Call the model once and parse its JSON step, retrying on invalid output.

    Returns the parsed Step after appending the assistant turn (including any
    retry feedback) to ``messages``.
    """
    base = messages
    feedback = []
    for attempt in range(1, max_retries + 1):
        try:
            content, reasoning = chat_fn(base + feedback)
        except Exception as e:
            reason = f"LLM 调用失败：{type(e).__name__}：{e}"
        else:
            try:
                step = parse_step(content)
            except (json.JSONDecodeError, ValueError, ValidationError) as e:
                reason = _parse_error_feedback(e)
            except Exception as e:
                reason = f"无法解析输出：{e}"
            else:
                assistant_msg = {"role": "assistant", "content": content}
                if reasoning:
                    assistant_msg["reasoning_content"] = reasoning
                messages.extend(feedback)
                messages.append(assistant_msg)
                return step
        print(f"[step] Attempt {attempt} failed: {reason}")
        feedback.append({
            "role": "user",
            "content": (
                f"[重试提示] 上一次调用出错：{reason}\n"
                "请修正后重新输出一条 JSON 调用（search / open_urls / result）。"
            ),
        })
        time.sleep(2)
    raise FetchError(f"step 输出在 {max_retries} 次重试后仍然无法解析")


# --- system prompt builder ---

MAIN_PROMPT_TEMPLATE = """你是"文章知识点预取助手"。在你之后，另一个翻译引擎会把同一篇文章翻译给读者阅读。你的任务，是在翻译之前，针对用户逐段给出的文章内容，找出其中"读者大概率不知道、但和文章主题密切相关"的重要知识点，例如重要的地名、人物、机构、事件、专有名词、概念等，并通过搜索引擎和网页抓取进行核实，最后以 result 调用输出。

选择知识点的原则：
1. 只输出读者大概率不知道、且重要、与文章主题直接相关的知识点。请参考下方的《用户画像》判断读者已知什么。
2. 不要放入读者很可能已经知道的信息、不要放入常识性内容、不要放入与主题关系不大的次要知识。
3. 优先选择需要查证才能获得准确信息的知识点（具体人名、地名、机构、事件细节、行业术语等）。
4. 只输出经你在网上核实过的信息，不要凭记忆编造；对拿不准的，宁可不输出。
5. 如果本段确实没有可靠的新知识点，result 调用中的知识列表也可以为空，但依然要输出 result 调用。

工作流程（对每一段文章输入）：
1. 阅读系统给你的文章段落。
2. 本段你最多可以进行 {max_calls} 次function call。可以选择：
   - 调用 search 进行网络搜索；
   - 调用 open_urls 打开搜索到的具体网页进行核实。
   每次只输出一条 function call JSON；等待系统返回 [工具结果] 后再决定下一步。
3. 信息核实完毕后，输出一次 result 调用，包含本段确认的新知识点。
4. 若 {max_calls} 次function call 已用完但仍未核实完，也必须输出 result 调用汇总当前已有信息。

另外，下方《已获取知识》中列出的知识点，是之前段落已经获取过的，不要重复获取或再次输出。

输出格式：
- function call 与 result 都必须是单独、合法的 JSON 对象，不要输出任何其它文字，不要使用 markdown 代码块。
- 具体结构见下方《function call schema》与《result schema》。
"""


def build_knowledge_section(known):
    """Render accumulated knowledge (list of (name, description)) for the prompt."""
    lines = ["## 已获取知识（以下知识点之前已经获取并输出过，不要重复获取或再次输出；URL已省略）"]
    if not known:
        lines.append("（暂无，目前还没有获取过任何知识）")
    else:
        for name, description in known:
            lines.append(f"- {name}：{description}")
    return "\n".join(lines)


def build_system_prompt(out_lang, profile, max_calls, known):
    """Compose the per-model system prompt.

    Sections: 1. main prompt  2. user profile  3. function call schema
    4. result schema  5. previously fetched knowledge (URLs stripped).
    ``known`` is a list of (name, description) tuples.
    """
    parts = [MAIN_PROMPT_TEMPLATE.format(max_calls=max_calls)]

    if profile:
        parts.append(
            "## 用户画像\n"
            f"{profile}\n"
            "请依据以上画像判断读者已知和未知的信息：只输出读者大概率不知道的知识点。"
        )

    tool_schemas = {
        "search": SearchCall.model_json_schema(),
        "open_urls": OpenUrlsCall.model_json_schema(),
    }
    parts.append(
        "## function call schema（tool 取值决定调用类型）\n"
        + json.dumps(tool_schemas, ensure_ascii=False, indent=2)
    )

    parts.append(
        "## result schema\n"
        + json.dumps(ResultCall.model_json_schema(), ensure_ascii=False, indent=2)
    )

    parts.append(build_knowledge_section(known))

    return "\n\n".join(parts)


def render_knowledge_line(name, description):
    """Render one knowledge point as a plain-text line (URL dropped)."""
    return f"{name}：{description}"


def load_prior_knowledge(path):
    """Parse a plain-text out file back into (name, description) pairs.

    Each line is rendered as ``name：description``; used to seed the "already
    fetched" section and dedup when resuming with ``--resume``.
    """
    prior = []
    if not path or not os.path.isfile(path):
        return prior
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            for sep in ("：", ":"):
                if sep in line:
                    name, description = line.split(sep, 1)
                    name, description = name.strip(), description.strip()
                    if name and description:
                        prior.append((name, description))
                    break
    return prior


# --- per-chunk tool-call loop ---

def process_chunk(chat_fn, browser, chunk_text, max_calls, max_retries, system_prompt, bar):
    """Run one article chunk through the model and return its KnowledgeObjects."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": chunk_text},
    ]
    tool_used = 0
    forced_rounds = 0
    while True:
        step = run_step(chat_fn, messages, max_retries)
        bar.set_postfix_str(f"In {_tokens[0]/1000:.1f}k,cached {_tokens[1]/1000:.1f}k,Out {_tokens[2]/1000:.1f}k")
        if isinstance(step, ResultCall):
            return step.knowledge_points

        if isinstance(step, SearchCall):
            if tool_used >= max_calls:
                forced_rounds += 1
                if forced_rounds >= 3:
                    raise FetchError(
                        f"function call 次数已用尽（{max_calls} 次），但模型始终不输出 result 调用"
                    )
                messages.append({
                    "role": "user",
                    "content": (
                        f"[系统提示] 你本段文章的 {max_calls} 次 function call 已经用完。"
                        "请立即输出 result 调用，汇总你已经核实到的全部知识点（不要再次调用任何工具）。"
                    ),
                })
                continue
            try:
                result = browser.search(step.query)
            except Exception as e:
                reason = f"{type(e).__name__}: {e}"
                print(f"[tool-error] search 失败：{reason}")
                result = f"[工具错误] 搜索失败：{reason}"
            messages.append({
                "role": "user",
                "content": f"[工具结果] 搜索结果（关键词：{step.query}）:\n{result}",
            })
        elif isinstance(step, OpenUrlsCall):
            if tool_used >= max_calls:
                forced_rounds += 1
                if forced_rounds >= 3:
                    raise FetchError(
                        f"function call 次数已用尽（{max_calls} 次），但模型始终不输出 result 调用"
                    )
                messages.append({
                    "role": "user",
                    "content": (
                        f"[系统提示] 你本段文章的 {max_calls} 次 function call 已经用完。"
                        "请立即输出 result 调用，汇总你已经核实到的全部知识点（不要再次调用任何工具）。"
                    ),
                })
                continue
            try:
                pages = browser.open_urls(step.urls)
            except Exception as e:
                reason = f"{type(e).__name__}: {e}"
                print(f"[tool-error] open_urls 失败：{reason}")
                pages = {url: f"[工具错误] 打开失败：{reason}" for url in step.urls}
            text = "\n\n".join(f"### {url}\n{body}" for url, body in pages.items())
            messages.append({
                "role": "user",
                "content": f"[工具结果] 打开的网页内容:\n{text}",
            })
        else:
            raise FetchError(f"未知的 step 类型：{type(step).__name__}")

        tool_used += 1
        time.sleep(0.5)


# --- chunked main loop ---

def fetch_knowledge(chat_fn, browser, out_lang, profile, base, batch_size, max_calls, max_retries,
                    prior, resume, out_path):
    """Fetch knowledge chunk by chunk.

    ``prior`` seeds the "already fetched" knowledge from the existing out file;
    processing starts at line ``resume``; new knowledge is appended to
    ``out_path`` as plain text (URL dropped).
    Returns the number of newly fetched knowledge points.
    """
    with open(base, encoding="utf-8-sig") as f:
        lines = f.readlines()
    known = list(prior)
    known_names = {name.casefold() for name, _ in known}
    system_prompt = build_system_prompt(out_lang, profile, max_calls, known)
    bar = tqdm.tqdm(total=len(lines), initial=resume, desc="FetchKnowledge")
    idx = resume
    new_count = 0
    with open(out_path, "a", encoding="utf-8") as outfile:
        while idx < len(lines):
            chunk = "".join(lines[idx : idx + batch_size])
            found = process_chunk(chat_fn, browser, chunk, max_calls, max_retries, system_prompt, bar)

            for item in found:
                if item.name.casefold() in known_names:
                    print(f"[dedupe] 跳过已获取知识点：{item.name}")
                    continue
                known_names.add(item.name.casefold())
                known.append((item.name, item.description))
                new_count += 1
                outfile.write(render_knowledge_line(item.name, item.description) + "\n")
            outfile.flush()
            system_prompt = build_system_prompt(out_lang, profile, max_calls, known)

            bar.update(batch_size)
            idx += batch_size
    bar.close()
    return new_count


def make_browser_tool(args):
    try:
        from chrome_tools.chrome_browser import BrowserTool
    except ImportError as e:
        raise SystemExit(
            "无法加载 Chrome MCP 工具，请先安装依赖：pip install mcp\n"
            "并确保 Node/npm 可用、Chrome 已启动远程调试端口（见 chrome_tools/chrome_browser.py）。"
        ) from e
    return BrowserTool(
        browser_url=args.browser_url,
        timeout_ms=args.timeout_ms,
        max_content_length=args.max_content_length,
        search_engine=args.search_engine,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Article knowledge pre-fetch with OpenAI-compatible API (llama.cpp) and Chrome MCP."
    )
    parser.add_argument("--base", type=str, required=True, help="Input text file path")
    parser.add_argument("--out", type=str, default="", help="Output plain-text file path (default: <base>.knowledge.txt)")
    parser.add_argument("--resume", type=int, default=0, help="Resume from line N; loads --out file as already-fetched knowledge and appends")
    parser.add_argument("--model", type=str, default="qwen2.5:14b", help="Model name")
    parser.add_argument("--key", type=str, default="", help="OpenAI API key")
    parser.add_argument(
        "--base-url", type=str, default="http://127.0.0.1:8080/v1/",
        help="OpenAI-compatible API base URL (llama.cpp)",
    )
    parser.add_argument("--out-lang", type=str, default="zh-cn", help="Result language code (e.g. zh-cn, en)")
    parser.add_argument("--profile", type=str, default="", help="User profile (identity, known knowledge, etc.) put into the system prompt")
    parser.add_argument("--profile-file", type=str, default="", help="Path to a user profile text file (merged with --profile)")
    parser.add_argument("--batch", type=int, default=10, help="Batch size in lines per chunk")
    parser.add_argument(
        "--max-calls", dest="max_calls", type=int, default=4,
        help="Max function calls (search / open_urls) allowed per chunk (default: 4)",
    )
    parser.add_argument(
        "--max-retries", dest="max_retries", type=int, default=3,
        help="Retries per invalid JSON step output (default: 3)",
    )
    parser.add_argument(
        "--browser-url", type=str, default="http://127.0.0.1:19222",
        help="Chrome DevTools MCP browser URL",
    )
    parser.add_argument(
        "--search-engine", type=str, default="google", choices=["google", "bing"],
        help="Search engine used by the search tool (default: google)",
    )
    parser.add_argument(
        "--timeout-ms", type=int, default=60000,
        help="Per-page navigation timeout in milliseconds",
    )
    parser.add_argument(
        "--max-content-length", type=int, default=15000,
        help="Cap for cleaned snapshot text length per page (default: 15000)",
    )
    parser.add_argument(
        "--no-preserve-thinking", dest="preserve_thinking", action="store_false",
        help="Disable preserving the model's reasoning/thinking across turns (llama.cpp chat_template_kwargs; enabled by default)",
    )
    parser.set_defaults(preserve_thinking=True)
    parser.add_argument(
        "--max-thinking-tokens", type=int, default=5120,
        help="Thinking token budget passed to the sampler (extra_body thinking_budget_tokens / reasoning_budget_tokens)",
    )

    args = parser.parse_args()

    chat_fn = get_client(args.model, args.key, args.base_url, args.preserve_thinking, args.max_thinking_tokens)

    profile = ""
    if args.profile_file and os.path.isfile(args.profile_file):
        with open(args.profile_file, encoding="utf-8") as f:
            profile = f.read().strip()
        print(f"[profile] Loaded user profile from {args.profile_file}")
    if args.profile:
        profile = (profile + "\n" if profile else "") + args.profile
    if profile:
        print(f"[profile] User profile: {profile}")

    out_path = args.out or os.path.splitext(args.base)[0] + ".knowledge.txt"
    prior = load_prior_knowledge(out_path)
    if prior:
        print(f"[resume] Loaded {len(prior)} previously fetched knowledge point(s) from {out_path}")

    browser = None
    try:
        browser = make_browser_tool(args)
        new_count = fetch_knowledge(
            chat_fn,
            browser,
            out_lang=args.out_lang,
            profile=profile,
            base=args.base,
            batch_size=args.batch,
            max_calls=args.max_calls,
            max_retries=args.max_retries,
            prior=prior,
            resume=args.resume,
            out_path=out_path,
        )
    finally:
        if browser is not None:
            browser.close()

    total = len(prior) + new_count
    print(f"\nFetched {new_count} new knowledge point(s); {total} total in {out_path}")

    # Print token usage summary (same format as article_translate.py)
    inp, cached, outp, total_tokens = _tokens
    print(f"Input={inp // 1000}k Cached={cached // 1000}k Output={outp // 1000}k")


if __name__ == "__main__":
    main()