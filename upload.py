import subprocess
import os
from typing import List, Tuple
import google.generativeai as genai
import tqdm
import time
import datetime
import argparse


def extract_mp3(ffmpeg_path: str, fullpath: str, tempdir: str, segment_sec: int):
    os.makedirs(tempdir, exist_ok=True)
    ret = subprocess.run([ffmpeg_path, "-i", fullpath, "-vn", "-c:a", "libmp3lame", "-q:a", "8", "-f",
                          "segment", "-segment_time", str(segment_sec), "-reset_timestamps", "1", os.path.join(tempdir, "output_%03d.mp3")])
    if ret.returncode != 0:
        raise RuntimeError("ffmpeg returns " + str(ret.returncode))


def upload(tempdir: str) -> List[str]:
    print("Uploading")
    files = []
    for file in os.listdir(tempdir):
        if file.startswith("output_") and file.endswith(".mp3"):
            files.append(os.path.join(tempdir, file))
    files = sorted(files)
    uri = []
    uripath = os.path.join(tempdir, "uri.txt")
    if os.path.exists(uripath):
        with open(uripath, 'r') as f:
            for line in f:
                line = line.strip()
                if len(line):
                    uri.append(line)
    bar = tqdm.tqdm(files[len(uri):])
    with open(uripath, "a") as f:
        for file in bar:
            bar.set_description(os.path.basename(file))
            sample_file = genai.upload_file(path=file,
                                            mime_type="audio/mp3")

            while sample_file.state.name == "PROCESSING":
                time.sleep(10)
                sample_file = genai.get_file(sample_file.name)

            if sample_file.state.name == "FAILED":
                raise ValueError(sample_file.state.name)
            f.write(sample_file.name+"\n")
            uri.append(sample_file.name)
    return uri


def extract_and_upload(fullpath: str, tempdir: str, ffmpeg_path: str, segment_sec: int, skip_extract: bool) -> List[str]:
    if not skip_extract:
        extract_mp3(ffmpeg_path, fullpath, tempdir, segment_sec)
    return upload(tempdir)

log_split_line = "!!!!!!!!!=================!!!!!!!!!!!!\n"
role_user = "ROLE=user,"
role_model = "ROLE=model,"
def record_translate_prompt(logf, file : genai.types.File):
    name = file.name
    logf.write(f"{log_split_line}{role_user}{name}\n")

def record_translate_result(logf, content : genai.types.GenerateContentResponse):
    logf.write(f"{log_split_line}ROLE=model,{str(content.text)}\n")

def recover_from_translate_result(path) -> Tuple[List[dict], List[str]]:
    prompt_parts = []
    responds = []
    with open(path, encoding="utf-8") as f:
        logs = f.read().split(log_split_line)
        for line in logs:
            line = line.strip()
            if len(line) == 0:
                continue
            if line.startswith(role_user):
                #file = genai.get_file(line[len(role_user):])
                #time.sleep(3)
                #prompt_parts.append({"role": "user", "parts": [file]})
                pass
            else:
                contents = line[len(role_model):]
                prompt_parts.append({"role": "model", "parts": [contents]})
                responds.append(contents)
    return prompt_parts, responds

def translate(tempdir: str, uris: List[str], segment: int):
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

    system_instruction = '''You make the subtitles for an audio. Transcribe the conversations in the audio.

The timestamp should be MM:SS which is minute and seconds in the current audio. It should be the start and end time for the conversation.
example of output subtitles format:
[[00:00~00:05
一行翻译内容

[[00:05~00:09
一行翻译内容

You MUST follow the following instructions:
Take down every sentences. Do not miss one.
Do not describe the audio. Do not describe the sound if it is NOT conversation. Instead, write down exactly the conversations.
The sentences spoken by different speakers should not be place in the same timestamp. Instead, put the dialogues into different timestamps.
Every sentence should NOT last more than 10 seconds.
You don't need to specify the person who say the line.

you need to do follows before output:
1. Remove the filler words, like "well", “嗯” ，“呃”.
For example, don't output "我们 嗯 吃了不少 嗯 东西". Instead, output "我们吃了不少东西".

2. Remove the unnecessary spaces between words in a sentence. Combine the words into sentences, instead of separated words and terms!
For example, don't output "我的 早饭 是 饭团". Instead, output ""我的早饭是饭团".

3. If a sentence is long, split it into multiple sentences with different timestamps using another block.
For example, don't output:
```
[[00:05~00:30
第一句话，第二句话。第三句话，这是非常长的一句话，真的非常非常非常非常非常非常非常非常非常非常非常长啊，没想到吧
```

Instead, output
```
[[00:05~00:10
第一句话，第二句话。第三句话

[[00:10~00:20
这是非常长的一句话

[[00:20~00:30
真的非常非常非常非常非常非常非常非常非常非常非常长啊，没想到吧
```

You should put the real timestamps for the splited blocks in the audio.
'''
    'If you reach the end of the audio, output an additional "ENDENDEND"'
    ''

    print("Translating")
    model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest",
                                  generation_config=generation_config,
                                  system_instruction=system_instruction,
                                  safety_settings=safety_settings)
    prompt_parts = []
    responses = []
    state_file_path = os.path.join(tempdir, "state.txt")
    if os.path.exists(state_file_path):
        print("Found state file of previous run. Resuming...")
        prompt_parts, responses = recover_from_translate_result(state_file_path)
        print("Continue from part", len(responses))
    bar = tqdm.tqdm(uris[len(responses):])
    with open(state_file_path, 'a', encoding="utf-8") as outf:
        for uri in bar:
            file = genai.get_file(uri)
            prompt_parts.append({"role": "user", "parts": [file]})
            for retries in range(4):
                try:
                    response = model.generate_content(prompt_parts, request_options={"timeout": 600})
                    time.sleep(15)
                    break
                except Exception as e:
                    if retries == 3:
                        raise e
                    print(f"Error!!!!!!!!!!!!!!{e}\\nsleeping")
                    time.sleep(60)
            # print(response.text)
            prompt_parts.pop()
            prompt_parts.append(response.candidates[0].content)
            # record_translate_prompt(outf, file)
            record_translate_result(outf, response)
            responses.append(response.text)
            # print(response.candidates[0].content)
            outf.flush()
            time.sleep(50)

    with open(os.path.join(tempdir, "raw.txt"), 'w', encoding="utf-8") as outf:
        for response in responses:
            outf.write(response)
            outf.write("\n=============================\n")

def convert(video_path: str, tempdir: str, segment: int):
    curtime = datetime.datetime.combine(
        datetime.datetime.today().date(), datetime.time(0, 0, 0))
    conversations = []
    results: List[Tuple[datetime.datetime, datetime.datetime, str]] = []

    def push_conversation():
        nonlocal conversations
        if len(conversations) == 0:
            return
        nonlocal st, ed
        conversations = " ".join(conversations).replace(
            "\t", " ").replace("\xa0", " ").split(" ")
        delta = ed - st
        totallen = len("".join(conversations))
        if totallen == 0:
            totallen = 1
        if totallen > 3100:
            '''# split the long sentence by the " "
            time_step = delta/totallen
            cur_start = st
            for didx, d in enumerate(conversations):
                if len(d.strip()) == 0:
                    continue
                mystart = cur_start
                myend = cur_start + time_step * len(d)
                cur_start = myend
                results.append(
                    f"{mystart.hour:02d}:{mystart.minute:02d}:{mystart.second:02d},{int(mystart.microsecond/1000.0):03d} --> {myend.hour:02d}:{myend.minute:02d}:{myend.second:02d},{int(myend.microsecond/1000.0):03d}\n{d}\n")
            '''
        else:
            mystart = st
            myend = ed
            d = " ".join(conversations)
            results.append([mystart, myend, d])
            # results.append(f"{mystart.hour:02d}:{mystart.minute:02d}:{mystart.second:02d},000 --> {myend.hour:02d}:{myend.minute:02d}:{myend.second:02d},000\n{d}\n")
        # d = "\n".join(conversations)
        # results.append(f"{st.hour:02d}:{st.minute:02d}:{st.second:02d},000 --> {ed.hour:02d}:{ed.minute:02d}:{ed.second:02d},000\n{d}\n")
        conversations = []
    with open(os.path.join(tempdir, "raw.txt"), 'r', encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            line = line.replace("ENDENDEND", "")
            if len(line) == 0:
                continue
            if "=========" in line:
                curtime += datetime.timedelta(seconds=segment)
                push_conversation()
            elif "[[" in line and "~" in line:
                try:
                    push_conversation()
                    start_time, end_time = line[2:].split("~")
                    st_spl = start_time.split(":")
                    ed_spl = end_time.split(":")
                    st = datetime.timedelta(minutes=int(
                        st_spl[0]), seconds=int(st_spl[1])) + curtime
                    ed = datetime.timedelta(minutes=int(
                        ed_spl[0]), seconds=int(ed_spl[1])) + curtime
                except Exception as e:
                    print("Error when parsing line: ", line)
                    raise e
            else:
                conversations.append(line)
        push_conversation()
    # adjust start timestamps
    idx = 0
    while idx < len(results):
        st, _, _ = results[idx]
        nidx = idx + 1
        # find equal range of st
        while nidx < len(results):
            st2, _, _ = results[nidx]
            if st2 != st:
                 break
            nidx += 1
        if nidx == idx + 1:
            idx = nidx
            continue
        nequal = nidx - idx
        delta = datetime.timedelta(seconds=1/nequal)
        for i in range(idx, nidx):
            results[i][0] = st + delta * (i - idx)
        idx = nidx
    # adjust end timestamps
    for idx, (st, ed, _) in enumerate(results):
        if idx > 0:
            _, last_ed, _ = results[idx-1]
            if st < last_ed:
                results[idx-1][1] = st
        if ed <= st:
            results[idx][1] = st + datetime.timedelta(seconds=0.5)

    outpathspl = video_path.split(".")
    outpathspl[-1] = "jp.srt"
    outpath = ".".join(outpathspl)
    with open(outpath, 'w', encoding="utf-8-sig") as outf:
        for idx, (mystart, myend, d) in enumerate(results):
            outf.write(f"{idx+1}\n{mystart.hour:02d}:{mystart.minute:02d}:{mystart.second:02d},{int(mystart.microsecond/1000.0):03d} --> {myend.hour:02d}:{myend.minute:02d}:{myend.second:02d},{int(myend.microsecond/1000.0):03d}\n{d}\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument(
        "--ffmpeg", type=str, default="D:\\ProgramFiles\\ffmpeg-master-latest-win64-gpl-shared\\bin\\ffmpeg.exe")
    parser.add_argument('--skip-extract', action="store_true", default=False)
    parser.add_argument('--key', type=str, required=True)
    parser.add_argument('--skip-translate', action="store_true", default=False)
    parser.add_argument(
        "--segment", type=int, default=180)
    args = parser.parse_args()
    genai.configure(api_key=args.key,  transport="rest")
    video_path = args.path
    fullpath = os.path.abspath(video_path)
    dirpath = os.path.dirname(fullpath)
    tempdir = fullpath + ".dir"

    if not args.skip_translate:
        uri = extract_and_upload(
            fullpath, tempdir, args.ffmpeg, args.segment, args.skip_extract)
        translate(tempdir, uri, args.segment)
    convert(video_path, tempdir, args.segment)
