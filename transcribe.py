import subprocess
import os
from typing import List, Tuple
import google.generativeai as genai
import tqdm
import time
import datetime
import argparse
import re

def parse_timedelta(s):
    try:
        hours, minutes, seconds = map(int, s.split(':'))
        return datetime.timedelta(hours=hours, minutes=minutes, seconds=seconds)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid time format: {s}. Expected format hh:mm:ss")

def parse_timedelta_tuple_list(s):
    tuples = s.split(',')
    result = []
    for t in tuples:
        times = t.split('-')
        if len(times) != 2:
            raise argparse.ArgumentTypeError(f"Invalid tuple format: {t}. Expected format hh:mm:ss-hh:mm:ss")
        start, end = times
        result.append((parse_timedelta(start), parse_timedelta(end)))
    return result

def extract_mp3(ffmpeg_path: str, fullpath: str, tempdir: str, segment_sec: int):
    os.makedirs(tempdir, exist_ok=True)
    ret = subprocess.run([ffmpeg_path, "-i", fullpath, "-vn", "-c:a", "libmp3lame", "-q:a", "8", "-f",
                          "segment", "-segment_time", str(segment_sec), "-reset_timestamps", "1", os.path.join(tempdir, "output_%03d.mp3")])
    if ret.returncode != 0:
        raise RuntimeError("ffmpeg returns " + str(ret.returncode))

skip_filename = "@SKIP@"
def upload(tempdir: str, segment_sec: int, time_ranges: List[Tuple[datetime.timedelta, datetime.timedelta]]) -> List[str]:
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
    cur = datetime.timedelta(seconds=len(uri)*segment_sec)
    def in_ranges(s: datetime.timedelta, e: datetime.timedelta):
        if len(time_ranges) == 0:
            return True
        for rst, rend in time_ranges:
            if s <= rend and e >= rst:
                return True
        return False

    with open(uripath, "a") as f:
        for file in bar:
            if not in_ranges(cur, cur + datetime.timedelta(seconds=segment_sec)):
                f.write(skip_filename+"\n")
                uri.append(skip_filename)
                cur += datetime.timedelta(seconds=segment_sec)
                continue
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
            cur += datetime.timedelta(seconds=segment_sec)
    return uri


def cleanup_timestamp(s: str) -> str:
    return "\n".join(filter(lambda x: not x.startswith("[["), s.split("\n")))

def extract_and_upload(fullpath: str, tempdir: str, ffmpeg_path: str, segment_sec: int, skip_extract: bool, time_ranges: List[Tuple[datetime.timedelta, datetime.timedelta]]) -> List[str]:
    if not skip_extract:
        extract_mp3(ffmpeg_path, fullpath, tempdir, segment_sec)
    return upload(tempdir, segment_sec, time_ranges)

log_split_line = "!!!!!!!!!=================!!!!!!!!!!!!\n"
role_user = "ROLE=user,"
role_model = "ROLE=model,"
def record_transcribe_prompt(logf, file : genai.types.File):
    name = skip_filename if file is None else file.name
    logf.write(f"{log_split_line}{role_user}{name}\n")

def record_transcribe_result(logf, content : genai.types.GenerateContentResponse):
    cont = skip_filename if content is None else str(content.text)
    logf.write(f"{log_split_line}ROLE=model,{cont}\n")

def recover_from_transcribe_result(path) -> Tuple[List[dict], List[str]]:
    prompt_parts = []
    responds = []
    with open(path, encoding="utf-8") as f:
        logs = f.read().split(log_split_line)
        for line in logs:
            line = line.strip()
            if len(line) == 0:
                continue
            if line.startswith(role_user):
                filename = line[len(role_user):]
                if filename != skip_filename:
                    # file = genai.get_file(filename)
                    # time.sleep(3)
                    # prompt_parts.append({"role": "user", "parts": [file]})
                    pass
            else:
                contents = line[len(role_model):]
                if contents != skip_filename:
                    prompt_parts.append({"role": "model", "parts": [contents]})
                    responds.append(contents)
                else:
                    responds.append("")

    return prompt_parts, responds

def transcribe(tempdir: str, uris: List[str], segment: int):
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
Split the transcription into short sentences and mark their time stamps.

I may give you previous conversation context in textual form. And I will give you an audio segment. You only need to transcribe the audio.
The textual inputs are only for your reference. Do NOT directly use them as output. Transcribe the given audio only.

The timestamp should be MM:SS which is minute and seconds in the current audio. It should be the start and end time for each conversation.
example of output subtitles format:
[[00:00~00:05
some contents

[[00:05~00:09
some contents

You MUST follow the following instructions:
Take down every sentences. Do not miss one.
Do not describe the audio. Do not describe the sound if it is NOT conversation. Instead, write down exactly the conversations.
The sentences spoken by different speakers should not be place in the same timestamp. Instead, put the dialogues into different timestamps.
Every sentence should NOT last more than 7 seconds.
You don't need to specify the person who say the line.

you need to do follows before output:
1. Remove the filler words, like "well", “嗯” ，“呃”.
For example, don't output "我们 嗯 吃了不少 嗯 东西". Instead, output "我们吃了不少东西".

2. Remove the unnecessary spaces between words in a sentence. Combine the words into sentences, instead of separated words and terms!
For example, don't output "我的 早饭 是 饭团". Instead, output ""我的早饭是饭团".

3. If a sentence is long, break it into multiple sentences with length less than 7 seconds, with different timestamps using another blocks. 
You should put the real timestamps for the splited blocks in the audio. Every sentence should NOT last more than 7 seconds.
Every sentence should NOT last more than 7 seconds.

Example output:

[[00:05~00:10
第一句话，第二句话。第三句话

[[00:10~00:14
这是非常长的一句话

[[00:14~00:19
真的非常非常非常非常非常非常非常非常非常非常非常长啊，没想到吧


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
        prompt_parts, responses = recover_from_transcribe_result(state_file_path)
        print("Continue from part", len(responses))
    last_result = "" if len(responses) == 0 else cleanup_timestamp(responses[-1])
    bar = tqdm.tqdm(uris[len(responses):])
    with open(state_file_path, 'a', encoding="utf-8") as outf:
        for uri in bar:
            if uri == skip_filename:
                record_transcribe_prompt(outf, None)
                record_transcribe_result(outf, None)
                responses.append("")
                outf.flush()
                continue
            file = genai.get_file(uri)
            prompt_parts = {"role": "user", "parts": ["Previous context:\n" + last_result, file] if last_result else [file]}
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
            last_result = cleanup_timestamp(response.text)
            record_transcribe_prompt(outf, file)
            record_transcribe_result(outf, response)
            responses.append(response.text)
            # print(response.candidates[0].content)
            outf.flush()
            time.sleep(50)

    with open(os.path.join(tempdir, "raw.txt"), 'w', encoding="utf-8") as outf:
        for response in responses:
            spl = response.split("\n")
            for idx, line in enumerate(spl):
                if not line.startswith("[[") and "0" in line and ":" in line and "~" in line:
                    spl[idx] = "[[" + line
            response = '\n'.join(spl)
            outf.write(response)
            outf.write("\n=============================\n")

def convert(video_path: str, tempdir: str, segment: int, lang: str):
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
    outpathspl[-1] = lang + ".srt"
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
    parser.add_argument('--skip-transcribe', action="store_true", default=False)
    parser.add_argument(
        "--segment", type=int, default=180)
    parser.add_argument(
        "--lang", type=str, default="jp")
    parser.add_argument('--times', type=parse_timedelta_tuple_list, help='List of time intervals in the format "hh:mm:ss-hh:mm:ss,hh:mm:ss-hh:mm:ss"', default=[])
    args = parser.parse_args()
    genai.configure(api_key=args.key,  transport="rest")
    video_path = args.path
    fullpath = os.path.abspath(video_path)
    dirpath = os.path.dirname(fullpath)
    tempdir = fullpath + ".dir"

    if not args.skip_transcribe:
        uri = extract_and_upload(
            fullpath, tempdir, args.ffmpeg, args.segment, args.skip_extract, args.times)
        transcribe(tempdir, uri, args.segment)
    convert(video_path, tempdir, args.segment, args.lang)
