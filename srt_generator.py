import json
import re
from datetime import timedelta

def format_srt_time(seconds):
    """Formats seconds into SRT time format: HH:MM:SS,mmm"""
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int(td.microseconds / 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

def clean_invisible(s):
    import re
    return re.sub(r'[\u200b\u200c\u200d\ufeff]', '', s)

def clean_spaces(s):
    return re.sub(r'\s+', '', s).strip()

def convert_json_to_srt(input_json, output_srt):
    with open(input_json, "r", encoding="utf-8") as f:
        subtitles = json.load(f)
    # 1. Pre-process: Handle "json" or "empty" and clean
    processed_subs = []
    for s in subtitles:
        s_lower = s.lower().strip()
        s_lower = clean_invisible(s_lower)
        s_lower = s_lower.replace('（', '(')
        s_lower = s_lower.replace('）', ')')
        s_lower = s_lower.replace('？', '?')

        if "```" in s_lower or "empty" in s_lower or not s or s == "空":
            processed_subs.append("")
        else:
            processed_subs.append(s_lower)

    # 2. Merge consecutive identical subtitles
    merged_subs = []
    if not processed_subs:
        print("No subtitles to process.")
        return

    current_text = processed_subs[0]
    start_time = 0

    for i in range(1, len(processed_subs)):
        if clean_spaces(processed_subs[i]) == clean_spaces(current_text):
            # Continue the current segment
            continue
        else:
            # Text changed, close current segment
            end_time = i
            # Only add if it is not an empty string
            if current_text != "":
                merged_subs.append({
                    "text": current_text,
                    "start": start_time,
                    "end": end_time
                })
            
            current_text = processed_subs[i]
            start_time = i

    # Add the last segment
    if current_text != "":
        merged_subs.append({
            "text": current_text,
            "start": start_time,
            "end": len(processed_subs)
        })

    # 3. Write to SRT
    with open(output_srt, "w", encoding="utf-8") as f:
        for idx, item in enumerate(merged_subs, 1):
            start_str = format_srt_time(item["start"])
            end_str = format_srt_time(item["end"])
            f.write(f"{idx}\n")
            f.write(f"{start_str} --> {end_str}\n")
            f.write(f"{item['text']}\n\n")

    print(f"Successfully converted {input_json} to {output_srt}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert JSON subtitle array to merged SRT format.")
    parser.add_argument("input", type=str, help="Input JSON file")
    parser.add_argument("output", type=str, help="Output SRT file")
    args = parser.parse_args()

    convert_json_to_srt(args.input, args.output)

