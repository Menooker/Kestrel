import os
import re
import json
import base64
import argparse
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm

def encode_image(image_path):
    """Encodes an image to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def extract_subtitles(api_key, base_url, model, folder_path, output_file):
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    # Pattern: exactly 5 digits followed by .jpg
    pattern = re.compile(r"^(\d{5})\.jpg$")
    
    # Collect and sort files to maintain index order
    image_files = []
    if not os.path.exists(folder_path):
        print(f"Error: Folder {folder_path} does not exist.")
        return

    for f in os.listdir(folder_path):
        if pattern.match(f):
            image_files.append(f)
    
    image_files.sort()
    
    if not image_files:
        print(f"No matching files found in {folder_path}")
        return

    results = []
    
    # The prompt is crucial here to satisfy "only Japanese, only subtitles"
    prompt = (
        "Please identify the Japanese subtitles "
        "Return ONLY the Japanese text itself. Do not include any explanations, "
        "or introductory remarks. If no Japanese subtitles "
        "are present, return an empty string. Do not return JSON for empty string, if it is empty, just return empty string"
    )

    print(f"Found {len(image_files)} images. Starting processing...")
    with open(output_file+"prog.txt", "w", encoding="utf-8") as f:
        for filename in tqdm(image_files):
            img_path = os.path.join(folder_path, filename)
            #print(f"Processing: {filename}")
            
            try:
                base64_image = encode_image(img_path)
                
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    },
                                },
                            ],
                        }
                    ],
                    max_tokens=300,
                )
                
                content = response.choices[0].message.content.strip()
                #print("content:", content)
                # Basic cleaning
                results.append(content)
                f.write(json.dumps(content) + "\n")
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                results.append("")

    # Write to JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print(f"Done! Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Japanese subtitles from {5digit}.jpg images using OpenAI API.")
    parser.add_argument("folder", type=str, help="Path to the folder containing images")
    parser.add_argument("--api-key", type=str, required=True, help="OpenAI API Key")
    parser.add_argument("--endpoint", type=str, default="https://api.openai.com/v1", help="OpenAI API endpoint")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., gpt-4o)")
    parser.add_argument("--output", type=str, default="subtitles.json", help="Output JSON file path")

    args = parser.parse_args()

    extract_subtitles(
        api_key=args.api_key,
        base_url=args.endpoint,
        model=args.model,
        folder_path=args.folder,
        output_file=args.output
    )

