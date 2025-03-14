import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import argparse
from pathlib import Path


def chapter_to_str(chapter):
    soup = BeautifulSoup(chapter.get_body_content(), 'html.parser')
    text = [para.get_text() for para in soup.find_all('p')]
    return '\n'.join(text)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True)

    args = parser.parse_args()

    book = epub.read_epub(args.file)
    items = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))


    p = Path(args.file)
    with open(str(p.parent/p.stem)+".jp.txt", 'w', encoding="utf-8-sig") as f:
        for ch in items:
            f.write(chapter_to_str(ch))
