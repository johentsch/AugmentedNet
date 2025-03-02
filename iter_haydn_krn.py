"""
1. Call print_target_paths() to generate the TARGET_PATH_ORDER
2. Call print_urls() and open all printed URLs (Verovio Humdrum Viewer) in the correct order.
3. In that same order, download each of the files as musicXML (File -> Save as MusicXML).
   They will be called data.musicxml, data(1).musicxml, etc...
4. Use move_musicxml() to copy the downloaded files to their correct locations and rename them.
"""

import os
import re
import shutil
from pprint import pprint

from AugmentedNet.common import ANNOTATIONSCOREDUPLES, DATASPLITS

TARGET_PATH_ORDER = [
 'training/haydnop20-no1-1.musicxml',
 'validation/haydnop20-no1-2.musicxml',
 'training/haydnop20-no1-3.musicxml',
 'training/haydnop20-no1-4.musicxml',
 'training/haydnop20-no2-1.musicxml',
 'test/haydnop20-no2-2.musicxml',
 'training/haydnop20-no2-3.musicxml',
 'validation/haydnop20-no2-4.musicxml',
 'training/haydnop20-no3-1.musicxml',
 'training/haydnop20-no3-2.musicxml',
 'validation/haydnop20-no3-3.musicxml',
 'test/haydnop20-no3-4.musicxml',
 'training/haydnop20-no4-1.musicxml',
 'training/haydnop20-no4-2.musicxml',
 'training/haydnop20-no4-3.musicxml',
 'training/haydnop20-no4-4.musicxml',
 'training/haydnop20-no5-1.musicxml',
 'validation/haydnop20-no5-2.musicxml',
 'test/haydnop20-no5-3.musicxml',
 'training/haydnop20-no5-4.musicxml',
 'training/haydnop20-no6-1.musicxml',
 'training/haydnop20-no6-2.musicxml',
 'training/haydnop20-no6-3.musicxml',
 'test/haydnop20-no6-4.musicxml'
]

def print_target_paths():
    target_paths = {
        nickname: split
        for split, files in DATASPLITS.items()
        for nickname in files
        if nickname.startswith("haydn")
    }
    pprint([
        os.path.join(target_paths[nickname], f"{nickname}.musicxml")
        for nickname
        in sorted(target_paths.keys())
    ])


def print_urls(base_url):
    for no in sorted(os.listdir("rawdata/haydn_op20_harm/op20")):
        for i, mvt in enumerate(("i", "ii", "iii", "iv"), 1):
            src_file = f"op20n{no}-0{i}"
            url = f"{base_url}/{no}/{mvt}/{src_file}.krn"
            print(url)

def move_musicxml(
        download_dir,
        target_dir,
        move=False
):
    regex = r"data(?:\((\d+)\))?\.musicxml"
    for file in os.listdir(download_dir):
        if not (match := re.match(regex, file)):
            continue
        src_path = os.path.join(download_dir, file)
        i = int(match.group(1)) if match.group(1) else 0
        target_path = os.path.join(target_dir, TARGET_PATH_ORDER[i])
        target_path = os.path.abspath(target_path)
        print(src_path, "=>", target_path)
        if move:
            os.rename(src_path, target_path)
        else:
            shutil.copy(src_path, target_path)



if __name__ == "__main__":
    base_url = "https://verovio.humdrum.org/?file=github:napulen/haydn_op20_harm/op20"
    print_urls(base_url)

    download_dir = os.path.expanduser("~/Downloads")
    target_dir = "events"
    move_musicxml(download_dir, target_dir)