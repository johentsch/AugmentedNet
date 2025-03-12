# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import os
os.chdir("..")

import pandas as pd
import git

from AugmentedNet.common import DATASPLITS, ANNOTATIONSCOREDUPLES


def resolve_dir(d):
    """Resolves '~' to HOME directory and turns ``d`` into an absolute path."""
    if d is None:
        return None
    d = str(d)
    if "~" in d:
        return os.path.expanduser(d)
    return os.path.abspath(d)

REPO = resolve_dir(".")
DATASET = resolve_dir("events")
SUBMODULES = resolve_dir("rawdata")

# %%
v100_ids = {}

i = 0
for split, files in DATASPLITS.items():
    for nickname in files:
        annotations_path, score_path = ANNOTATIONSCOREDUPLES[nickname]
        file_info = (nickname, split)
        # if annotations_path in v100_ids:
        #     print(f"{nickname} | anno: {annotations_path} was already in for {v100_ids[annotations_path]}")
        v100_ids[annotations_path] = file_info
        if score_path in v100_ids:
            existing_nn, existing_split = v100_ids[score_path]
            file_info = ((existing_nn, nickname), (existing_split, split))
            #print(f"{nickname} | score: {score_path} was already in for {v100_ids[score_path]}")
        v100_ids[score_path] = file_info
        i += 1
    
#assert len(v100_ids) == i * 2, f"dict length {len(v100_ids)} != {i * 2} ({i} * 2)"

# %%
repo = git.Repo(REPO)
augmentednet_version = "v1.0.0" 
submodule_commits = {
    sm.name: sm.hexsha
    for sm in repo.submodules
}
submodule_commits

# %%
data = []
for data_dir in os.listdir(SUBMODULES):
    if data_dir in submodule_commits:
        repo_name = data_dir
        version = submodule_commits[data_dir]
    else:
        repo_name = "AugmentedNet"
        version = augmentednet_version
    version = submodule_commits.get(data_dir, augmentednet_version)
    for path, subdirs, files in os.walk(os.path.join(SUBMODULES, data_dir)):
        rel_path = os.path.relpath(path, REPO)
        if rel_path == os.path.join("rawdata", "When-in-Rome"):
            subdirs[:] = ["Corpus"]
            continue
        for file in files:
            fname, fext = os.path.splitext(file)
            if not fext or fext in (".md", ".csv", ".tsv", ".py", ".sh", ".pdf"):
                continue
            fname_lower = fname.lower()
            if "feedback" in fname_lower or "template" in fname_lower:
                continue
            filepath = os.path.join(rel_path, file)
            info_dict = dict(path=filepath, extension=fext[1:])
            if filepath in v100_ids:
                nickname, split = v100_ids[filepath]
                info_dict["v1.0.0_id"] = nickname
                info_dict["v1.0.0_split"] = split
            data.append(info_dict)
            
df = pd.DataFrame.from_records(data)      
df.to_csv("augnet_rawdata_overview.tsv", sep="\t", index=False)
df.head()

# %%
len(v100_ids)

# %%
path_df = df.set_index("path")
for filepath, (nickname, split) in v100_ids.items():
    path_df.loc[filepath]
    # info_dict["v1.0.0_id"] = nickname
    # info_dict["v1.0.0_split"] = split

# %%
summary = pd.read_csv(
    os.path.join(DATASET, "dataset_summary.tsv"),
    sep="\t",
    index_col=0
)
summary

# %%
