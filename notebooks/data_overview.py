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

import pandas as pd
import git
from typing import Dict
from AugmentedNet.common import DATASPLITS, ANNOTATIONSCOREDUPLES


def resolve_dir(d):
    """Resolves '~' to HOME directory and turns ``d`` into an absolute path."""
    if d is None:
        return None
    d = str(d)
    if "~" in d:
        return os.path.expanduser(d)
    return os.path.abspath(d)

REPO_PATH = resolve_dir("..")
DATASET = "events"
augmentednet_repo = git.Repo(REPO_PATH)
augmentednet_version = "v1.0.0" 
print(REPO_PATH)

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
submodule_repos: Dict[str, git.Repo] = {
    sm.name: sm.module()
    for sm in augmentednet_repo.submodules
}
submodule_versions = {
    name: sm_repo.git.describe(tags=True, always=True)
    for name, sm_repo in submodule_repos.items()
}
submodule_versions

# %%
EXCLUDED_EXTENSIONS = (".md", ".csv", ".tsv", ".py", ".sh", ".pdf", ".jl")
EXCLUDED_NAME_COMPONENTS = ("feedback", "template", "requirements")

def get_commit_where_file_last_changed(repo: git.Repo, paths=str):
    try:
        return next(repo.iter_commits(paths=paths))
    except StopIteration as e:
        raise StopIteration(f"{repo!r} does not have any commits for {paths}") from e

data = []
rawdata_path = os.path.join(REPO_PATH, "rawdata")

for data_dir in os.listdir(rawdata_path):
    data_dir_path = os.path.join(rawdata_path, data_dir)
    if data_dir in submodule_versions:
        current_repo_name = data_dir
        git_path_base = data_dir_path
    else:
        current_repo_name = "AugmentedNet"
        git_path_base = REPO_PATH
    current_repo = submodule_repos.get(data_dir, augmentednet_repo)
    current_repo_version = submodule_versions.get(data_dir, augmentednet_version)
    for path, subdirs, files in os.walk(data_dir_path):
        rel_path = os.path.relpath(path, REPO_PATH)
        if rel_path == os.path.join("rawdata", "When-in-Rome"):
            subdirs[:] = ["Corpus"]
            continue
        for file in files:
            fname, fext = os.path.splitext(file)
            if not fext or fext in EXCLUDED_EXTENSIONS:
                continue
            fname_lower = fname.lower()
            if any(comp in fname_lower for comp in EXCLUDED_NAME_COMPONENTS):
                continue
            filepath = os.path.join(rel_path, file)
            folder_name = os.path.basename(rel_path)
            # git_filepath = os.path.relpath(os.path.join(path, file), git_path_base)
            # file_last_changed_commit = get_commit_where_file_last_changed(current_repo, paths=git_filepath)
            # file_last_changed_commit_sha = file_last_changed_commit.hexsha
            # file_last_changed_commit_version = current_repo.git.describe(file_last_changed_commit_sha, tags=True, always=True)
            info_dict = dict( 
                dataset = data_dir,
                repository = current_repo_name,
                repo_version = current_repo_version,
                directory = rel_path,
                folder = folder_name,
                filepath=filepath,
                fname = fname,
                extension=fext[1:],
                #last_modified=file_last_changed_commit_version
            )
            if filepath in v100_ids:
                nickname, split = v100_ids[filepath]
                info_dict["v1.0.0_id"] = nickname
                info_dict["v1.0.0_split"] = split
            data.append(info_dict)
            
df = pd.DataFrame.from_records(data)      
df.to_csv("../augnet_rawdata_overview.tsv", sep="\t", index=False)
df.head()

# %%
len(v100_ids)

# %%
path_df = df.set_index("filepath")
for filepath, (nickname, split) in v100_ids.items():
    path_df.loc[filepath]
    # info_dict["v1.0.0_id"] = nickname
    # info_dict["v1.0.0_split"] = split

# %%
summary = pd.read_csv(
    os.path.join(REPO_PATH, DATASET, "dataset_summary.tsv"),
    sep="\t",
    index_col=0
)
summary

# %%
