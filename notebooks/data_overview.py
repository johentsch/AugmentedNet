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
# DO NOT TRY TO RUN THIS ON WINDOWS

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
def add_safely(dictionary, key, pair):
    if key in dictionary:
        left, right = pair
        existing_left, existing_right = dictionary[key]
        if isinstance(existing_left, str):
            new_pair = ((existing_left, left), (existing_right, right))
        else:
            new_pair = (existing_left + (left,), existing_right + (right,))
        dictionary[key] = new_pair
    else:
        dictionary[key] = pair

def assemble_ids_and_splits(datasplits, annotationscoreduples):
    path2name_and_split = {}
    
    n = 0
    for split, files in datasplits.items():
        for nickname in files:
            annotations_path, score_path = annotationscoreduples[nickname]
            file_info = (nickname, split)
            add_safely(path2name_and_split, annotations_path, file_info)
            n += 1
            add_safely(path2name_and_split, score_path, file_info)
            n += 1
    return path2name_and_split, n

path2name_and_split, n_files = assemble_ids_and_splits(DATASPLITS, ANNOTATIONSCOREDUPLES)
print(f"{n_files} uses of {len(path2name_and_split)} files overall (some scores are used multiple times).")
#assert len(path2name_and_split) == i * 2, f"dict length {len(path2name_and_split)} != {i * 2} ({i} * 2)"

# %%
SUBMODULE_REPOS: Dict[str, git.Repo] = {
    sm.name: sm.module()
    for sm in augmentednet_repo.submodules
}
SUBMODULE_VERSIONS = {
    name: sm_repo.git.describe(tags=True, always=True)
    for name, sm_repo in SUBMODULE_REPOS.items()
}
SUBMODULE_VERSIONS

# %%
REPO_URLS = {
 'AugmentedNet': 'https://github.com/napulen/AugmentedNet',
 'TAVERN': 'https://github.com/jcdevaney/TAVERN',
 'ABC': 'https://github.com/DCMLab/ABC',
 'haydn_op20_harm': 'https://github.com/napulen/haydn_op20_harm',
 'When-in-Rome': 'https://github.com/MarkGotham/When-in-Rome',
 'music21_corpus': 'https://github.com/cuthbertLab/music21',
 'functional-harmony-micchi': 'https://github.com/napulen/functional-harmony-micchi'
}

# %%
EXCLUDED_EXTENSIONS = (".h5", ".jl", "krn~", ".md", ".pdf", ".py", ".sh", ".swp")
# (".cfg", ".css", ".csv", ".h5", ".html", ".in", ".ipynb", ".jl", ".js", ".md", ".pdf", ".png", ".py", ".rst", ".sh", ".tsv", ".yml")
EXCLUDED_NAME_COMPONENTS = ("feedback", "license", "slices", "template", "requirements")
PRINT_SYMBOLS = dict(validation="/", training="|", test="\\")
PATH_FILTERS = {
    # for rel_paths matching a key, go only through the subdirectories in the corresponding list
    os.path.join("rawdata", "When-in-Rome"): ["Corpus"],
    os.path.join("rawdata", "music21_corpus"): ["music21"],
    os.path.join("rawdata", "music21_corpus", "music21"): ["corpus"],
    os.path.join("rawdata", "music21_corpus", "music21", "corpus"): ["bach", "monteverdi"],
}
SUBCORPUS_POSITION = {
 'AugmentedNet': 2, # rawdata/corrections/ABC
 'TAVERN': 0, # TAVERN/Beethoven
 'ABC': None,
 'haydn_op20_harm': None,
 'When-in-Rome': 1, # When-in-Rome/Corpus/Early_Choral
 'music21_corpus': 2, # music21/corpus/bach
 'functional-harmony-micchi': 1 # data/19th_Century_Songs
}

def get_commit_where_file_last_changed(repo: git.Repo, paths=str):
    try:
        return next(repo.iter_commits(paths=paths))
    except StopIteration as e:
        raise StopIteration(f"{repo!r} does not have any commits for {paths}") from e

    
def create_data_overview(
        rawdata_path, 
        path2name_and_split,
        augnet_version
):
    data = []
    for data_dir in os.listdir(rawdata_path):
        if data_dir == "TAVERN": continue # the original files are not actually used and there are many, many, many
        print(f"\n{data_dir}")
        data_dir_path = os.path.join(rawdata_path, data_dir)
        if data_dir in SUBMODULE_VERSIONS:
            current_repo_name = data_dir
            git_path_base = data_dir_path
        else:
            current_repo_name = "AugmentedNet"
            git_path_base = REPO_PATH
        current_repo = SUBMODULE_REPOS.get(data_dir, augmentednet_repo)
        current_repo_version = SUBMODULE_VERSIONS.get(data_dir, augmentednet_version)
        current_repo_url = REPO_URLS.get(current_repo_name).strip("/")
        subcorpus_position = SUBCORPUS_POSITION.get(current_repo_name)
        for path, subdirs, files in os.walk(data_dir_path):
            rel_path = os.path.relpath(path, REPO_PATH)
            if rel_path in PATH_FILTERS:
                subdirs[:] = PATH_FILTERS[rel_path]
                continue
            for file in files:
                fname, fext = os.path.splitext(file)
                if not fext or fext in EXCLUDED_EXTENSIONS:
                    print(".", end="")
                    continue
                fname_lower = fname.lower()
                if any(comp in fname_lower for comp in EXCLUDED_NAME_COMPONENTS):
                    print(".", end="")
                    continue
                filepath = os.path.join(rel_path, file)
                folder_name = os.path.basename(rel_path)
                git_filepath = os.path.relpath(os.path.join(path, file), git_path_base)
                subcorpus = None
                if subcorpus_position is not None:
                    split_git_path = git_filepath.split(os.sep)
                    try:
                        subcorpus = split_git_path[subcorpus_position]
                    except Exception:
                        pass
                file_last_changed_commit = get_commit_where_file_last_changed(current_repo, paths=git_filepath)
                file_last_changed_commit_sha = file_last_changed_commit.hexsha
                file_last_changed_commit_version = current_repo.git.describe(file_last_changed_commit_sha, tags=True, always=True)
                file_change_commit_url = f"{current_repo_url}/blob/{file_last_changed_commit_version}/{git_filepath}"
                aug_ver = augnet_version.replace(".", "")
                info_dict = dict( 
                    dataset = data_dir,
                    subcorpus = subcorpus,
                    file=file,
                    fname = fname,
                    extension=fext[1:],
                )
                if filepath in path2name_and_split:
                    nickname, split = path2name_and_split[filepath]
                    which_set = split if isinstance(split, str) else split[0]
                    print_symbol = PRINT_SYMBOLS.get(which_set)
                else:
                    print_symbol = ":"
                    nickname, split = None, None
                info_dict.update({
                    f"id_{aug_ver}": nickname,
                    f"split_{aug_ver}": split,
                    f"last_modified_{aug_ver}": file_last_changed_commit_version,
                    f"file_change_commit_url_{aug_ver}": file_change_commit_url,
                    "repository": current_repo_name,
                    f"repo_version_{aug_ver}": current_repo_version,
                    "folder": folder_name,
                    "folderpath": rel_path,
                    "filepath": filepath
                })
                data.append(info_dict)
                print(print_symbol, end="")
    return pd.DataFrame.from_records(data).sort_values(["dataset", "subcorpus", "file", "filepath"])

rawdata_path = os.path.join(REPO_PATH, "rawdata")
df = create_data_overview(rawdata_path, path2name_and_split=path2name_and_split, augnet_version = augmentednet_version)
df.to_csv("../augnet_rawdata_v100.tsv", sep="\t", index=False)
df.head()

# %%
attributed_filepaths = df[f"split_{augmentednet_version.replace('.', '')}"].notna().sum()
assert attributed_filepaths == len(path2name_and_split), f"Not all of the {len(path2name_and_split)} used files have been attributed in the Dataframe, probably due to exclusion criteria."
