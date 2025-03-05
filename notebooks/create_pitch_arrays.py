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
#     display_name: ms3
#     language: python
#     name: ms3
# ---

# %%

import os
from typing import Optional

import ms3
import pandas as pd

from notebooks.utils import DivMaker, onset2beat

DLC_PATH = ms3.resolve_dir("~/distant_listening_corpus")


# %%

# %%
def filter_corpus(corpus):
    corpus.view.include("facets", "scores")#, "expanded")
    #corpus.disambiguate_facet("expanded")
    corpus.disambiguate_facet("scores")
    corpus.view.pieces_with_incomplete_facets = False
    
def get_ms3_corpus(corpus_path):
    corpus = ms3.Corpus(corpus_path)
    filter_corpus(corpus)
    return corpus

for subcorpus_dir in os.listdir(DLC_PATH):
    subcorpus_path = os.path.join(DLC_PATH, subcorpus_dir)
    if os.path.isfile(subcorpus_path): continue
    corpus = get_ms3_corpus(subcorpus_path)
    break
    
corpus

# %%
corpus = get_ms3_corpus("~/distant_listening_corpus/beethoven_piano_sonatas")

for _, piece in corpus.iter_pieces():
    break
    
for fileinfo, facets in piece.iter_extracted_facets(
        ("notes", "expanded"),
        force=True,
        unfold=True,
        interval_index=False
):
    break
    
notes, labels = facets["notes"], facets["expanded"]
notes["is_onset"] = (notes.tied.fillna(1) == 1)
labels["is_onset"] = True
display(notes.head(3))
labels.head(3)

# %%
merged = pd.merge(
    left = notes, 
    right = labels, 
    on = "quarterbeats_playthrough",
    how = "outer",
    suffixes = ("", "_label"),
    indicator=True
)
merged

# %%
KEEP_ORIGINAL_COLUMNS = ["mc", "mn", "mc_playthrough", "mn_playthrough", "quarterbeats_playthrough", "duration", "staff", "voice"]
RENAME_ORIGINAL_COLUMNS = dict(
        midi = "pitch",
    is_onset = "note_is_onset",
    is_onset_label = "harmony_is_onset",
    )
COLUMN_ORDER = ["onset_div", "duration_div", "pitch", "step", "alter", "ts_beats", "ts_beat_type", "staff", "voice"]

def make_pitch_array(notes: pd.DataFrame, beat_decimals: Optional[int] = 3) -> pd.DataFrame:
    
    div_maker = DivMaker(
        onsets = notes.quarterbeats_playthrough, 
        durations =notes.duration * 4 # normally duration_qb but due to a bug these are currently floats
    )
    onset_div, duration_div = div_maker[("onsets", "durations")]
    
    keep_original_columns = [col for col in KEEP_ORIGINAL_COLUMNS if col in notes.columns]
    original_columns = notes[keep_original_columns]
    
    rename_original_columns = {k: v for k, v in RENAME_ORIGINAL_COLUMNS.items() if k in notes.columns}
    renamed_columns = notes[list(rename_original_columns.keys())].rename(columns=rename_original_columns)
    
    new_dataframes = []  # will be added as-is
    new_columns = dict() # will be renamed based on the keys
    
    # specific pitch
    specific_pitch = notes.name.str.extract(r"^(?P<step>[A-G])(?P<accidentals>b*|#*)(?P<octave>\d)$")
    new_dataframes.append(specific_pitch[["step", "octave"]])
    new_columns["alter"] = specific_pitch.accidentals.str.count("#") - specific_pitch.accidentals.str.count("b")
    
    # time signatures & beats
    new_dataframes.append(
        notes.timesig.str.extract(r"^(?P<ts_beats>\d+)/(?P<ts_beat_type>\d+)$")
    )
    new_columns["beat"] = ms3.transform(merged, onset2beat, ["mn_onset", "timesig"], round_to=beat_decimals)

    result = pd.concat(
        [
            pd.DataFrame(
                dict(
                    onset_div=onset_div,
                    duration_div=duration_div
                )
            ),
            pd.concat(new_columns, axis=1),
            renamed_columns,
            original_columns
        ] + new_dataframes,
        axis=1
    )
    column_order = [col for col in COLUMN_ORDER if col in result.columns]
    column_order += [col for col in result.columns if col not in column_order]
    return result[column_order]

pitch_array = make_pitch_array(merged)
pitch_array.to_csv("beethoven1.tsv", sep="\t", index=False)
pitch_array

# %%
