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
from enum import Enum
from typing import Optional, Tuple

import ms3
import pandas as pd
from IPython.display import display

from notebooks.utils import DivMaker, onset2beat

DLC_PATH = ms3.resolve_dir("~/distant_listening_corpus")


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
        ("measures", "notes", "expanded"),
        force=True,
        unfold=True,
        interval_index=False
):
    break

measures = facets["measures"]
display(measures.head(3))
notes = facets["notes"]
notes.head(3)


# %%
def make_section_start_column(
        measures: pd.DataFrame,
) -> pd.Series:
    """Returns a column of nullable "boolean" dtype."""
    section_start = (measures.repeats == "firstMeasure").fillna(False).rename("section_start").astype("boolean")
    section_start |= (measures.repeats == "start")
    section_start |= (measures.repeats.shift() == "end")
    section_start |= (measures.breaks.shift().str.contains("section"))
    section_start |= (measures.barline.shift() == "double")
    return section_start

def prepare_measures(
        measures:pd.DataFrame,
) -> pd.DataFrame:
    section_start_column = make_section_start_column(measures)
    measures = pd.concat([
        measures.rename(columns=dict(quarterbeats="quarterbeats_playthrough")), 
        section_start_column
    ], axis=1)
    measures.keysig = measures.keysig.astype("Int64") 
    return measures


# %%
MERGE_MEASURE_COLUMNS = ["keysig"]
MERGE_LABEL_COLUMNS = ["section_start"]

def prepare_notes_with_measure_information(
        notes: pd.DataFrame,
        measures: pd.DataFrame,
        label_notes: bool = False
) -> pd.DataFrame:
    """ Add key signature from measure table and, optionally, labels created from it.
    
    Args:
        label_notes: 
            If set to True, the measures table is used to create binary labels that are True for MCs
            where a new section begins.
    """
    prepared_measures = prepare_measures(measures)
    potential_columns = ["quarterbeats_playthrough"] + MERGE_MEASURE_COLUMNS
    if label_notes:
        potential_columns += MERGE_LABEL_COLUMNS
    merge_measure_columns = [
        col for col 
        in potential_columns
        if col in prepared_measures.columns]
    
    merged = pd.merge(
        left = notes, 
        right = prepared_measures[merge_measure_columns], 
        on = "quarterbeats_playthrough",
        how = "outer",
    )
    merged.keysig = merged.keysig.ffill()
    if label_notes:
        merged.section_start = merged.section_start.fillna(False)
    return merged


# %%
KEEP_ORIGINAL_COLUMNS = ["mc", "mn", "mc_playthrough", "mn_playthrough", "quarterbeats_playthrough", "duration",
                         "staff", "voice", "is_note_onset"]
KEEP_ORIGINAL_LABEL_COLUMNS = ["section_start"]
RENAME_ORIGINAL_COLUMNS = dict(
    midi="pitch",
    keysig="ks_fifths"
)
COLUMN_ORDER = ["onset_div", "duration_div", "pitch", "step", "alter", "ts_beats", "ts_beat_type", "staff", "voice"]


def make_pitch_array(
        notes: pd.DataFrame,
        measures: Optional[pd.DataFrame] = None,
        beat_decimals: Optional[int] = 3,
        label_notes: bool = False
) -> pd.DataFrame:
    """Transformation of a notes table to a pitch array that can be transformed into a graph. 
    
    Args:
        beat_decimals: 
            Integer controlling the number of decimal places in the column "beat". If you pass None,
            the column will contain :obj:`Fraction` objects.
        label_notes: 
            By default, this function includes only transformations that are part of the input representation.
            Set to True in order to include training labels from the measures table as well (for details,
            see prepare_notes_with_measure_information()).
    
    
    """
    if measures is not None:
        prepared_notes = prepare_notes_with_measure_information(notes, measures, label_notes=label_notes)
    else:
        prepared_notes = notes

    div_maker = DivMaker(
        onsets=prepared_notes.quarterbeats_playthrough,
        durations=prepared_notes.duration * 4  # normally duration_qb but due to a bug these are currently floats
    )
    onset_div, duration_div = div_maker[("onsets", "durations")]

    potential_columns = list(KEEP_ORIGINAL_COLUMNS)
    if label_notes:
        potential_columns += KEEP_ORIGINAL_LABEL_COLUMNS
    keep_original_columns = [col for col in potential_columns if col in prepared_notes.columns]
    original_columns = prepared_notes[keep_original_columns]

    rename_original_columns = {k: v for k, v in RENAME_ORIGINAL_COLUMNS.items() if k in prepared_notes.columns}
    renamed_columns = prepared_notes[list(rename_original_columns.keys())].rename(columns=rename_original_columns)

    new_dataframes = []  # will be added as-is
    new_columns = dict()  # will be renamed based on the keys

    new_columns["is_note_onset"] = (prepared_notes.tied.fillna(1) == 1)

    # specific pitch
    specific_pitch = prepared_notes.name.str.extract(r"^(?P<step>[A-G])(?P<accidentals>b*|#*)(?P<octave>\d)$")
    new_dataframes.append(specific_pitch[["step", "octave"]])
    new_columns["alter"] = specific_pitch.accidentals.str.count("#") - specific_pitch.accidentals.str.count("b")

    # time signatures & beats
    new_dataframes.append(
        prepared_notes.timesig.str.extract(r"^(?P<ts_beats>\d+)/(?P<ts_beat_type>\d+)$")
    )
    new_columns["beat"] = ms3.transform(prepared_notes, onset2beat, ["mn_onset", "timesig"], round_to=beat_decimals)
        

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


pitch_array = make_pitch_array(notes, measures, label_notes=False)
pitch_array.to_csv("beethoven1.tsv", sep="\t", index=False)
pitch_array


# %%
class Mode(str, Enum):
    minor = "minor"
    major = "major"
    
    

def get_globalkey(labels: pd.DataFrame) -> Tuple[int, Mode]:
    gk_column = labels.globalkey
    assert not gk_column.isna().any(), "Column 'globalkey' contains missing values"
    gkeys = list(gk_column.unique())
    assert len(gkeys) == 1, f"Expected a single global key but got {gkeys}"
    gkey = gkeys[0]
    return ms3.name2fifths(gkey), Mode.minor if gkey.islower() else Mode.major


def make_global_scale_degree_column(
        notes: pd.DataFrame,
        globalkey_tonic_fifths: int,
):
    return (notes.tpc - globalkey_tonic_fifths).rename("global_scale_degree")


def prepare_labels(labels: pd.DataFrame) -> pd.DataFrame:
    labels = labels.copy()
    labels["is_harmony_onset"] = True
    labels.is_harmony_onset = labels.is_harmony_onset.where(
        labels.chord.notna() & (labels.chord != labels.chord.shift(-1)),
        False, # set False where the label does not define a harmony or merely the same harmony as the preceding one
    ) 
    labels.index.rename("unfolded_harmony_index", inplace=True)
    labels.reset_index(drop=False, inplace=True)
    globalkey_tonic_fifths, globalkey_mode = get_globalkey(labels)
    labels["globalkey_tonic_fifths"] = globalkey_tonic_fifths
    labels["globalkey_mode"] = str(globalkey_mode.value)
    return labels
    

def make_labeled_pitch_array(
        notes: pd.DataFrame,
        labels: pd.DataFrame,
        measures: Optional[pd.DataFrame] = None
):
    pitch_array = make_pitch_array(notes, measures, label_notes=True)
    prepared_labels = prepare_labels(labels)
    merged = pd.merge(
        left = pitch_array, 
        right = prepared_labels, 
        on = "quarterbeats_playthrough",
        how = "outer",
        suffixes = ("", "_label"),
        indicator=True
    )
    return merged
    
    
# labeled_pitch_array = make_labeled_pitch_array(
#     notes=notes, 
#     labels=facets["expanded"],
#     measures=measures
# )
# 
# tmp_labeled_pitch_array = labeled_pitch_array[[
#     col for col 
#     in labeled_pitch_array.columns 
#     if col not in pitch_array.columns]]
tmp_labeled_pitch_array = prepare_labels(facets["expanded"])
tmp_labeled_pitch_array.to_csv("beethoven1_labeled.tsv", sep="\t", index=False)
tmp_labeled_pitch_array

# %%
