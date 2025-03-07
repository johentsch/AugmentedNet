# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: dimcat
#     language: python
#     name: dimcat
# ---

# %%

import os
from typing import Optional, Tuple

import ms3
from dimcat.data.resources.facets import extend_harmony_feature, extend_keys_feature, extend_cadence_feature
import pandas as pd

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
    section_start |= (measures.repeats == "start").fillna(False)
    section_start |= (measures.repeats.shift() == "end").fillna(False)
    section_start |= measures.breaks.shift().str.contains("section").fillna(False)
    section_start |= (measures.barline.shift() == "double").fillna(False)
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

# make_section_start_column(measures)


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
                         "staff", "voice", "is_note_onset", "tpc"]
KEEP_ORIGINAL_LABEL_COLUMNS = ["section_start"]
RENAME_ORIGINAL_COLUMNS = dict(
    midi="pitch",
    keysig="ks_fifths"
)
COLUMN_ORDER = ["onset_div", "duration_div", "pitch", "tpc", "step", "alter", "ts_beats", "ts_beat_type", "staff", "voice"]


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
def convert_roman_numerals_to_fifths(labels: pd.DataFrame) -> pd.DataFrame:
    concatenate_this = [
        labels,
        (
            globalkey_tpc := ms3.transform(
                labels.globalkey,
                ms3.name2fifths,
            )
        ).rename("globalkey_tpc"),
        (
                ms3.transform(
                    labels[["localkey", "globalkey_is_minor"]], ms3.roman_numeral2fifths
                ) + globalkey_tpc
        ).rename("localkey_tpc"),
        (
                ms3.transform(
                    labels[["effective_localkey_resolved", "globalkey_is_minor"]], ms3.roman_numeral2fifths
                ) + globalkey_tpc
        ).rename("tonicized_tpc"),
    ]
    labels = pd.concat(concatenate_this, axis=1)
    return labels

INT_COLUMNS = ['unfolded_harmony_index', 'root', 'bass_note', 'globalkey_tpc', 'localkey_tpc', 'tonicized_tpc', ]
BOOL_COLUMNS = ['globalkey_is_minor', 'localkey_is_minor', 'is_harmony_onset', ]
STRING_COLUMNS = ['section_start', 'label', 'alt_label', 'globalkey', 'localkey', 'pedal', 'chord', 'special', 'numeral', 'form', 'figbass', 'changes', 'relativeroot', 'cadence', 'phraseend', 'chord_type', 'globalkey_mode', 'localkey_mode', 'localkey_resolved', 'localkey_and_mode', 'root_roman', 'relativeroot_resolved', 'effective_localkey', 'effective_localkey_resolved', 'effective_localkey_is_minor', 'chord_reduced', 'chord_reduced_and_mode', 'pedal_resolved', 'chord_and_mode', 'applied_to_numeral', 'numeral_or_applied_to_numeral', 'cadence_type', '_merge']
OBJECT_COLUMNS = ['chord_tones', 'added_tones', ] # unused, leave them as they are

def convert_column_types(labels: pd.DataFrame) -> pd.DataFrame:
    conversion_dict = {col: "Int64" for col in INT_COLUMNS if col in labels.columns}
    conversion_dict.update(
        {col: "boolean" for col in BOOL_COLUMNS if col in labels.columns}
    )
    conversion_dict.update(
        {col: "string" for col in STRING_COLUMNS if col in labels.columns}
    )
    return labels.astype(conversion_dict)


NON_FORWARD_FILLING_COLUMNS = [
    "is_harmony_onset", "cadence", "cadence_type", "cadence_subtype", "phraseend", "section_start"
] # these are not propagated over the whole duration of their harmony label and are therefore moved to the left

def prepare_labels(labels: pd.DataFrame) -> pd.DataFrame:
    labels = labels.copy()
    labels["is_harmony_onset"] = True
    labels.is_harmony_onset = labels.is_harmony_onset.where(
        labels.chord.notna() & (labels.chord != labels.chord.shift(-1)),
        False, # set False where the label does not define a harmony or merely the same harmony as the preceding one
    ) 
    labels.index.rename("unfolded_harmony_index", inplace=True)
    labels.reset_index(drop=False, inplace=True)
    labels = extend_keys_feature(labels)
    labels = extend_harmony_feature(labels)
    labels = convert_roman_numerals_to_fifths(labels)
    labels = extend_cadence_feature(labels)
    labels = convert_column_types(labels)
    column_order = [col for col in NON_FORWARD_FILLING_COLUMNS if col in labels.columns]
    column_order += [col for col in labels.columns if col not in column_order]
    return labels[column_order]

def compute_interval_classes_to_keys(merged: pd.DataFrame) -> pd.DataFrame:
    concatenate_this = [
        merged,
        (merged.tpc - merged.globalkey_tpc).rename("sic_with_global"),
        (merged.tpc - merged.localkey_tpc).rename("sic_with_local"),
        (merged.tpc - merged.tonicized_tpc).rename("sic_with_tonicized"),
    ]
    return pd.concat(concatenate_this, axis=1)

def add_boolean_label_columns(merged: pd.DataFrame) -> pd.DataFrame:
    
    def is_in_chord_tones(sic: int, chord_tones: Tuple[int]) -> bool:
        """Used for element-wise containment check"""
        return sic in chord_tones
    
    concatenate_this = [
        merged,
        ms3.transform(
            labeled_pitch_array, 
            is_in_chord_tones, 
            ["sic_with_local", "chord_tones"]
        ).rename("tpc_is_in_label"),
        (merged.sic_with_local == merged.root).rename("tpc_is_root"),
        (merged.sic_with_local == merged.bass_note).rename("tpc_is_bass")
    ]
    return pd.concat(concatenate_this, axis=1)

def make_labeled_pitch_array(
        notes: pd.DataFrame,
        labels: pd.DataFrame,
        measures: Optional[pd.DataFrame] = None
):
    pitch_array = make_pitch_array(notes, measures, label_notes=True)
    prepared_labels = prepare_labels(labels)
    
    merged = pd.merge(
        left = pitch_array, 
        right = prepared_labels.drop(columns=[
            "mc", "mn", "mc_playthrough", "mn_playthrough", "quarterbeats_all_endings", "duration_qb", "mc_onset",
            "mn_onset", "timesig", "staff", "voice"
        ]), 
        on = "quarterbeats_playthrough",
        how = "outer",
        suffixes = ("", "_label"),
        indicator=False
    )
    merged.is_harmony_onset = merged.is_harmony_onset.fillna(False)
    
    harmony_index_col = merged.columns.get_loc("unfolded_harmony_index")
    pitch_side = merged.iloc[:, :harmony_index_col]
    harmony_side = merged.iloc[:, harmony_index_col:]

    harmony_grouper = (harmony_side.unfolded_harmony_index.
                       where(harmony_side.chord.notna()).   # takes only index positions for which a harmony is defined
                       ffill())                             # and forward-fills gaps with indices of the harmonies
    merged = pd.concat([
        pitch_side,
        harmony_side.groupby(harmony_grouper).ffill()
    ], axis=1)
    merged = compute_interval_classes_to_keys(merged)
    merged = add_boolean_label_columns(merged)
    return merged
    
    
labeled_pitch_array = make_labeled_pitch_array(
    notes=notes, 
    labels=facets["expanded"],
    measures=measures
)

tmp_labeled_pitch_array = labeled_pitch_array[[
    col for col 
    in labeled_pitch_array.columns 
    if col not in pitch_array.columns]]
tmp_labeled_pitch_array = prepare_labels(facets["expanded"])
labeled_pitch_array.to_csv("beethoven1_labeled.tsv", sep="\t", index=False)
labeled_pitch_array

# %%
# concatenate_this = [
#     # adds columns to harmony labels before joining on the notes
#     feature_df,
#     
#     (feature_df.root + localkey_tpc).rename("root_per_globalkey"),
#     (feature_df.root - relativeroot_tpc).rename("root_per_tonicization"),
#     ms3.transform(
#             feature_df[["relativeroot_resolved", "localkey_is_minor"]],
#             ms3.roman_numeral2fifths,
#         ).fillna(0).rename("relativeroot_tpc"),
# ]
