"""Turns a (score, annotation) pair into a joint pandas DataFrame."""

import re

import numpy as np
import pandas as pd

from . import annotation_parser
from . import score_parser
from .common import FIXEDOFFSET

J_COLUMNS = (
    score_parser.S_COLUMNS
    + annotation_parser.A_COLUMNS
    + [
        "qualityScoreNotes",
        "qualityNonChordTones",
        "qualityMissingChordTones",
        "qualitySquaredSum",
    ]
)

J_LISTTYPE_COLUMNS = (
    score_parser.S_LISTTYPE_COLUMNS
    + annotation_parser.A_LISTTYPE_COLUMNS
    + ["qualityScoreNotes"]
)


def _measureAlignmentScore(df):
    df["measureMisalignment"] = df.s_measure != df.a_measure
    return df


def _qualityMetric(df):
    df["qualityScoreNotes"] = np.nan
    df["qualityNonChordTones"] = np.nan
    df["qualityMissingChordTones"] = np.nan
    df["qualitySquaredSum"] = np.nan
    notesdf = df.explode("s_notes")
    annotations = df.a_annotationNumber.unique()
    for n in annotations:
        # All the rows spanning this annotation
        rows = notesdf[notesdf.a_annotationNumber == n]
        # No octave information; just pitch names
        scoreNotes = [re.sub(r"\d", "", n) for n in rows.s_notes]
        annotationNotes = rows.iloc[0].a_pitchNames
        missingChordTones = set(annotationNotes) - set(scoreNotes)
        nonChordTones = [n for n in scoreNotes if n not in annotationNotes]
        missingChordTonesScore = len(missingChordTones) / len(
            set(annotationNotes)
        )
        nonChordTonesScore = len(nonChordTones) / len(scoreNotes)
        squaredSumScore = (missingChordTonesScore + nonChordTonesScore) ** 2
        df.loc[df.a_annotationNumber == n, "qualityScoreNotes"] = str(
            scoreNotes
        )
        df.loc[df.a_annotationNumber == n, "qualityNonChordTones"] = round(
            nonChordTonesScore, 2
        )
        df.loc[df.a_annotationNumber == n, "qualityMissingChordTones"] = round(
            missingChordTonesScore, 2
        )
        df.loc[df.a_annotationNumber == n, "qualitySquaredSum"] = round(
            squaredSumScore, 2
        )
    df["qualityScoreNotes"] = df["qualityScoreNotes"].apply(eval)
    return df


def _inversionMetric(df):
    df["incongruentBass"] = np.nan
    annotationIndexes = df[df.a_isOnset].a_pitchNames.index.to_list()
    annotationBasses = df[df.a_isOnset].a_bass.to_list()
    annotationIndexes.append("end")
    annotationRanges = [
        (
            annotationIndexes[i],
            annotationIndexes[i + 1],
            annotationBasses[i],
        )
        for i in range(len(annotationBasses))
    ]
    for start, end, annotationBass in annotationRanges:
        if end == "end":
            slices = df[start:]
        else:
            slices = df[start:end].iloc[:-1]
        scoreBasses = [re.sub(r"\d", "", c[0]) for c in slices.s_notes]
        counts = scoreBasses.count(annotationBass)
        inversionScore = 1.0 - counts / len(scoreBasses)
        df.loc[slices.index, "incongruentBass"] = round(inversionScore, 2)
    return df


def parseAnnotationAndScore(
    a, s, qualityAssessment=True, fixedOffset=FIXEDOFFSET, eventBased=False
):
    """Process a RomanText and score files simultaneously.

    a is a RomanText file
    s is a .mxl|.krn|.musicxml file

    Create the dataframes of both. Generate a new, joint, one.
    """
    # Parse each file
    adf = annotation_parser.parseAnnotation(a, fixedOffset=fixedOffset, eventBased=eventBased)
    sdf = score_parser.parseScore(s, fixedOffset=fixedOffset, eventBased=eventBased)
    # Create the joint dataframe
    jointdf = pd.concat([sdf, adf], axis=1)
    jointdf.index.name = "j_offset"
    # Sometimes, scores are longer than annotations (trailing empty measures)
    # In that case, ffill the annotation portion of the new dataframe
    jointdf["a_isOnset"].fillna(False, inplace=True)
    jointdf.fillna(method="ffill", inplace=True)
    if qualityAssessment:
        jointdf = _measureAlignmentScore(jointdf)
        jointdf = _qualityMetric(jointdf)
        jointdf = _inversionMetric(jointdf)
    return jointdf

def m21_metadata2dict(metadata, key_prefix=None):
    if not key_prefix:
        return dict(metadata.all())
    return {f"{key_prefix}{k}": v for k, v in metadata.all()}

def parseAnnotationAndScoreEvents(
        a, s#, qualityAssessment=True
):
    """Process a RomanText and score files simultaneously.

    a is a RomanText file
    s is a .mxl|.krn|.musicxml file

    Create the dataframes of both. Generate a new, joint, one.
    """
    # Parse each file
    extended_adf = annotation_parser.parseAnnotationEvents(a)
    sdf = score_parser.parseScoreEvents(s)
    metadata = dict(
        m21_metadata2dict(extended_adf.metadata, "a_"),
        **m21_metadata2dict(sdf.metadata, "s_")
    )
    # Create the joint dataframe
    original_columns = [col for col in extended_adf.columns if col[:2] in ("a_", "s_")]
    adf = extended_adf[original_columns].copy()
    jointdf = pd.merge(
        left = sdf,
        right = adf,
        left_on = "s_offset",
        right_on = "a_offset",
        how = "outer"
    )
    # Sometimes, scores are longer than annotations (trailing empty measures)
    # In that case, ffill the annotation portion of the new dataframe
    jointdf["a_isOnset"].fillna(False, inplace=True)
    j_offset = jointdf.s_offset.rename("j_offset")
    labels_not_coinciding_with_any_note_mask = jointdf.s_offset.isna()
    if labels_not_coinciding_with_any_note_mask.any():
        # these are typically labels coinciding only with rests
        # there is, however, a residue risk that they are symptom of a score-annotation misalignment
        j_offset = j_offset.fillna(jointdf.a_offset)
        print(f"Score has {labels_not_coinciding_with_any_note_mask.sum()} labels not coinciding with any note.")
    jointdf.index = j_offset
    jointdf = jointdf.sort_index()
    # forward-fill annotation label features only, do not fill note features for label onsets
    jointdf.loc[:, adf.columns] = jointdf.loc[:, adf.columns].ffill()
    jointdf = jointdf.drop(columns=["s_offset", "a_offset"])
    # if qualityAssessment:
    #     jointdf = _measureAlignmentScore(jointdf)
    #     jointdf = _qualityMetric(jointdf)
    #     jointdf = _inversionMetric(jointdf)

    extended_adf = extended_adf.rename(columns=dict(
        a_offset = "quarterbeats",
        a_measure = "mn",
    ))
    return extended_adf, sdf, jointdf, metadata


def parseAnnotationAndAnnotation(
    a, qualityAssessment=True, fixedOffset=FIXEDOFFSET, texturize=True
):
    """Synthesize a RomanText file to treat it as both analysis and score.

    a is a RomanText file

    When synthesizing the file, texturize it if `texturize=True`
    """
    adf = annotation_parser.parseAnnotation(a, fixedOffset=fixedOffset)
    sdf = score_parser.parseAnnotationAsScore(
        a, texturize=texturize, fixedOffset=fixedOffset
    )
    jointdf = pd.concat([sdf, adf], axis=1)
    jointdf["a_isOnset"].fillna(False, inplace=True)
    jointdf.fillna(method="ffill", inplace=True)
    if qualityAssessment:
        jointdf = _measureAlignmentScore(jointdf)
        jointdf = _qualityMetric(jointdf)
        jointdf = _inversionMetric(jointdf)
    return jointdf
