"""Turns a RomanText file into a pandas DataFrame."""

import re
from fractions import Fraction
from typing import List

import music21
import numpy as np
import pandas as pd

from . import utils
from .common import FIXEDOFFSET, FLOATSCALE

A_COLUMNS = [
    "a_offset",
    "a_measure",
    "a_duration",
    "a_annotationNumber",
    "a_romanNumeral",
    "a_isOnset",
    "a_pitchNames",
    "a_bass",
    "a_root",
    "a_inversion",
    "a_quality",
    "a_pcset",
    "a_localKey",
    "a_tonicizedKey",
    "a_degree1",
    "a_degree2",
]


# These have to be made lists again if read from a csv. They're stored as str.
A_LISTTYPE_COLUMNS = [
    "a_pitchNames",
    "a_pcset",
]


def _m21Parse(f):
    return music21.converter.parse(f, format="romantext")


def _fixRnSynonyms(figure):
    ret = figure.replace("6/4", "64")
    ret = ret.replace("6/5", "65")
    ret = ret.replace("4/3", "43")
    ret = ret.replace("4/2", "42")
    ret = ret.replace("42", "2")
    ret = ret.replace("bII", "N")
    return ret


def _simplifyRomanNumeral(figure):
    missingAdd = re.compile(r"(\[.*\])")
    return missingAdd.sub("", figure)


def _removeInversion(figure):
    ret = figure.replace("65", "7")
    ret = ret.replace("43", "7")
    ret = ret.replace("64", "")
    ret = ret.replace("6", "")
    ret = ret.replace("2", "7")
    return ret


def _preprocessRomanNumeral(figure):
    return _removeInversion(_simplifyRomanNumeral(_fixRnSynonyms(figure)))


def _initialDataFrame(s):
    """Parses an annotation RomanText file and produces a pandas dataframe.

    Unpacking a roman numeral is slightly more complicated here than in
    previous approaches/papers, the reason is that I include more features
    than usual (e.g., inversion). It may be easier to predict which features
    lead to a better Roman numeral reconstruction this way.
    """
    dfdict = {col: [] for col in A_COLUMNS}
    for idx, rn in enumerate(s.flatten().getElementsByClass("RomanNumeral")):
        dfdict["a_offset"].append(round(float(rn.offset), FLOATSCALE))
        dfdict["a_measure"].append(rn.measureNumber)
        dfdict["a_duration"].append(round(float(rn.quarterLength), FLOATSCALE))
        dfdict["a_annotationNumber"].append(idx)
        dfdict["a_romanNumeral"].append(_preprocessRomanNumeral(rn.figure))
        dfdict["a_isOnset"].append(True)
        dfdict["a_pitchNames"].append(tuple(rn.pitchNames))
        dfdict["a_bass"].append(rn.pitchNames[0])
        dfdict["a_root"].append(rn.root().name)
        dfdict["a_inversion"].append(rn.inversion())
        dfdict["a_quality"].append(rn.commonName)
        dfdict["a_pcset"].append(tuple(sorted(set(rn.pitchClasses))))
        localKey = rn.key.tonicPitchNameWithCase
        dfdict["a_localKey"].append(localKey)
        secondaryKey = rn.secondaryRomanNumeralKey
        if secondaryKey:
            tonicizedKey = secondaryKey.tonicPitchNameWithCase
            dfdict["a_tonicizedKey"].append(tonicizedKey)
        else:
            # if there is no tonicization, encode the local key
            dfdict["a_tonicizedKey"].append(localKey)
        scaleDegree, alteration = rn.scaleDegreeWithAlteration
        if alteration:
            scaleDegree = f"{alteration.modifier}{scaleDegree}"
        else:
            scaleDegree = f"{scaleDegree}"
        dfdict["a_degree1"].append(str(scaleDegree))
        secondaryDegree = rn.secondaryRomanNumeral
        if secondaryDegree:
            scaleDegree, alteration = secondaryDegree.scaleDegreeWithAlteration
            if alteration:
                scaleDegree = f"{alteration.modifier}{scaleDegree}"
            else:
                scaleDegree = f"{scaleDegree}"
            dfdict["a_degree2"].append(scaleDegree)
        else:
            dfdict["a_degree2"].append("None")
    df = pd.DataFrame(dfdict)
    df.set_index("a_offset", inplace=True)
    return df


def extendedDataFrame(s):
    """Parses an annotation RomanText file and produces a pandas dataframe.

    Unpacking a roman numeral is slightly more complicated here than in
    previous approaches/papers, the reason is that I include more features
    than usual (e.g., inversion). It may be easier to predict which features
    lead to a better Roman numeral reconstruction this way.
    """
    df_records = []
    first_key = next(s.flatten().getElementsByClass("RomanNumeral")).key
    globalkey = first_key.tonicPitchNameWithCase.replace("-", "b")
    for idx, rn in enumerate(s.flatten().getElementsByClass("RomanNumeral")):
        dfdict = dict(
            a_offset=round(float(rn.offset), FLOATSCALE),
            a_measure=rn.measureNumber,
            mn_onset=Fraction((rn.beat - 1) * rn.beatDuration.quarterLength / 4),
            a_duration=round(float(rn.quarterLength), FLOATSCALE),
            a_annotationNumber=idx,
            label=rn.figure,
            a_romanNumeral=_preprocessRomanNumeral(rn.figure),
            a_isOnset=True,
            a_pitchNames=tuple(rn.pitchNames),
            a_bass=rn.pitchNames[0],
            a_root=rn.root().name,
            a_inversion=rn.inversion(),
            a_quality=rn.commonName,
            a_pcset=tuple(sorted(set(rn.pitchClasses))),
        )
        localKey = rn.key.tonicPitchNameWithCase
        dfdict["a_localKey"] = localKey
        dfdict["localkey_abs"] = localKey.replace("-", "b")
        dfdict["globalkey"] = globalkey
        secondaryKey = rn.secondaryRomanNumeralKey
        if secondaryKey:
            tonicizedKey = secondaryKey.tonicPitchNameWithCase
            dfdict["a_tonicizedKey"] = tonicizedKey
        else:
            # if there is no tonicization, encode the local key
            dfdict["a_tonicizedKey"] = localKey
        scaleDegree, alteration = rn.scaleDegreeWithAlteration
        if alteration:
            scaleDegree = f"{alteration.modifier}{scaleDegree}"
        else:
            scaleDegree = f"{scaleDegree}"
        dfdict["a_degree1"] = str(scaleDegree)
        secondaryDegree = rn.secondaryRomanNumeral
        if secondaryDegree:
            scaleDegree, alteration = secondaryDegree.scaleDegreeWithAlteration
            if alteration:
                scaleDegree = f"{alteration.modifier}{scaleDegree}"
            else:
                scaleDegree = f"{scaleDegree}"
            dfdict["a_degree2"] = scaleDegree
        else:
            dfdict["a_degree2"] = "None"
        df_records.append(dfdict)
    df = pd.DataFrame.from_records(df_records)
    return df


def _reindexDataFrame(df, fixedOffset=FIXEDOFFSET):
    """Reindexes a dataframe according to a fixed note-value.

    It could be said that the DataFrame produced by parseScore
    is a "salami-sliced" version of the score. This is intuitive
    for humans, but does not really work in machine learning.

    What works, is to slice the score in fixed note intervals,
    for example, a sixteenth note. This reindex function does
    exactly that.
    """
    firstRow = df.head(1)
    lastRow = df.tail(1)
    minOffset = firstRow.index.to_numpy()[0]
    maxOffset = (lastRow.index + lastRow.a_duration).to_numpy()[0]
    newIndex = np.arange(minOffset, maxOffset, fixedOffset)
    # All operations done over the full index, i.e., fixed-timesteps
    # plus original onsets. Later, original onsets (e.g., triplets)
    # are removed and just the fixed-timesteps are kept
    df = df.reindex(index=df.index.union(newIndex))
    # here onsets are easier, every "injected" index is not an onset
    df.a_isOnset = df.a_isOnset.fillna(value=False).astype("boolean")
    df = df.ffill()
    df = df.reindex(index=newIndex)
    return df


def parseAnnotation(f, fixedOffset=FIXEDOFFSET, eventBased=False):
    """Generates the DataFrame from a RomanText file.

    Parses the file using music21. Creates an initial DataFrame
    with every onset event of the music21 stream. Finally,
    does the sampling at symbolically regular durations fixedOffset.
    """
    # Step 0: Use music21 to parse the score
    s = _m21Parse(f)
    # Step 1: Parse and produce a salami-sliced dataset
    df = _initialDataFrame(s)
    # Step 2: Turn salami-slice into fixed-duration steps
    if not eventBased:
        df = _reindexDataFrame(df, fixedOffset=fixedOffset)
    df.metadata = s.metadata
    return df


def get_cad64_index(group_df: pd.DataFrame) -> List[int]:
    """Takes a group of rows of an inversed annotations DataFrame whose last label / first row
    represents a sonority on scale degree 5: returns the indices of all rows that immediately
    follow that sonority and represent a 2nd-inversion tonic in the same key.
    """
    if len(group_df) == 1:
        return []
    cad64_idx = []
    for i, current_row in enumerate(group_df.itertuples()):
        if i == 0:
            previous_row = current_row
            continue
        if (
            current_row.a_degree1 != "1" or current_row.a_inversion != 2
        ):  # ToDo: to integer
            break
        if current_row.a_localKey != previous_row.a_localKey:
            break
        if current_row.a_degree2 != previous_row.a_degree2:
            break
        cad64_idx.append(current_row.Index)
        previous_row = current_row
    return cad64_idx


def replace_cadential_64(df, verbose=True):
    """Replaces all 2nd-inversion tonics directly preceding a sonority on scale degree 5 with
    dominant triads. This function mutates the original DataFrame.
    """
    df_reversed = df.iloc[::-1]
    root5_mask = df_reversed.a_degree1 == "5"
    root5_groups = root5_mask.cumsum()
    cad64_selector = sorted(
        df_reversed.groupby(root5_groups).apply(get_cad64_index).sum()
    )
    df.loc[cad64_selector, "a_romanNumeral"] = "V"
    df.loc[cad64_selector, "a_simpleNumeral"] = "V"
    df.loc[cad64_selector, "a_root"] = df.loc[cad64_selector, "a_bass"]
    df.loc[cad64_selector, "a_inversion"] = 0
    df.loc[cad64_selector, "a_quality"] = "major triad"
    df.loc[cad64_selector, "a_degree1"] = "5"
    if verbose:
        tonic_2nd_inv_mask = (df.a_degree1 == "1") & (df.a_inversion == 2)
        print(
            f"{len(cad64_selector)} 2nd-inversion tonics interpreted as V(64), "
            f"{tonic_2nd_inv_mask.sum()} left untouched."
        )
    return df


def parseAnnotationEvents(f):
    """Generates the DataFrame from a RomanText file.

    Parses the file using music21. Creates an initial DataFrame
    with every onset event of the music21 stream. Finally,
    does the sampling at symbolically regular durations fixedOffset.
    """
    # Step 0: Use music21 to parse the score
    s = _m21Parse(f)
    df = extendedDataFrame(s)
    df = utils.convert_romanNumeral_to_simpleNumeral(df)
    replace_cadential_64(df)
    df.metadata = s.metadata
    return df
