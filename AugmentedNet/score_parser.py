"""Turns a MusicXML file into a pandas DataFrame."""

import io
import warnings
from fractions import Fraction
from itertools import combinations
from typing import Optional

import music21
import numpy as np
import pandas as pd
from music21.chord import Chord
from music21.interval import Interval
from music21.note import Note, Rest
from music21.pitch import Pitch
from music21.stream import Part, Voice

from .common import FIXEDOFFSET, FLOATSCALE
from .texturizers import (
    applyTextureTemplate,
    available_durations,
    available_number_of_notes,
)
from .utils import make_continuous_mc_beats_series, onset2beat

S_COLUMNS = [
    "s_offset",
    "s_duration",
    "s_measure",
    "s_notes",
    "s_intervals",
    "s_isOnset",
]

S_LISTTYPE_COLUMNS = [
    "s_notes",
    "s_intervals",
    "s_isOnset",
]


def _m21Parse(f, fmt=None):
    return music21.converter.parse(f, format=fmt)


def _measureNumberShift(m21Score):
    firstMeasure = m21Score.parts[0].measure(0) or m21Score.parts[0].measure(1)
    isAnacrusis = True if firstMeasure.paddingLeft > 0.0 else False
    if isAnacrusis and firstMeasure.number == 1:
        measureNumberShift = -1
    else:
        measureNumberShift = 0
    return measureNumberShift


def _lastOffset(m21Score):
    lastMeasure = m21Score.parts[0].measure(-1)
    filledDuration = lastMeasure.duration.quarterLength / float(
        lastMeasure.barDurationProportion()
    )
    lastOffset = lastMeasure.offset + filledDuration
    return lastOffset


def _initialDataFrame(s, fmt=None):
    """Parses a score and produces a pandas dataframe.

    The features obtained are the note names, their position in the score,
    measure number, and their ties (in case something fancy needs to be done,
    with the tie information).
    """
    dfdict = {col: [] for col in S_COLUMNS}
    measureNumberShift = _measureNumberShift(s)
    for c in s.chordify().flat.notesAndRests:
        dfdict["s_offset"].append(round(float(c.offset), FLOATSCALE))
        dfdict["s_duration"].append(round(float(c.quarterLength), FLOATSCALE))
        dfdict["s_measure"].append(c.measureNumber + measureNumberShift)
        if isinstance(c, Rest):
            # We need dummy entries for rests at the beginning of a measure
            dfdict["s_notes"].append(np.nan)
            dfdict["s_intervals"].append(np.nan)
            dfdict["s_isOnset"].append(np.nan)
            continue
        dfdict["s_notes"].append([n.pitch.nameWithOctave for n in c])
        intvs = [Interval(c[0].pitch, p).simpleName for p in c.pitches[1:]]
        dfdict["s_intervals"].append(intvs)
        onsets = [(not n.tie or n.tie.type == "start") for n in c]
        dfdict["s_isOnset"].append(onsets)
    df = pd.DataFrame(dfdict)
    currentLastOffset = float(df.tail(1).s_offset) + float(df.tail(1).s_duration)
    deltaDuration = _lastOffset(s) - currentLastOffset
    df.loc[len(df) - 1, "s_duration"] += deltaDuration
    df.set_index("s_offset", inplace=True)
    df = df[~df.index.duplicated()]
    return df


def make_interval_index(measures: pd.DataFrame) -> pd.IntervalIndex:
    breaks = measures.offset.tolist()
    last_measure = measures.iloc[-1]
    end_of_piece = breaks[-1] + last_measure.duration
    breaks.append(end_of_piece)
    return pd.IntervalIndex.from_breaks(breaks, closed="left")


def shift_cumsum(durations: pd.Series):
    cumsum = np.cumsum(durations.div(4).to_list())
    shifted_cumsum = [Fraction(0)] + list(cumsum)[:-1]
    return shifted_cumsum


def make_mc_offset_column(measures):
    result = []
    for ix, grouped_durations in measures.groupby("measureNumber").duration:
        if len(grouped_durations) == 1:
            result.append(0)
        else:
            result.extend(shift_cumsum(grouped_durations))
    return pd.Series(result, index=measures.index, name="mc_offset")


def make_measureOffsetMap_with_ks_and_ts(score: music21.stream.Score) -> dict:
    measureOffsetMap = {}
    for offset, measure_list in score.measureOffsetMap().items():
        first_measure, *other_measures = measure_list
        for m in other_measures:
            if first_measure.keySignature and first_measure.timeSignature:
                break
            if not first_measure.keySignature and m.keySignature:
                first_measure.keySignature = m.keySignature
            if not first_measure.timeSignature and m.timeSignature:
                first_measure.timeSignature = m.timeSignature
        measureOffsetMap[offset] = first_measure
    return measureOffsetMap


def get_measures_table(score: music21.stream.Score):
    offset2measure_object = make_measureOffsetMap_with_ks_and_ts(score)
    offset2measure_features = {
        offset: dict(
            mc=mc,
            offset=m.offset,
            quarterbeats=Fraction(m.offset),
            measureNumber=m.measureNumber,
            measureNumberWithSuffix=m.measureNumberWithSuffix(),
            barDuration=Fraction(m.barDuration.quarterLength),
            duration=Fraction(m.duration.quarterLength),
            ks_fifths=m.keySignature.sharps if m.keySignature else pd.NA,
            timeSignature=m.timeSignature.ratioString if m.timeSignature else pd.NA,
        )
        for mc, (offset, m) in enumerate(offset2measure_object.items(), 1)
    }
    measures = pd.DataFrame.from_dict(offset2measure_features, orient="index")
    measures.ks_fifths = measures.ks_fifths.astype("Int64").ffill()
    measures.timeSignature = measures.timeSignature.ffill()
    measures.index = make_interval_index(measures)
    measures["act_dur"] = measures.duration / 4
    measures = pd.concat(
        [
            measures,
            make_continuous_mc_beats_series(measures, beat_decimals=3),
            measures.timeSignature.str.extract(
                r"^(?P<ts_beats>\d+)/(?P<ts_beat_type>\d+)$"
            ),
            make_mc_offset_column(measures),
        ],
        axis=1,
    )
    return measures


def extendedDataFrame(s, fmt=None):
    """Parses a score and produces a pandas dataframe.

    The features obtained are the note names, their position in the score,
    measure number, and their ties (in case something fancy needs to be done,
    with the tie information).
    """
    df_records = []
    measureNumberShift = _measureNumberShift(s)

    measures = get_measures_table(s)

    def add_note(
        note, part_id: Optional[str] = None, voice_id: Optional[str] = None
    ) -> None:
        """Updates the row's dfdict with the note information and adds it to the records."""
        nonlocal dfdict, measure_info
        if not part_id and (part := note.getContextByClass(Part)):
            part_id = part.id
        if not voice_id and (voice := note.getContextByClass(Voice)):
            voice_id = voice.id
        else:
            voice_id = "1"

        timesig = measure_info.get("timeSignature")
        measure_offset = measure_info.get("quarterbeats")
        note_offset = dfdict.get("s_offset_frac")
        relative_note_offset = note_offset - measure_offset
        mn_onset = Fraction(relative_note_offset) / 4 + measure_info.get("mc_offset")
        beat_float = onset2beat(mn_onset, timesig=timesig, beat_decimals=3)
        continuous_beats = measure_info.get("continuous_beats") + beat_float - 1
        p = note.pitch
        note_record = dict(
            dfdict,
            continuous_beats=continuous_beats,
            s_note=p.nameWithOctave,
            s_midi=p.midi,
            s_isOnset=(not note.tie or note.tie.type == "start"),
            s_step=p.step,
            s_alter=int(p.alter),
            mn_onset=mn_onset,
            s_beat_float=beat_float,
            s_downbeat=int(beat_float) if beat_float.is_integer() else 0,
            s_part_id=part_id,
            s_voice_id=voice_id,
        )
        df_records.append(note_record)

    for note_or_rest in s.semiFlat.notesAndRests:
        measure_info = measures.loc[note_or_rest.offset].to_dict()
        dfdict = dict(
            s_offset=round(float(note_or_rest.offset), FLOATSCALE),
            s_offset_frac=Fraction(note_or_rest.offset),
            s_duration=round(float(note_or_rest.quarterLength), FLOATSCALE),
            s_duration_frac=Fraction(note_or_rest.quarterLength),
            s_measure=note_or_rest.measureNumber + measureNumberShift,
        )
        for key in ("ks_fifths", "ts_beats", "ts_beat_type", "measureNumberWithSuffix"):
            dfdict[key] = measure_info[key]
        if isinstance(note_or_rest, Rest):
            # Different from AugmentedNet, we don't need dummy entries for rests at the beginning of a measure
            # The code was left here (rather than iterating through s.notes in the first place) in case someone
            # needs the rests
            # dfdict.update(
            #     s_notes = np.nan,
            #     s_isOnset = np.nan
            # )
            # df_records.append(dfdict)
            continue

        if isinstance(note_or_rest, Note):
            add_note(note_or_rest)
        elif isinstance(note_or_rest, Chord):
            part_id = (
                part.id if (part := note_or_rest.getContextByClass(Part)) else None
            )
            voice_id = (
                voice.id if (voice := note_or_rest.getContextByClass(Voice)) else None
            )
            for note in note_or_rest:
                add_note(note, part_id=part_id, voice_id=voice_id)
        else:
            warnings.warn(
                f"Encountered unexpected music21 object: {type(note_or_rest)!r}"
            )
            continue
    df = pd.DataFrame.from_records(df_records)

    currentLastOffset = float(df.iloc[-1].s_offset) + float(df.iloc[-1].s_duration)
    last_offset = measures.index.values[-1].right
    deltaDuration = last_offset - currentLastOffset
    df.loc[len(df) - 1, "s_duration"] += deltaDuration
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
    maxOffset = (lastRow.index + lastRow.s_duration).to_numpy()[0]
    newIndex = np.arange(minOffset, maxOffset, fixedOffset)
    # All operations done over the full index, i.e., fixed-timesteps
    # plus original onsets. Later, original onsets (e.g., triplets)
    # are removed and just the fixed-timesteps are kept
    df = df.reindex(index=df.index.union(newIndex))
    df.s_notes.fillna(method="ffill", inplace=True)
    df.s_notes.fillna(method="bfill", inplace=True)
    # the "isOnset" column is hard to generate in fixed-timesteps
    # however, it allows us to encode a "hold" symbol if we wanted to
    newCol = pd.Series(
        [[False] * n for n in df.s_notes.str.len().to_list()], index=df.index
    )
    df.s_isOnset.fillna(value=newCol, inplace=True)
    df.fillna(method="ffill", inplace=True)
    df.fillna(method="bfill", inplace=True)
    df = df.reindex(index=newIndex)
    return df


def _engraveScore(df):
    """Useful for debugging _texturizeAnnotationScore."""
    chords = music21.stream.Stream()
    for row in df.itertuples():
        if row.s_measure == 0:
            continue
        pitches = row.s_notes
        duration = Fraction(row.s_duration).limit_denominator(2048)
        chord = Chord(pitches, quarterLength=duration)
        chords.append(chord)
    return chords


def _texturizeAnnotationScore(df, duration, numberOfNotes):
    # Preemptively, remove any notion of held notes in an annotation file
    df["s_isOnset"] = df.s_isOnset.apply(lambda lst: [True for _ in lst])
    outputdf = df.copy()
    # A copy because we don't want these two temporary columns in the output
    df["notesNumber"] = df.s_notes.apply(len)
    df["allOnsets"] = df.s_isOnset.apply(all)
    # Which block chords can we replace with a more complex texture
    replaceable = df[
        (df.s_duration == duration) & (df.notesNumber == numberOfNotes) & (df.allOnsets)
    ]
    for row in replaceable.itertuples():
        offset = row.Index
        measure = row.s_measure
        notes = row.s_notes
        intervals = [
            Interval(Pitch(n1), Pitch(n2)).simpleName
            for n1, n2 in combinations(notes, 2)
        ]
        texture = applyTextureTemplate(duration, notes, intervals)
        textureF = io.StringIO(texture)
        texturedf = pd.read_csv(textureF)
        texturedf["s_offset"] += offset
        texturedf["s_measure"] = measure
        for col in S_LISTTYPE_COLUMNS:
            texturedf[col] = texturedf[col].apply(eval)
        texturedf.set_index("s_offset", inplace=True)
        for index, row in texturedf.iterrows():
            outputdf.loc[index] = row
    outputdf.sort_index(inplace=True)
    return outputdf


def parseScore(f, fmt=None, fixedOffset=FIXEDOFFSET, eventBased=False):
    # Step 0: Use music21 to parse the score
    s = _m21Parse(f, fmt)
    # Step 1: Parse and produce a salami-sliced dataset
    df = _initialDataFrame(s, fmt)
    # Step 2: Turn salami-slice into fixed-duration steps
    if not eventBased:
        df = _reindexDataFrame(df, fixedOffset=fixedOffset)
    df.metadata = s.metadata
    return df


def parseScoreEvents(f, fmt=None):
    # Step 0: Use music21 to parse the score
    s = _m21Parse(f, fmt)
    df = extendedDataFrame(s, fmt)
    df.metadata = s.metadata
    return df


def parseAnnotationAsScore(
    f, texturize=False, fixedOffset=FIXEDOFFSET, eventBased=False
):
    """Generates a DataFrame from a synthesized RomanText file.

    Args:
        f (string): The path to the input RomanText file.
        texturize (bool, optional): Texturize the synthetic score. Defaults to False.
        fixedOffset (float, optional): The sampling rate in quarter notes. Defaults to FIXEDOFFSET.
        eventBased (bool, optional): If True, no fixedOffset sampling is done. Defaults to False.

    Returns:
        DataFrame: The output DataFrame
    """
    fmt = "romantext"
    if not texturize:
        return parseScore(f, fmt=fmt, fixedOffset=fixedOffset)
    # Step 0: Use music21 to parse the score
    s = _m21Parse(f, fmt=fmt)
    # Step 1: Parse and produce a salami-sliced dataset
    df = _initialDataFrame(s, fmt=fmt)
    # Step 2: Texturize the dataframe
    for duration in available_durations:
        for numberOfNotes in available_number_of_notes:
            df = _texturizeAnnotationScore(df, duration, numberOfNotes)
    # Step 3: Turn salami-slice into fixed-duration steps
    if not eventBased:
        df = _reindexDataFrame(df, fixedOffset=fixedOffset)
    return df
