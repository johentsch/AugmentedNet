"""Turns a (score, annotation) pair into a joint pandas DataFrame."""
import itertools
import re
import warnings
from fractions import Fraction
from functools import lru_cache
from typing import Iterable, Optional, Union, Tuple

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

# region DivMaker

class DivMaker():
    """This is a convenient object for turning sequences of fractions into commensurate divs.
    It is equivalent to concatenating all sequences, passing them to the function shown below, and splitting them again.

        def fractions2divs(fracs: Iterable[Fraction]) -> NDArray[int]:
            numerators, denominators = np.array([(item.numerator,item.denominator) for item in fracs]).T
            lcm = np.lcm.reduce(denominators) # least common multiple
            return (numerators * lcm / denominators).astype(int)

    Example:

        STAR_WARS = np.array([ # durations of the star wars theme
            (1, 12),
            (1, 12),
            (1, 12),
            (1, 2),
            (1, 2),
            (1, 12),
            (1, 12),
            (1, 12),
            (1, 2),
            (1, 4)
        ])
        div_maker = DivMaker(STAR_WARS)
        div_maker[0] # yields [1, 1, 1, 6, 6, 1, 1, 1, 6, 3]

        POSITIONS = [Fraction(1, 20), Fraction(1, 32)] # fractions that we need our durations to be commensurate with
        div_maker.add_iterable_of_fractions(POSITIONS)
        durations, pos = div_maker # object is iterable (iterates through sequences added without names)
        list(durations) # yields [40, 40, 40, 240, 240, 40, 40, 40, 240, 120]

        OTHER_VALUES = (Fraction(i, 7) for i in range(7))
        div_maker.add_iterable_or_array(OTHER_VALUES, "other") # add the other values with a name
        div_maker[(1, "other", 0)] # when retrieving we can mix assigned names and indices of nameless sequences
        # OUTPUT:
        # (array([168, 105]),
        #  array([   0,  480,  960, 1440, 1920, 2400, 2880]),
        #  array([ 280,  280,  280, 1680, 1680,  280,  280,  280, 1680,  840]))

        div_maker.lcm # yields 3660, the common denominator for all values (least common multiple)
    """


    def __init__(
            self,
            *iterable_or_array: Iterable[Fraction],
            **named_iterables_or_arrays: Iterable[Fraction]
    ):
        """Pass one or several 2d-arrays (where one axis has shape 2) or one or several iterables of fractions.
        By passing keyword arguments you can assign names which you can use to retrieve the respective div sequences.
        """
        self.dict_of_frac_arrays = {}
        for ioa in iterable_or_array:
            _ = self.add_iterable_or_array(ioa)
        for name, ioa in named_iterables_or_arrays.items():
            _ = self.add_iterable_or_array(ioa, name)

    @staticmethod
    def iterable_of_fractions_to_array(
            iterable_of_fractions: Iterable[Fraction]
    ):
        """Returns a numpy array of shape (2,n) for a given iterable of n :obj:`Fraction` objects."""
        return np.array([
            (frac.numerator, frac.denominator)
            for frac in iterable_of_fractions
        ]).T

    def _get_next_consecutive_integer(self) -> int:
        return next(i for i in itertools.count() if i not in self.dict_of_frac_arrays)

    def add_iterable_of_fractions(
            self,
            iterable_of_fractions: Iterable[Fraction],
            name: Optional[Union[str, int]] = None
    ) -> Union[str, int]:
        """Adds some iterable of :obj:`Fraction` objects that can then be retrieved as divs.
        If you assign a name you can retrieve it under that name, otherwise by the integer corresponding to the
        order in which it was added. Iteration over the object goes only through nameless objects in their adding
        order, meaning that you can assign an integer name that will not be taken into account when iterating
        through the object.
        """
        arr = self.iterable_of_fractions_to_array(iterable_of_fractions)
        return self.add_frac_array(arr, name=name)

    @staticmethod
    def _check_array(
            arr
    ):
        arr = np.asarray(arr)
        assert arr.ndim == 2, f"Expected a 2D numpy array, not {arr.ndim}D"
        assert 2 in arr.shape, f"One of the 2 dimensions needs to have shape 2. Received shape: {arr.shape}"
        if arr.shape[0] == 2:
            return arr
        return arr.T

    def add_frac_array(
            self,
            arr,
            name: Optional[Union[str, int]] = None
    ) -> Union[str, int]:
        """Adds a 2d-array where one axis has shape 2, representing numerators and denominators of
        a sequence of fractions.
        If you assign a name you can retrieve it under that name, otherwise by the integer corresponding to the
        order in which it was added. Iteration over the object goes only through nameless objects in their adding
        order, meaning that you can assign an integer name that will not be taken into account when iterating
        through the object.
        """
        arr = self._check_array(arr)
        if name is None:
            name = self._get_next_consecutive_integer()
        assert isinstance(name, (str, int)), f"Name is expected to be a string or int, not a {type(name)!r}"
        if name in self.dict_of_frac_arrays:
            warnings.warn(f"A sequence for the name {name!r} had already been added. It was overwritten.")
        self.dict_of_frac_arrays[name] = arr
        return name

    def add_iterable_or_array(
            self,
            iterable_or_array: Iterable[Fraction],
            name: Optional[Union[str, int]] = None
    ):
        """Convenience function for calling either .add_iterable_of_fractions() or .add_frac_array() based on the input.
        """
        if isinstance(iterable_or_array, np.ndarray):
            return self.add_frac_array(iterable_or_array, name)
        return self.add_iterable_of_fractions(iterable_or_array, name)

    def concatenated_frac_arrays(
            self,
            names: Optional[Union[str,  int, Iterable[Union[str, int]]]] = None
    ):
        """Concatenate the requested arrays in order to compute their LCM. All arrays have shape (2, n) and so does
        their concatenation ("horizontal stacking").
        """
        if names:
            names = self._names_to_tuple(names)
            arrays = tuple(self.dict_of_frac_arrays[name] for name in names)
        else:
            if len(self.dict_of_frac_arrays) == 0:
                raise ValueError(
                    f"No data has been added to this object. "
                    f"Use the method .add_iterable_or_array() first"
                    )
            arrays = tuple(self.dict_of_frac_arrays.values())
        if len(arrays) == 1:
            return arrays[0]
        return np.hstack(arrays)

    def get_divs(
            self,
            name: Union[str, int]
    ):
        """Retrieve one of the previous inputs as divs, based on the LCM computed for all inputs together.
        Name can be a number for retrieving nameless inputs based on their input order.
        """
        if name not in self.dict_of_frac_arrays:
            raise KeyError(name)
        numerators, denominators = self.dict_of_frac_arrays[name]
        lcm = self.least_common_multiple()
        return (numerators * lcm / denominators).astype(int)

    @lru_cache
    def _least_common_multiple(
            self,
            names: Tuple[Union[str, int]]
    ) -> int:
        _, denominators = self.concatenated_frac_arrays(names)
        return np.lcm.reduce(denominators)

    def least_common_multiple(
            self,
            names: Optional[Union[Union[str, int], Iterable[Union[str, int]]]] = None
    ) -> int:
        """By default, the LCM is computed based on all sequences of fractions that this object holds.
        When you retrieve divs, they are always commensurate between all sequences."""
        names = self._names_to_tuple(names)
        return self._least_common_multiple(names)

    @property
    def lcm(self):
        """For convenience."""
        return self.least_common_multiple()

    def _names_to_tuple(
            self,
            names: Optional[Union[Union[str, int], Iterable[Union[str, int]]]] = None
    ) -> Tuple[Union[str, int]]:
        """Process input arguments."""
        if not names:
            names = tuple(self.dict_of_frac_arrays.keys())
        elif isinstance(names, (str, int)):
            names = (names,)
        else:
            names = tuple(names)
        assert len(names) > 0, f"Cannot compute LCM from the names {names!r}"
        return names


    def __getitem__(
            self,
            names: Union[Union[str, int], Iterable[Union[str, int]]]
    ):
        if isinstance(names, (str, int)):
            return self.get_divs(names)
        names = tuple(names)
        return tuple(self.get_divs(name) for name in names)

    def __iter__(self):
        existing_consecutive_integers = itertools.takewhile(lambda x: x in self.dict_of_frac_arrays, itertools.count())
        for i in existing_consecutive_integers:
            yield self.get_divs(i)

# endregion DivMaker

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

def extend_joint_df(jointdf: pd.DataFrame) -> pd.DataFrame:
    df_without_missing = jointdf[jointdf.s_offset_frac.notna()]
    div_maker = DivMaker(
        onsets=df_without_missing.s_offset_frac,
        durations=df_without_missing.s_duration_frac
    )
    onset_div, duration_div = div_maker[("onsets", "durations")]
    div_columns = pd.DataFrame(
        dict(
            onset_div=onset_div,
            duration_div=duration_div
        ),
        index=df_without_missing.index
    )
    return pd.concat([
        div_columns.reindex(jointdf.index),
        jointdf
    ], axis=1)

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
    jointdf.index = j_offset # the index will be reset later but index-sorting is better here than value-sorting
    jointdf = jointdf.sort_index().reset_index() # anyway, j_offset is not suitable as index because it's non-unique
    # forward-fill annotation label features only, do not fill note features for label onsets
    jointdf.loc[:, adf.columns] = jointdf.loc[:, adf.columns].ffill()
    jointdf = jointdf.drop(columns=["s_offset", "a_offset"])
    jointdf = extend_joint_df(jointdf)

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
