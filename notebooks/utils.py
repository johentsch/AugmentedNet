import itertools
import warnings
from fractions import Fraction
from functools import cache
from typing import Iterable, Dict, Optional, Tuple, overload, Literal

import numpy as np
import pandas as pd
from numpy._typing import NDArray


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
            *iterable_or_array: Iterable[Fraction] | NDArray[int],
            **named_iterables_or_arrays: Iterable[Fraction] | NDArray[int]
    ):
        """Pass one or several 2d-arrays (where one axis has shape 2) or one or several iterables of fractions.
        By passing keyword arguments you can assign names which you can use to retrieve the respective div sequences.
        """
        self.dict_of_frac_arrays: Dict[int | str, NDArray[int]] = {}
        for ioa in iterable_or_array:
            _ = self.add_iterable_or_array(ioa)
        for name, ioa in named_iterables_or_arrays.items():
            _ = self.add_iterable_or_array(ioa, name)

    @staticmethod
    def iterable_of_fractions_to_array(
            iterable_of_fractions: Iterable[Fraction]
    ) -> NDArray[int]:
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
            name: Optional[str | int] = None
    ) -> int | str:
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
            arr: NDArray
    ) -> NDArray:
        arr = np.asarray(arr)
        assert arr.ndim == 2, f"Expected a 2D numpy array, not {arr.ndim}D"
        assert 2 in arr.shape, f"One of the 2 dimensions needs to have shape 2. Received shape: {arr.shape}"
        if arr.shape[0] == 2:
            return arr
        return arr.T

    def add_frac_array(
            self,
            arr: NDArray[int],
            name: Optional[str | int] = None
    ) -> int | str:
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
            iterable_or_array: Iterable[Fraction] | NDArray[int],
            name: Optional[str | int] = None
    ):
        """Convenience function for calling either .add_iterable_of_fractions() or .add_frac_array() based on the input.
        """
        if isinstance(iterable_or_array, np.ndarray):
            return self.add_frac_array(iterable_or_array, name)
        return self.add_iterable_of_fractions(iterable_or_array, name)

    def concatenated_frac_arrays(
            self,
            names: Optional[str | int | Iterable[str | int]] = None
    ) -> NDArray:
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
            name: str | int
    ) -> NDArray[int]:
        """Retrieve one of the previous inputs as divs, based on the LCM computed for all inputs together.
        Name can be a number for retrieving nameless inputs based on their input order.
        """
        if name not in self.dict_of_frac_arrays:
            raise KeyError(name)
        numerators, denominators = self.dict_of_frac_arrays[name]
        lcm = self.least_common_multiple()
        return (numerators * lcm / denominators).astype(int)

    @cache
    def _least_common_multiple(
            self,
            names: Tuple[str | int]
    ) -> int:
        _, denominators = self.concatenated_frac_arrays(names)
        return np.lcm.reduce(denominators)

    def least_common_multiple(
            self,
            names: Optional[str | int | Iterable[str | int]] = None
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
            names: Optional[str | int | Iterable[str | int]] = None
    ) -> Tuple[str | int]:
        """Process input arguments."""
        if not names:
            names = tuple(self.dict_of_frac_arrays.keys())
        elif isinstance(names, (str, int)):
            names = (names,)
        else:
            names = tuple(names)
        assert len(names) > 0, f"Cannot compute LCM from the names {names!r}"
        return names

    @overload
    def __getitem__(
            self,
            names: str | int
    ) -> NDArray:
        ...

    @overload
    def __getitem__(
            self,
            names: Iterable[str | int]
    ) -> Tuple[NDArray]:
        ...

    def __getitem__(
            self,
            names: str | int | Iterable[str | int]
    ) -> NDArray | Tuple[NDArray]:
        if isinstance(names, (str, int)):
            return self.get_divs(names)
        names = tuple(names)
        return tuple(self.get_divs(name) for name in names)

    def __iter__(self):
        existing_consecutive_integers = itertools.takewhile(lambda x: x in self.dict_of_frac_arrays, itertools.count())
        for i in existing_consecutive_integers:
            yield self.get_divs(i)


def _ts_beat_size(numerator: int, denominator: int) -> Fraction:
    beat_numerator = 3 if numerator % 3 == 0 and numerator > 3 else 1
    result = Fraction(beat_numerator, denominator)
    return result


@cache
def ts_beat_size(ts: str) -> Fraction:
    """ Pass a time signature to get the beat size which is based on the fraction's
        denominator ('2/2' => 1/2, '4/4' => 1/4, '4/8' => 1/8). If the nominator is
        a higher multiple of 3, the threefold beat size is returned
        ('12/8' => 3/8, '6/4' => 3/4).
    """
    numerator, denominator = str(ts).split('/')
    result = _ts_beat_size(int(numerator), int(denominator))
    return result


@overload
def onset2beat(
        onset: Fraction,
        timesig: str,
        round_to: Literal[None]
) -> Fraction:
    ...


@overload
def onset2beat(
        onset: Fraction,
        timesig: str,
        round_to: int
) -> float:
    ...


@cache
def onset2beat(
        onset: Fraction,
        timesig: str,
        round_to: Optional[int] = None
) -> float | Fraction:
    """ Turn an offset in whole notes into a beat based on the time signature.
        Uses: ts_beat_size()

    Args:
        onset:
            Offset from the measure's beginning as fraction of a whole note.
        timesig:
            Time signature, i.e., a string representing a fraction.
        round_to:

    """
    size = ts_beat_size(timesig)
    beat, remainder = divmod(onset, size)
    subbeat = remainder / size
    result = beat + 1 + subbeat
    return result if round_to is None else round(float(result), round_to)


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
