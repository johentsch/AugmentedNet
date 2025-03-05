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
import itertools
import os
from fractions import Fraction
from functools import cache
from typing import Iterable, Dict, Optional, overload

import ms3
import numpy as np
import pandas as pd
from numpy._typing import NDArray

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
merged.head()

# %%
from typing import Tuple


class DivMaker():
    """This is a convenient object for turning sequences of fractions into commensurate divs.
    It is equivalent to concatenating all sequences, passing them to the function shown below, and splitting them again.
    
        def fractions2divs(fracs: Iterable[Fraction]) -> NDArray[int]:
            numerators, denominators = np.array([(item.numerator,item.denominator) for item in fracs]).T
            lcm = np.lcm.reduce(denominators) # least common multiple
            return (numerators * lcm / denominators).astype(int)
    """
    
    dict_of_frac_arrays: Dict[int | str, NDArray[int]] = {}
    
    def __init__(
            self,
            *iterable_of_fractions: Iterable[Fraction],
            **named_iterables: Iterable[Fraction]
    ):
        for iof in iterable_of_fractions:
            self.add_iterable_of_fractions(iof)
        for name, iof in named_iterables.items():
            self.add_iterable_of_fractions(iof, name)
        
    @staticmethod    
    def iterable_of_fractions_to_array(
            iterable_of_fractions: Iterable[Fraction]
    ) -> NDArray[int]:
        """Returns a numpy array of shape (2,n) for a given iterable of n :obj:`Fraction` objects."""
        return np.array([(frac.numerator, frac.denominator) for frac in iterable_of_fractions]).T
        
            
    def add_iterable_of_fractions(
            self, 
            iterable_of_fractions: Iterable[Fraction],
            name: Optional[str|int] = None
    ) -> None:
        if name is None:
            name = next(i for i in itertools.count() if i not in self.dict_of_frac_arrays)
        assert isinstance(name, (str, int)), f"Name is expected to be a string or int, not a {type(name)!r}"
        self.dict_of_frac_arrays[name] = self.iterable_of_fractions_to_array(iterable_of_fractions)
        
    def concatenated_frac_arrays(
            self, 
            names: Optional[str | int | Iterable[str | int]] = None
    ) -> NDArray:
        if names:
            names = self._names_to_tuple(names)
            arrays = tuple(self.dict_of_frac_arrays[name] for name in names)
        else:
            if len(self.dict_of_frac_arrays) == 0:
                raise ValueError(f"No data has been added to this object. "
                                 f"Use the method .add_iterable_of_fractions() first")
            arrays = tuple(self.dict_of_frac_arrays.values())
        if len(arrays) == 1:
            return arrays[0]
        return np.hstack(arrays)
    
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
        """By default, the LCM is computed based on all sequences of fractions that his object holds. 
        When you retrieve divs, they are always commensurate between all sequences."""
        names = self._names_to_tuple(names)
        return self._least_common_multiple(names)


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

    def get_divs(self, name: str | int) -> NDArray[int]:
        if name not in self.dict_of_frac_arrays:
            raise KeyError(name)
        numerators, denominators = self.dict_of_frac_arrays[name]
        lcm = self.least_common_multiple()
        return (numerators * lcm / denominators).astype(int)
    
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
        
            
        
            
        
        
        
        


def make_pitch_array(merged: pd.DataFrame) -> pd.DataFrame:
    keep_original_columns = ["mc", "mn", "mc_playthrough", "mn_playthrough", "quarterbeats_playthrough", "duration"]
    result = merged[keep_original_columns]
    div_maker = DivMaker(
        onsets = merged.quarterbeats_playthrough, 
        durations = merged.duration * 4
    )
    onset_div, duration_div = div_maker[("onsets", "durations")]
    result = pd.concat([
        pd.DataFrame(dict(
            onset_div = onset_div,
            duration_div = duration_div
        )),
        result
    ], axis=1)
    return result

make_pitch_array(merged)

# %%
