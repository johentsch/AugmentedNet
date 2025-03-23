"""Combine all available (score, annotation) pairs into tsv files."""

import os
import shutil
from pathlib import Path
from typing import Optional

import pandas as pd

from . import cli
from .common import (
    ANNOTATIONSCOREDUPLES,
    DATASETSUMMARYFILE,
    DATASPLITS,
)
from .joint_parser import (
    parseAnnotationAndAnnotation,
    parseAnnotationAndScore,
    parseAnnotationAndScoreEvents,
)


def generateDataset(
    synthesize=False, texturize=False, tsvDir="dataset", eventBased=False
):
    statsdict = {
        "file": [],
        "annotation": [],
        "score": [],
        "collection": [],
        "split": [],
        "misalignmentMean": [],
        "qualityMean": [],
        "incongruentBassMean": [],
    }
    datasetDir = f"{tsvDir}-synth" if synthesize else tsvDir
    Path(datasetDir).mkdir(exist_ok=True)
    for split, files in DATASPLITS.items():
        Path(os.path.join(datasetDir, split)).mkdir(exist_ok=True)
        for nickname in files:
            print(nickname)
            annotation, score = ANNOTATIONSCOREDUPLES[nickname]
            if not synthesize:
                df = parseAnnotationAndScore(annotation, score, eventBased=eventBased)
            else:
                df = parseAnnotationAndAnnotation(annotation, texturize=texturize)
            outpath = os.path.join(datasetDir, split, nickname + ".tsv")
            df.to_csv(outpath, sep="\t")
            collection = nickname.split("-")[0]
            statsdict["file"].append(nickname)
            statsdict["annotation"].append(annotation)
            statsdict["score"].append(score)
            statsdict["collection"].append(collection)
            statsdict["split"].append(split)
            misalignment = round(df.measureMisalignment.mean(), 2)
            statsdict["misalignmentMean"].append(misalignment)
            qualitySquaredSum = round(df.qualitySquaredSum.mean(), 2)
            statsdict["qualityMean"].append(qualitySquaredSum)
            incongruentBass = round(df.incongruentBass.mean(), 2)
            statsdict["incongruentBassMean"].append(incongruentBass)
            df = pd.DataFrame(statsdict)
            df.to_csv(os.path.join(datasetDir, DATASETSUMMARYFILE), sep="\t")
    return df


def store_labeled_pitch_array_and_label_tsv(
    nickname: str,
    score_path: str,
    annotation_path: str,
    datasetDir: str,
    split: str,
    assembled_dir: Optional[str] = None,
    include_metadata: bool = True,
):
    v100_processing = split == "test"
    extended_adf, sdf, jointdf, metadata = parseAnnotationAndScoreEvents(
        annotation_path, score_path, v100_processing=v100_processing
    )
    for df, suffix in [(jointdf, "joint")]:  # , (sdf, "slices")]:
        outpath = os.path.join(datasetDir, split, f"{nickname}_{suffix}.tsv")
        df.to_csv(outpath, sep="\t", index=False)
    if assembled_dir:
        outpath = os.path.join(assembled_dir, "labels", f"{nickname}.tsv")
        extended_adf.to_csv(outpath, sep="\t", index=False)
    # copy and rename original score
    _, score_ext = os.path.splitext(score_path)
    new_score_path = os.path.join(datasetDir, split, f"{nickname}{score_ext}")
    shutil.copy(score_path, new_score_path)
    collection = nickname.split("-")[0]
    stats = dict(
        file=nickname,
        annotation=annotation_path,
        score=score_path,
        collection=collection,
        split=split,
    )
    if include_metadata:
        stats.update(metadata)
    return stats


def generateEventsDataset(
    tsvDir="events", assembled_dir="assembled", include_metadata=True
):
    statsrecords = []
    datasetDir = tsvDir
    Path(datasetDir).mkdir(exist_ok=True)
    for split, files in DATASPLITS.items():
        Path(os.path.join(datasetDir, split)).mkdir(exist_ok=True)
        for nickname in files:
            print(nickname)
            annotation, score = ANNOTATIONSCOREDUPLES[nickname]
            stats = store_labeled_pitch_array_and_label_tsv(
                nickname,
                score,
                annotation,
                datasetDir,
                split,
                assembled_dir,
                include_metadata,
            )
            statsrecords.append(stats)
            # misalignment = jointdf.measureMisalignment.mean().round(2)
            # statsdict["misalignmentMean"].append(misalignment)
            # qualitySquaredSum = jointdf.qualitySquaredSum.mean().round(2)
            # statsdict["qualityMean"].append(qualitySquaredSum)
            # incongruentBass = jointdf.incongruentBass.mean().round(2)
            # statsdict["incongruentBassMean"].append(incongruentBass)
            jointdf = pd.DataFrame.from_records(statsrecords)
            jointdf.to_csv(os.path.join(datasetDir, DATASETSUMMARYFILE), sep="\t")
    return jointdf


if __name__ == "__main__":
    parser = cli.tsv()
    args = parser.parse_args()
    kwargs = vars(args)
    # generateDataset(**kwargs)
    generateEventsDataset()
