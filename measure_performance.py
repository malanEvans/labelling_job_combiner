import os
import json
import numpy as np
import h5py
import argparse

from job_combiner import JobCombiner

def compute_performance(h5_file: str, imshape: tuple, classes: list) -> None:
    class_pixels = 0
    overlap_pixels = 0
    h5_obj = h5py.File(h5_file, 'r+')
    for i in range(imshape[0]):
        for j in range(imshape[1]):
            labels = []
            for cls in classes:
                label = h5_obj[cls][i][j]
                if label != -1:
                    labels.append(label)
            if len(labels) != len(set(labels)):
                overlap_pixels += 1
            if labels:
                class_pixels += 1
    
    return (class_pixels - overlap_pixels)/class_pixels * 100.0


if __name__ == "__main__":
    arg_parse = argparse.ArgumentParser(
        description="to show a combined class availability"
    )
    arg_parse.add_argument('--loc_id', dest="loc_id", required=True,
                           help="the location id of the image")
    arg_parse.add_argument('--outpath', dest="outpath", default='output',
                           help="optionally mention the outpath")

    args = arg_parse.parse_args()

    h5_file = os.path.join(
        args.outpath, args.loc_id, 'combined',
        'combined_mask.h5'
    )
    meta_file = os.path.join(
        args.outpath, args.loc_id, 'combined',
        'meta.json'
    )
    if not os.path.exists(h5_file):
        raise FileNotFoundError(
            "the location id has not combined yet. Please combine before viewing"
        )
    
    meta_dict = json.load(open(meta_file))
    classes = meta_dict.get('classes')
    imshape = meta_dict.get('shape')
    if len(classes) < 2:
        raise UserWarning(
            f"there should be atleast two classes to measure performance"
        )
    
    metric = compute_performance(h5_file, imshape, classes)
    print(f"Performance metric: {metric}%")