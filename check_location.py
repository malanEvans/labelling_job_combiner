import os
import json
import numpy as np
from PIL import Image
import h5py
import argparse

from job_combiner import JobCombiner

def show_image(h5_file: str, imclass: str) -> None:
    h5_obj = h5py.File(h5_file, 'r+')
    imarray = np.array(h5_obj[imclass])
    imarray[imarray > 0] = 255
    imarray[imarray < 0] = 100
    im = Image.fromarray(imarray.astype('uint8'))
    im.show()

if __name__ == "__main__":
    arg_parse = argparse.ArgumentParser(
        description="to show a combined class availability"
    )
    arg_parse.add_argument('--loc_id', dest="loc_id", required=True,
                           help="the location id of the image")
    arg_parse.add_argument('--class', dest="imclass", required=True,
                           help="the class of the image that need to be shown")
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
    if args.imclass not in classes:
        raise UserWarning(
            f"the specified class doesnot exist, please specify from {classes}"
        )
    
    show_image(h5_file, args.imclass)

