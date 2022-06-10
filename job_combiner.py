import os
from glob import glob
import json
import h5py
from pyspark.sql import SparkSession
import tarfile

COMBINED_MASK = 'combined_mask.h5'
META_FILE = 'meta.json'

class JobCombiner():
    """
    class to implement behaviours of job combiner
    """
    def __init__(self, loc_id: int, date: str,
                 path: str, classes: list, outpath: str = None) -> None:
        if outpath:
            self._combined_outpath = os.path.join(
                outpath, str(loc_id), 'combined'
            )
        else:
            self._combined_outpath = os.path.join(
                path, str(loc_id), 'combined'
            )
        self._jobs_path_fmt = os.path.join(
            path, str(loc_id), '_'.join((
                '*', str(loc_id), date
            )) + '.tar.gz'
        )

        self._cur_combined = None
        self._classes = set(map(str, classes))
        self._cur_jobs = 0
        self._imshape = None
        print(f"Combining {loc_id} for {date} with {classes}")

    def _modify_mask_val(self, val: int) -> int:
        """
        modify mask value to have -1 when class is absent,
        1 when class is present and 0 when indefined.
        when this is done, it's easier to combine and find the majority
        label
        """
        if val < 0:
            return 0
        elif val == 0:
            return -1
        return 1


    def _read_mask(self, h5_file: str) -> list:
        """
        read label for each class in a mask
        return a list of tuples with ((class, pixel_location), label)
        """
        h5_mask = h5py.File(h5_file, 'r+')
        mask_vals = []
        for cls in h5_mask.keys():
            try:
                n, m = h5_mask[cls].shape
                for i in range(n):
                    for j in range(m):
                        mask_val = h5_mask[cls][i][j]
                        mask_val = self._modify_mask_val(mask_val)
                        mask_vals.append(
                            ((cls, i, j), mask_val))
            except:
                continue
        return mask_vals

    def _load_cur_combined(self) -> None:
        """
        load meta data if exists and set combined mask path
        """
        if not os.path.exists(self._combined_outpath):
            os.makedirs(self._combined_outpath)

        self._cur_combined = os.path.join(
            self._combined_outpath,
            COMBINED_MASK
        )

        meta_file = os.path.join(self._combined_outpath,
                                 META_FILE)
        if os.path.exists(meta_file):
            meta_dict = json.load(open(meta_file))
            self._cur_jobs = meta_dict.get('jobs', 0)
            self._classes.update(set(meta_dict.get('classes', [])))
            self._imshape = meta_dict.get('shape')
    
    def _get_combined_label(self, cls_pixel_sums: tuple) -> tuple:
        """
        accept a sum of transformed labels as a tuple
        ((classs, pixel_location), label_sum)
        and adjust the label in the following manner:
            if label_sum > 0 then label = 1
            elif label_sum == 0 then label = -1
            else label = 0
        then return ((classs, pixel_location), label)
        in this way, if there are equal number of presents and absents,
        it will be taken as undefined
        """
        cls_pixel, label_sum = cls_pixel_sums
        if label_sum < 0:
            return [(cls_pixel, 0)]
        elif label_sum == 0:
            return [(cls_pixel, -1)]
        return [(cls_pixel, 1)]

    def _get_hdf5_object(self, val_tuples: list, h5_file: str) -> None:
        """
        when ((class, pixel_location), label) like tuple list is given,
        this method creates a HDF5 file which is the combined mask file
        """
        hf = h5py.File(h5_file, 'w')
        for cls in self._classes:
            hf.create_dataset(cls, (self._imshape[0], self._imshape[1]),
                              dtype='i8')
        
        for cls_pixel, val in val_tuples:
            cls, i, j = cls_pixel
            hf[cls][i, j] = val

        hf.close()

    def _extract_and_save(self, job_tar: str) -> str:
        """
        this method read a tar file related to a job and 
        dump a temporary h5 file
        """
        tar = tarfile.open(job_tar, 'r:gz')
        for _f in tar.getnames():
            if not _f.endswith('.h5'):
                continue
            tar_member = tar.getmember(_f)
            h5_obj = tar.extractfile(tar_member)
            temp_file = os.path.basename(_f) + '.temp'

            if h5_obj:
                with open(temp_file, 'wb') as fp:
                    fp.write(h5_obj.read())
                return temp_file
                break
        return None

    def _write_meta_data(self) -> None:
        """
        write meta data
        """
        meta_file = os.path.join(self._combined_outpath,
                                 META_FILE)
        with open(meta_file, 'w') as fp:
            json.dump({'jobs': self._cur_jobs,
                       'classes': list(self._classes),
                       'shape': self._imshape,
                       'jobs': self._cur_jobs},
                      fp, indent=4)
    
    def combine_jobs(self) -> None:
        """
        combine job main method
        """
        # load meta data if available
        self._load_cur_combined()

        # get all job tar files for current location and date
        job_tar_files = glob(self._jobs_path_fmt)
        job_files = []

        # create temp h5 file for job tar file and have temp h5 files as a list
        for job_tar in job_tar_files:
            job_mask_h5 = self._extract_and_save(job_tar)
            # set image shape if not found otherwise
            if job_mask_h5:
                if self._imshape is None:
                    h5_mask = h5py.File(job_mask_h5, 'r+')
                    for cls in h5_mask.keys():
                        self._imshape = h5_mask[cls].shape
                        del h5_mask
                        break
                job_files.append(job_mask_h5)
                self._cur_jobs += 1
        
        # append the current combined mask if available as h5 file
        if os.path.exists(self._cur_combined):
            job_files.append(self._cur_combined)
        
        # create sparksession and get the sparkcontext object
        spark = SparkSession.builder.master('local[1]').appName(
            'JobCombiner').getOrCreate()
        sc = spark.sparkContext

        # create an rdd with list of temp h5 files
        rdd = sc.parallelize(job_files)
        # map h5 files into a list of tuples for each class and pixel value location
        # with label
        rdd = rdd.flatMap(self._read_mask)
        # get the summation of list of tuples according to key which class and pixel value
        # location
        rdd = rdd.reduceByKey(lambda a,b:a+b)
        # map sum to the label
        rdd = rdd.flatMap(self._get_combined_label)
        combined_mask_vals = rdd.collect()
        
        # write combined h5 file
        self._get_hdf5_object(combined_mask_vals, self._cur_combined)

        # remove temp files
        os.system('rm -rf *.temp')

        #write meta data
        self._write_meta_data()