# labelling_job_combiner
this is part of a test given by an employer

it can combine multiple labelling jobs to get the combined mask. In addition to that, it can show the image for a unique class and it can return performance mettric for each location.

following libraries and packages are used:
  PIL - to show image
  numpy - array formation
  h5py - read h5 files
  pyspark - map reduce
  tarfile - read tat.gz files
  
 Usage:
  if there's a .csv file with location_id, date and classes use;
      python driver.py <csv_file>
  
  to see image for a unique class for a location id
      python check_location.py --loc_id <location_id> --class <class>
  
  to get performance metric for a location
      python measure_performance.py --loc_id <location_id>
  
  Assumptions:
  =============
  
  * when there are an equal number of absents and presents it should be an undefined pixel
  * performance metric is measured as the number of class overlaps over number of pixels with a class
  
  
  Improvements:
  =============
  This solution is scalable and it will be better to use a spark cluster to run this. In that manner the performance could be improved.
  
  If more time and resources are given, the solution can have multiprocessing and distributed computing to improve the efficiency.
