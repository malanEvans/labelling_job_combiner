import sys
import pandas as pd
from job_combiner import JobCombiner

csv_file = sys.argv[1]

df = pd.read_csv(csv_file)

for i, row in df.iterrows():
    if i < 110:
        continue
    loc_id = int(row.loc_id)
    date = row.date
    classes = list(map(int, row.classes.strip('{}').split(',')))

    jc = JobCombiner(loc_id, date, 'New_Data', classes, 'output')
    jc.combine_jobs()