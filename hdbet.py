import os
from glob import glob
from tqdm import tqdm
from pathlib import Path
import time
import numpy as np

input_path = "/home/jovyan/deface/data/skullstrip/synthstrip_test/input"
output_path = "/home/jovyan/deface/data/skullstrip/synthstrip_test/pred_hdbet"


input_files = glob(input_path + "/*.nii.gz")

times = []

for file in tqdm(input_files):
    start_time = time.time()
    os.system("hd-bet -i " + file + " -o " + os.path.join(output_path, Path(file).name))
    end_time = time.time()
    times.append(round((end_time - start_time), 3))
    
print("Execution time:", np.sum(times), "seconds")
print(times)