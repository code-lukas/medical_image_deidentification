import os
from glob import glob
from tqdm import tqdm
from pathlib import Path
import time
import numpy as np

input_path = "/home/jovyan/deface/data/skullstrip/synthstrip_test/input"
output_path = "/home/jovyan/deface/data/skullstrip/synthstrip_test/pred_mrisynthstrip"


os.system("export FREESURFER_HOME=/home/jovyan/deface/freesurfer")
os.system("source $FREESURFER_HOME/SetUpFreeSurfer.sh")

input_files = glob(input_path + "/*.nii.gz")

times = []

for file in tqdm(input_files):
    start_time = time.time()
    os.system("mri_synthstrip -i " + file + " -o " + os.path.join(output_path, Path(file).name) + " -g")
    end_time = time.time()
    times.append(round((end_time - start_time), 3))
    
print("Execution time:", np.sum(times), "seconds")
print(times)