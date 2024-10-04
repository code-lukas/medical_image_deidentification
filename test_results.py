import torch
import nibabel as nib
from glob import glob
from tqdm import tqdm
from utils.validation import segmentation_validation

pred_folder = "anonymizer/data/test_label"
label_folder = "anonymizer/data/test_pred"

pred_files = glob(pred_folder + "/*.nii.gz")
label_files = glob(label_folder + "/*.nii.gz")

pred_files.sort()
label_files.sort()

n = 0
dsc = 0
iou = 0

for pred_file, label_file in zip(tqdm(pred_files), label_files):
    pred = nib.load(pred_file).get_fdata()
    label = nib.load(label_file).get_fdata()
    pred = torch.tensor(pred)
    label = torch.tensor(label)
    for slice in range(pred.shape[2]):
        pred_slice = pred[..., slice]
        label_slice = label[..., slice]
        pred_slice = torch.where(pred_slice > 0, 1, 0)
        label_slice = torch.where(label_slice > 0, 1, 0)
        if pred_slice.sum() == label_slice.sum() == 0:
            continue
        scores = segmentation_validation(pred_slice, label_slice)
        n += 1
        dsc += scores["dsc"]
        iou += scores["iou"]

print("DSC: ", dsc / n)
print("IOU: ", iou / n)

