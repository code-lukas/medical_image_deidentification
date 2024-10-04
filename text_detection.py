import pydicom
import cv2 as cv
import numpy as np
import os
import pytesseract
import cv2
from glob import glob
import nibabel as nib
from PIL import Image


class TextRemoval:
    """
    Class for performing text removal on images.

    Attributes:
        None

    Methods:
        predict: Apply text removal algorithm to an image.
        __call__: Apply text removal to a directory of images.
    """

    def __init__(self):
        pass

    @staticmethod
    def predict(img: np.array) -> np.array:
        """
        Apply text removal algorithm (tesseract) to an image.

        Args:
            img (np.array): Input image as a NumPy array.

        Returns:
            np.array: Image with text removed.
        """

        threshold = 100

        # Insert rectangle in middle of image to ignore this part in the first iteration
        height, width = img.shape[:2]
        left = int(width / 4)
        top = int(height / 4)
        right = int(width * 3 / 4)
        bottom = int(height * 3 / 4)

        img_covered = cv2.rectangle(img.copy(), (left, top), (right, bottom), (255, 255, 255), -1)
        boxes = pytesseract.image_to_boxes(img_covered, output_type=pytesseract.Output.DICT, nice=1)

        for left, bottom, right, top in zip(boxes["left"], boxes["bottom"], boxes["right"], boxes["top"]):
            if right - left < threshold:
                img = cv2.rectangle(img, (left, height - bottom), (right, height - top), (255, 255, 255), -1)

        # Another iteration without the rectangle in the middle of the image
        try:
            boxes = pytesseract.image_to_boxes(img, output_type=pytesseract.Output.DICT, nice=1)

            for left, bottom, right, top in zip(boxes["left"], boxes["bottom"], boxes["right"], boxes["top"]):
                if right - left < threshold:
                    img = cv2.rectangle(img, (left, height - bottom), (right, height - top), (255, 255, 255), -1)
        except:
            pass

        return img

    def __call__(self, directory: str) -> None:
        """
        Apply text removal to a directory of images.

        Args:
            directory (str): Path to the directory containing the images.

        Returns:
            None
        """

        if os.path.isdir(directory):
            files = glob(os.path.join(directory, '**', '*'), recursive=True)
        else:
            files = [directory]
        for filepath in files:
            file_ending = filepath.split('.')[-1].lower()
            match file_ending:
                # nifti
                case 'png' | 'jpg':
                    img = cv.imread(filepath, 0)
                    base_fn = filepath[:-4]
                case 'jpeg':
                    img = cv.imread(filepath, 0)
                    base_fn = filepath[:-5]
                case 'dcm':
                    dcm = pydicom.dcmread(filepath, force=True)
                    img = dcm.pixel_array
                    base_fn = filepath[:-4]
                case 'nii':
                    nifti = nib.load(filepath)
                    img = np.array(Image.fromarray(nifti.get_fdata().squeeze()).convert("RGB"))
                    base_fn = filepath[:-4]
                case 'gz':
                    nifti = nib.load(filepath)
                    img = np.array(Image.fromarray(nifti.get_fdata().squeeze()).convert("RGB"))
                    base_fn = filepath[:-7]
                case _:
                    raise NotImplementedError(
                        f'File ending {file_ending} not compatible, must be .dcm, .png, .jpg or .jpeg')

            img = self.predict(img=img)

            match file_ending:
                # nifti
                case 'png' | 'jpg' | 'jpeg':
                    cv.imwrite(f'{base_fn}_text_removed.png', img)
                case 'dcm':
                    dcm.PixelData = img.tobytes()
                    dcm.save_as(f'{base_fn}_text_removed.png')
                case 'nii':
                    nifti = nib.Nifti1Image(img, nifti.affine)
                    nib.save(nifti, f'{base_fn}_text_removed.nii')
                case 'gz':
                    nifti = nib.Nifti1Image(img, nifti.affine)
                    nib.save(nifti, f'{base_fn}_text_removed.nii.gz')
