import pydicom
from PIL import Image
import numpy as np
from pydicom.dataset import Dataset, FileDataset


def dicom_load(filename="data/test.dcm"):
    ds = pydicom.dcmread(filename)

    if "PatientName" in ds:
        print(f"PatientName: {ds.PatientName}")
    if "ImageComments" in ds:
        print(f"ImageComments: {ds.ImageComments}")
    if "StudyDate" in ds:
        print(f"StudyDate: {ds.StudyDate}")

    arr = ds.pixel_array
    img = Image.fromarray(np.uint8(arr * 255))
    return np.asarray(img, dtype="uint8")


def dicom_save(img, filename, patient, comments, date):
    # file_meta = Dataset()
    # ds = FileDataset(filename, {}, file_meta=file_meta, preamble="\0" * 128)
    ds = pydicom.dcmread(filename)
    ds.PatientName = patient
    ds.ImageComments = comments
    ds.StudyDate = date

    ds.PixelData = img.tobytes()
    ds.Rows, ds.Columns = img.shape
    ds.save_as(filename)
