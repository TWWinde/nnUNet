import multiprocessing
import shutil
from multiprocessing import Pool

from batchgenerators.utilities.file_and_folder_operations import *

from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw
from skimage import io
from acvl_utils.morphology.morphology_helper import generic_filter_components
from scipy.ndimage import binary_fill_holes
from PIL import Image
import numpy as np
import cv2
import nibabel as nib


def load_and_covnert_case(input_image: str, input_seg: str, output_image: str, output_seg: str,
                          min_component_size: int = 50):
    seg = io.imread(input_seg)
    # seg[seg == 255] = 1
    #image = io.imread(input_image)
    #image = image.sum(2)
    #mask = image == (3 * 255)
    # the dataset has large white areas in which road segmentations can exist but no image information is available.
    # Remove the road label in these areas
    #mask = generic_filter_components(mask, filter_fn=lambda ids, sizes: [i for j, i in enumerate(ids) if
                                                                        # sizes[j] > min_component_size])
    #mask = binary_fill_holes(mask)
    #seg[mask] = 0
    io.imsave(output_seg, seg, check_contrast=False)
    shutil.copy(input_image, output_image)


def remove_background(image):
    # Apply Gaussian blur to the image (optional)
    binary_image = np.where(image > 45, 1, 0)
    binary_image = binary_image.astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=4)

    result_image = np.zeros_like(binary_image)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area > 1000:
            result_image[labels == label] = 255
    result_image = result_image.astype(np.uint8)
    result_image = cv2.multiply(image, result_image * 255)

    return result_image


def get_2d_images(ct_path, label_path):
    n = 0
    for i in range(int(len(ct_path) * 0.9)):
        nifti_img = nib.load(ct_path[i])
        img_3d = nifti_img.get_fdata()
        nifti_seg = nib.load(label_path[i])
        seg_3d = nifti_seg.get_fdata()

        for z in range(seg_3d.shape[2]):
            seg_slice = seg_3d[:, :, z]
            seg_slice = seg_slice[72:328, 65:321]
            seg_slice = seg_slice.astype(np.uint8)
            img_slice = img_3d[:, :, z]
            img_slice = img_slice[72:328, 65:321]

            if img_slice.max() != img_slice.min() and seg_slice.max() != seg_slice.min():
                image = (((img_slice - img_slice.min()) / (img_slice.max() - img_slice.min())) * 255).astype(np.uint8)
                image = remove_background(image)
                image = Image.fromarray(image)
                image = image.convert('RGB')
                image.save(f'/misc/data/private/autoPET/data_nnunet/train/images/slice_{n}.png')
                cv2.imwrite(f'/misc/data/private/autoPET/data_nnunet/train/labels/slice_{n}.png', seg_slice)
                n += 1

    print("finished train data set")
    n = 0
    for j in range(int(len(ct_path) * 0.9), int(len(ct_path) * 0.95)):
        nifti_img = nib.load(ct_path[j])
        img_3d = nifti_img.get_fdata()
        nifti_seg = nib.load(label_path[j])
        seg_3d = nifti_seg.get_fdata()

        for z in range(seg_3d.shape[2]):
            seg_slice = seg_3d[:, :, z]
            seg_slice = seg_slice[72:328, 65:321]
            seg_slice = seg_slice.astype(np.uint8)
            img_slice = img_3d[:, :, z]
            img_slice = img_slice[72:328, 65:321]

            if img_slice.max() != img_slice.min() and seg_slice.max() != seg_slice.min():
                image = (((img_slice - img_slice.min()) / (img_slice.max() - img_slice.min())) * 255).astype(np.uint8)
                image = remove_background(image)
                image = Image.fromarray(image)
                image = image.convert('RGB')
                image.save(f'/misc/data/private/autoPET/data_nnunet/test/images/slice_{n}.png')
                cv2.imwrite(f'/misc/data/private/autoPET/data_nnunet/test/labels/slice_{n}.png', seg_slice)
                n += 1

    print("finished test data set")
    n = 0
    for k in range(int(len(ct_path) * 0.95), len(ct_path)):
        nifti_img = nib.load(ct_path[k])
        img_3d = nifti_img.get_fdata()
        nifti_seg = nib.load(label_path[k])
        seg_3d = nifti_seg.get_fdata()

        for z in range(seg_3d.shape[2]):
            seg_slice = seg_3d[:, :, z]
            seg_slice = seg_slice[72:328, 65:321]
            seg_slice = seg_slice.astype(np.uint8)
            img_slice = img_3d[:, :, z]
            img_slice = img_slice[72:328, 65:321]

            if img_slice.max() != img_slice.min() and seg_slice.max() != seg_slice.min():
                image = (((img_slice - img_slice.min()) / (img_slice.max() - img_slice.min())) * 255).astype(np.uint8)
                image = remove_background(image)
                image = Image.fromarray(image)
                image = image.convert('RGB')
                image.save(f'/misc/data/private/autoPET/data_nnunet/val/images/slice_{n}.png')
                cv2.imwrite(f'/misc/data/private/autoPET/data_nnunet/val/labels/slice_{n}.png', seg_slice)
                n += 1

    print("finished validation data set")

def list_images(path):
    ct_path = []
    label_path = []
    # read autoPET files names
    names = os.listdir(path)
    ct_names = list(filter(lambda x: x.endswith('0001.nii.gz'), names))

    for i in range(len(ct_names)):
        ct_path.append(os.path.join(path, ct_names[i]))
        label_path.append(os.path.join(path, ct_names[i].replace('0001.nii.gz', '0002.nii.gz')))

    return ct_path, label_path


if __name__ == "__main__":
    os.makedirs('/misc/data/private/autoPET/data_nnunet/train/images', exist_ok=True)
    os.makedirs('/misc/data/private/autoPET/data_nnunet/train/labels', exist_ok=True)
    os.makedirs('/misc/data/private/autoPET/data_nnunet/test/images', exist_ok=True)
    os.makedirs('/misc/data/private/autoPET/data_nnunet/test/labels', exist_ok=True)
    os.makedirs('/misc/data/private/autoPET/data_nnunet/val/images', exist_ok=True)
    os.makedirs('/misc/data/private/autoPET/data_nnunet/val/labels', exist_ok=True)
    path_imagesTr = "/misc/data/private/autoPET/imagesTr"
    ct_paths, label_paths = list_images(path_imagesTr)
    get_2d_images(ct_paths, label_paths)
    print('data prepare finished')
    # extracted archive from https://www.kaggle.com/datasets/insaff/massachusetts-roads-dataset?resource=download
    source = '/misc/data/private/autoPET/data_nnunet'

    dataset_name = 'Dataset522_body'

    imagestr = join(nnUNet_raw, dataset_name, 'imagesTr')
    imagests = join(nnUNet_raw, dataset_name, 'imagesTs')
    labelstr = join(nnUNet_raw, dataset_name, 'labelsTr')
    labelsts = join(nnUNet_raw, dataset_name, 'labelsTs')
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)
    maybe_mkdir_p(labelsts)

    train_source = join(source, 'train')
    test_source = join(source, 'test')

    with multiprocessing.get_context("spawn").Pool(8) as p:

        # not all training images have a segmentation
        valid_ids = subfiles(join(train_source, 'labels'), join=False, suffix='png')
        num_train = len(valid_ids)
        r = []
        for v in valid_ids:
            r.append(
                p.starmap_async(
                    load_and_covnert_case,
                    ((
                         join(train_source, 'images', v),
                         join(train_source, 'labels', v),
                         join(imagestr, v[:-4] + '_0000.png'),
                         join(labelstr, v),
                         50
                     ),)
                )
            )

        # test set
        valid_ids = subfiles(join(test_source, 'labels'), join=False, suffix='png')
        for v in valid_ids:
            r.append(
                p.starmap_async(
                    load_and_covnert_case,
                    ((
                         join(test_source, 'images', v),
                         join(test_source, 'labels', v),
                         join(imagests, v[:-4] + '_0000.png'),
                         join(labelsts, v),
                         50
                     ),)
                )
            )
        _ = [i.get() for i in r]

    generate_dataset_json(join(nnUNet_raw, dataset_name), {0: 'R', 1: 'G', 2: 'B'}, {'background': 0, '1': 1,
                                                                                     '2': 2, '3': 3,
                                                                                     '4': 4, '5': 5,
                                                                                     '6': 6, '7': 7,
                                                                                     '8': 8, '9': 9,
                                                                                     '10': 10, '11': 11,
                                                                                     '12': 12, '13': 13,
                                                                                     '14': 14, '15': 15,
                                                                                     '16': 16, '17': 17,
                                                                                     '18': 18, '19': 19,
                                                                                     '20': 20, '21': 21,
                                                                                     '22': 22, '23': 23,
                                                                                     '24': 24, '25': 25,
                                                                                     '26': 26, '27': 27,
                                                                                     '28': 28, '29': 29,
                                                                                     '30': 30, '31': 31,
                                                                                     '32': 32, '33': 33,
                                                                                     '34': 34, '35': 35,
                                                                                     '36': 36 }, num_train, '.png', dataset_name=dataset_name)
