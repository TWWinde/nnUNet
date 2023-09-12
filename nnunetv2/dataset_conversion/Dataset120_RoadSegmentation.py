import multiprocessing
import shutil
from multiprocessing import Pool

from batchgenerators.utilities.file_and_folder_operations import *

from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw
from skimage import io
from acvl_utils.morphology.morphology_helper import generic_filter_components
from scipy.ndimage import binary_fill_holes


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


if __name__ == "__main__":
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
