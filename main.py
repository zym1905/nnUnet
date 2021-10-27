import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.dataset_conversion.utils import generate_dataset_json
from nnunet.paths import nnUNet_raw_data, preprocessing_output_dir
from nnunet.utilities.file_conversions import convert_2d_image_to_nifti

# main
if __name__ == '__main__':
    # data directory
    dataDir = '/Users/zhangyong/work/MLIM/data/data-science-bowl-2018/stage1_train'

    # convert to nii
    task_name = 'Task599_CellSeg'
    target_base = join(nnUNet_raw_data, task_name)
    target_imagesTr = join(target_base, "imagesTr")
    target_imagesTs = join(target_base, "imagesTs")
    target_labelsTs = join(target_base, "labelsTs")
    target_labelsTr = join(target_base, "labelsTr")

    maybe_mkdir_p(target_imagesTr)
    maybe_mkdir_p(target_labelsTs)
    maybe_mkdir_p(target_imagesTs)
    maybe_mkdir_p(target_labelsTr)

    count = 0
    train_image_dir = subdirs(dataDir)
    for directory in train_image_dir:
        print(directory)
        imageDir = join(directory, 'images')
        image = subfiles(imageDir, suffix='.png')[0]
        # print(image)
        labelDir = join(directory, 'masks')
        label = subfiles(labelDir, suffix='.png')[0]
        unique_name = os.path.basename(image)
        unique_name = unique_name[:-4]

        output_image_file = ''
        output_label_file = ''
        # train data 470 out of 670
        if count < 470:
            output_image_file = join(target_imagesTr, unique_name)
            output_label_file = join(target_labelsTr, unique_name)
        # test data 200 out of 670
        else:
            output_image_file = join(target_imagesTs, unique_name)
            output_label_file = join(target_labelsTs, unique_name)

        print('image:' + image)
        print('output_image_file:' + output_image_file)
        convert_2d_image_to_nifti(image, output_image_file, is_seg=False)
        convert_2d_image_to_nifti(label, output_label_file, is_seg=True, transform=lambda x: (x == 255).astype(int))
        count += 1

        # finally we can call the utility for generating a dataset.json
        generate_dataset_json(join(target_base, 'dataset.json'), target_imagesTr, target_imagesTs,
                              ('Red', 'Green', 'Blue'),
                              labels={0: 'background', 1: 'cell'}, dataset_name=task_name, license='hands off!')
