from __future__ import print_function, absolute_import
import os.path as osp
import glob
import re
from ..utils.data import BaseImageDataset


class llcm_ir(BaseImageDataset):
    """
    sysu_rgb
    train in market1501 type data
    test in orignal sysu data
    """
    #dataset_dir = 'rgb_modify/'

    def __init__(self, root, verbose=True, **kwargs):
        super(llcm_ir, self).__init__()
        self.dataset_dir = root  # osp.join(root, self.dataset_dir)
        self.train_list = osp.join(root, 'idx', 'train_nir.txt')    # for training
        self.test_list = osp.join(root, 'idx', 'test_nir.txt')  # not use

        self._check_before_run()
        train = self._process_dir(self.train_list)
        test = self._process_dir(self.test_list)

        print("=> llcm_ir loaded")
        self.train = train
        self.test = test

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_test_pids, self.num_test_imgs, self.num_test_cams = self.get_imagedata_info(self.test)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(self.num_train_pids, self.num_train_imgs, self.num_train_cams))
        print("  test    | {:5d} | {:8d} | {:9d}".format(self.num_test_pids, self.num_test_imgs, self.num_test_cams))
        print("  ----------------------------------------")

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))

    def _process_dir(self, img_list):
        with open(img_list) as f:
            data_file_list = open(img_list, 'rt').read().splitlines()

            # Get full list of image and labels
            dataset = []
            for line in data_file_list:
                splited = line.split(' ')
                fname = splited[0]
                img_path = osp.join(self.dataset_dir, fname)
                pid = int(splited[1])
                camid = int(fname.split('_c')[1][0:2])
                assert 1 <= camid <= 9
                camid -= 1  # index starts from 0
                dataset.append((img_path, pid, camid))
                #print(fname, pid, camid)

        return dataset
