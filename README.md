# Modality-Aware Bias Mitigation and Invariance Learning for Unsupervised Visible-Infrared Person Re-Identification

This repo contains implementation for our paper: [Modality-Aware Bias Mitigation and Invariance Learning for Unsupervised Visible-Infrared Person Re-Identification](https://arxiv.org/abs/2512.07760).

![teaser](figure/framework.pdf)


## Dataset Preparation
Download the VI-reID datasets and put them under your dataset path.


## Training
Change the necessary data paths and run the following:
```shell
sh run_train_sysu.sh   # for SYSU-MM01
sh run_train_regdb.sh  # for RegDB
```

## Test
Change the necessary data paths and run the following:
```shell
./test_sysu.sh    # for SYSU-MM01
./test_regdb.sh   # for RegDB
```

# Citation
```bibtex
@inproceedings{Wang2026BMIL,
    author    = {Menglin Wang and Xiaojin Gong and Jiachen Li and Genlin Ji},
    title     = {Modality-Aware Bias Mitigation and Invariance Learning for Unsupervised Visible-Infrared Person Re-Identification},
    booktitle = {AAAI Conference on Artificial Intelligence},
    year      = {2026}
}
```


# Acknowledgements
The code is implemented based on [PGM](https://github.com/zesenwu23/USL-VI-ReID) and [RPNR](https://arxiv.org/abs/2405.05613) codebase. Thanks to their open-source implementations.
