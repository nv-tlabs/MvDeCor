# MvDeCor

This is an official code release of

### MvDeCor: Multi-view Dense Correspondence Learning for Fine-grained 3D Segmentation

Gopal Sharma, Kangxue Yin, Subhransu Maji, Evangelos Kalogerakis, Or Litany and Sanja Fidler
<img src=docs/teaser.png width="1024">
****

### Requirements

- Python 3.9 is supported.
- Pytorch 1.5.1.
- This code is tested with CUDA 10.1 toolkit
- Use the following script to install conda environment

```
bash install.sh
```

### Dataset download and processing
Use the following script to download and process the dataset

```bash dataset_process.sh```

update ```categories``` to categories you want.

### Training and testing
For pretraining, run the following script (requires 2 GPUS):

```bash partnet.sh```

For training few shot segmentation on partnet dataset, run the following script (require 1 GPU):

```bash partnet_seg.sh```

Note that test is automatically done in the code, after training is completed.

### Citation

```
@inproceedings{mvdecor2022,
    title={MvDeCor: Multi-view Dense Correspondence Learning for Fine-grained 3D Segmentation},
    author={Gopal Sharma and  Kangxue Yin and  Subhransu Maji and  Evangelos Kalogerakis and  Or Litany and Sanja Fidler},
    booktitle={Proceedings of the European Conference on Computer Vision Workshops (ECCV)},
    year={2022}
}
```