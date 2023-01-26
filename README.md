# Multi-level-colonoscopy-malignant-tissue-detection-with-adversarial-CAC-UNet [![Tweet](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/intent/tweet?text=Codes%20and%20Data%20for%20Our%20Paper:%20"Multi-level%20colonoscopy%20malignant%20tissue%20detection%20with%20adversarial%20CAC-UNet"%20&url=https://github.com/bupt-ai-cz/CAC-UNet-DigestPath2019)
Implementation detail for our paper ["Multi-level colonoscopy malignant tissue detection with adversarial CAC-UNet"](https://www.researchgate.net/publication/348780589_Multi-level_colonoscopy_malignant_tissue_detection_with_adversarial_CAC-UNet#fullTextFileContent)

DigestPath 2019

The proposed scheme in this paper achieves the best results in MICCAI DigestPath2019 challenge (https://digestpath2019.grand-challenge.org/Home/) on colonoscopy tissue segmentation and classification task.

## Dataset
Description of dataset can be found here:
https://digestpath2019.grand-challenge.org/Dataset/

To download the the DigestPath2019 dataset, please sign the DATABASE USE AGREEMENT first at here:
https://digestpath2019.grand-challenge.org/Download/

If you have problems about downing the dataset, please contact Prof. Hongsheng Li:hsli@ee.cuhk.edu.hk

Image sample:
![](https://github.com/PkuMaplee/Multi-level-colonoscopy-malignant-tissue-detection-with-adversarial-CAC-UNet/blob/master/sample-image.jpg)

## Envs
- Pytorch 1.0
- Python 3+
- cuda 9.0+

install
```
$ pip install -r  requirements.txt
```

`apex` :  Tools for easy mixed precision and distributed training in Pytorch
```
$ git clone https://github.com/NVIDIA/apex
$ cd apex
$ pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

## Dataset
```
├── data/
│   ├── tissue-train-neg/     
│   ├── tissue-train-pos-v1/
```
## Preprocessing
```
$ cd code/
$ python preprocessing.py
```

## Training
```
$ cd code/
$ python train.py --config_file='config/cac-unet-r50.yaml'
```
## Citation
Please cite this paper in your publications if it helps your research:

```
@article{zhu2021multi,
  title={Multi-level colonoscopy malignant tissue detection with adversarial CAC-UNet},
  author={Zhu, Chuang and Mei, Ke and Peng, Ting and Luo, Yihao and Liu, Jun and Wang, Ying and Jin, Mulan},
  journal={Neurocomputing},
  volume={438},
  pages={165--183},
  year={2021},
  publisher={Elsevier}
}
```

About the multi-level adversarial segmentation part, you can read our [ICASSP](https://arxiv.org/pdf/2002.08587.pdf) paper for more details:

```
@inproceedings{mei2020cross,
  title={Cross-stained segmentation from renal biopsy images using multi-level adversarial learning},
  author={Mei, Ke and Zhu, Chuang and Jiang, Lei and Liu, Jun and Qiao, Yuanyuan},
  booktitle={ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1424--1428},
  year={2020},
  organization={IEEE}
}
```
The challenge paper [DigestPath: a Benchmark Dataset with Challenge Review for the Pathological Detection and Segmentation of Digestive-System](https://github.com/bupt-ai-cz/CAC-UNet-DigestPath2019/tree/main/papers/DigestPath-a Benchmark Dataset with Challenge Review.pdf) should be also cited:
```
@article{da2022digestpath,
  title={DigestPath: A benchmark dataset with challenge review for the pathological detection and segmentation of digestive-system},
  author={Da, Qian and Huang, Xiaodi and Li, Zhongyu and Zuo, Yanfei and Zhang, Chenbin and Liu, Jingxin and Chen, Wen and Li, Jiahui and Xu, Dou and Hu, Zhiqiang and others},
  journal={Medical Image Analysis},
  volume={80},
  pages={102485},
  year={2022},
  publisher={Elsevier}
}
```

## Author
Ke Mei, Ting Peng, Chuang Zhu
- email: czhu@bupt.edu.cn；raykoo@bupt.edu.cn
- wechat: meikekekeke

If you have any questions, you can contact me directly.
