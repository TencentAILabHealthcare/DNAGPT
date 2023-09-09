# DNAGPT: A Generalized Pre-trained Tool for Versatile DNA Sequence Analysis Tasks

The official implementation of [DNAGPT: A Generalized Pre-trained Tool for Versatile DNA Sequence Analysis Tasks](https://www.biorxiv.org/content/10.1101/2023.07.11.548628v2.full.pdf).

## Getting Started

### Download codes

```bash
git clone https://github.com/TencentAILabHealthcare/DNAGPT.git
```

### Download pre-trained weights
You can download the weights from
* [Google Drive](https://drive.google.com/drive/folders/10UPPx6V13oQW6knuLV7d8SRIA3D6hYor?usp=drive_link)
* [Tencent Weiyun](https://share.weiyun.com/car87dsv)

and save model weights to checkpoint dir
```bash
cd DNAGPT/checkpoints
# download or copy model weight to this default directory
```
#### Foundation model
* [dna_gpt0.1b_h.pth](https://drive.google.com/file/d/15m6CH3zaMSqflOaf6ec5VPfiulg-Gh0u/view?usp=drive_link): DNAGPT 0.1B params model pretrained with human genomes
* [dna_gpt0.1b_m.pth](https://drive.google.com/file/d/1C0BRXfz7RNtCSjSY1dKQeR1yP7I3wTyx/view?usp=drive_link): DNAGPT 0.1B params model pretrained with mutli-organism genomes
* [dna_gpt3b_m.pth](https://drive.google.com/file/d/1pQ3Ai7C-ObzKkKTRwuf6eshVneKHzYEg/view?usp=drive_link): DNAGPT 3B params model pretrained with mutli-organism genomes

#### Finetune model
* [regression.pth](https://drive.google.com/file/d/1_BDbfB5iNmfus3imx1_YSD1ac6OiJkaY/view?usp=drive_link): Human RNA experssion level regression model
* [classification.pth](https://drive.google.com/file/d/1TdMCiJO6rq32WSka73VdKI0Cthitd9Bb/view?usp=drive_link): Human AATAAA GSR classification model

## Install

### Pre-requirements
* python >= 3.8

### Required packages
```bash
cd DNAGPT
pip install -r requirements.txt
```

### Test
### Example
```bash
python test.py --task=<task type> --input=<your dna data> --weight=<path to the pre-trained weight> --name=<the model you want to use> --num_samples=<number of samples seq>
```
go to directory "scripts" for more test examples.
#### Generation
```bash
# gpt 0.1b human genomes model
python test.py --task 'generation' --input '<R>AGAGAAAAGAGT' --name 'dna_gpt0.1b_h' --weight 'checkpoints/dna_gpt0.1b_h.pth' --num_samples 10 --max_len 256
# gpt 0.1b multi-organism model
python test.py --task 'generation' --input '<R>AGAGAAAAGAGT' --name 'dna_gpt0.1b_m' --weight 'checkpoints/dna_gpt0.1b_m.pth' --num_samples 10 --max_len 256
# gpt 3b multi-organism model
python test.py --task 'generation' --input '<R>AGAGAAAAGAGT' --name 'dna_gpt3b_m' --weight 'checkpoints/dna_gpt3b_m.pth' --num_samples 10 --max_len 256
```
#### Regression
```shell
python test.py --task 'regression' --input xxxxx --numbers xxxxx --name 'dna_gpt0.1b_h' --weight 'checkpoints/regression.pth'
```
#### Classification
```shell
python test.py --task 'classification' --input xxxxx --name 'dna_gpt0.1b_m' --weight 'checkpoints/classification.pth'
```

### Tips:
1. 'dna_gpt0.1b_m' supports a maximum input length of 24564 bps and 'dna_gpt0.1b_s', 'dna_gpt3b_m' support a maximum input length of 3060 bps. 
2. The spec_token is set default to 'R' which means human. special token should use with "<", ">", like "<R>"

### Citation

**DNAGPT**

```
@article{zhang2023dnagpt,
  title={DNAGPT: A Generalized Pretrained Tool for Multiple DNA Sequence Analysis Tasks},
  author={Zhang, Daoan and Zhang, Weitong and He, Bing and Zhang, Jianguo and Qin, Chenchen and Yao, Jianhua},
  journal={bioRxiv},
  pages={2023--07},
  year={2023},
  publisher={Cold Spring Harbor Laboratory}
}
```



