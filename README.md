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
* [Tencent Weiyun](https://share.weiyun.com/J1BWWkQF)

and save model weights to checkpoint dir
```bash
cd DNAGPT
mkdir checkpoints
# download or copy model weight to this default directory
```
#### Foundation model
* [dna_gpt0.1b_s.pth](https://drive.google.com/file/d/1C0BRXfz7RNtCSjSY1dKQeR1yP7I3wTyx/view?usp=drive_link): DNAGPT 0.1B params model pretrained with human genomes
* [dna_gpt0.1b_m.pth](https://drive.google.com/file/d/1h6tcP1qncw2uf1d4vRIwIBRUAjgNMtUa/view?usp=drive_link): DNAGPT 0.1B params model pretrained with mutli-organism genomes
* [dna_gpt3b_m.pth](https://drive.google.com/file/d/18Su9-DGwWaONX6UgVnU5if7ClQXS299Y/view?usp=drive_link): DNAGPT 3B params model pretrained with mutli-organism genomes

## Install

### Pre-requirements
* python >= 3.8

### Required packages
```bash
cd DNAGPT
pip install -r requirements.txt
```

### Test
```bash
python sample.py --input=<your dna data> --weight=<path to the pre-trained weight> --name=<the model you want to use> --num_samples=<number of samples seq>
```
### Example
```bash
python sample.py --input '<R>AGAGAAAAGAGT' --name 'dna_gpt0.1b_s' --num_samples 10
```

### Tips:
1. 'dna_gpt0.1b_m' supports a maximum input length of 24564 bps and 'dna_gpt0.1b_s', 'dna_gpt3b_m' support a maximum input length of 3060 bps. 
2. The spec_token is set default to 'R' which means human. special token should use with "<", ">", like "<R>"





