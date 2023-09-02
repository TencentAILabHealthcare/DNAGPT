# DNAGPT: A Generalized Pre-trained Tool for Versatile DNA Sequence Analysis Tasks

## Download code and pre-trained weights

### Download codes

```bash
git clone https://github.com/TencentAILabHealthcare/DNAGPT.git
```

### Download pre-trained weights
You can download the weights from
[Google Drive](https://drive.google.com/drive/folders/10UPPx6V13oQW6knuLV7d8SRIA3D6hYor?usp=drive_link)

## Set up the environment

```bash
# create and activate virtual python environment
conda create -n dnagpt python=3.10
conda activate dnagpt

# install required packages
cd DNAGPT
pip install -r requirements.txt
```

## Test on your data

```bash
python test.py --input_dna=<your dna data> --weight_path=<path to the pre-trained weight> --model_name=<the model you want to use> --spec_token=<the specical token corresponding to species of the dna>
```

### Some tips:

1.The model name must be choosen from \['dnagpt_m', 'dnagpt_s_512', 'dnagpt_b_512'\] which is the same with the name of the pre-trained weights. 'dnagpt_m' supports a maximum input length of 24564 bps and 'dnagpt_s_512', 'dnagpt_b_512' support a maximum input length of 3060 bps.

2.The spec_token is set default to 'R' which means human.

### Example:
```bash
python test.py --input_dna='AATAAAAA' --weight_path='/home/DNAGPT_S_512.pth' --model_name='dnagpt_s_512'  --spec_token='R'
```






