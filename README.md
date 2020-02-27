# Document_layout_analysis

Implementation code for document layout analysis

This work is implemented by "Machine fancy behavour Lab" Team

Presentation(Chinese):

link: https://pan.baidu.com/s/1qk9LP3zi6xX4V2sB-3PBiQ code: dumw

PPT(Chinese):

link: https://pan.baidu.com/s/1kgwZ5_-apCpG-W_EnuPW-g code: q1kv 

contect us: 
 
- yiming.lin17@student.xjtlu.edu.cn
- jicen.yu17@student.xjtlu.edu.cn
- yeyun.zou17@student.xjtlu.edu.cn

## Contents
1. [Environment Setup](#environment-setup)
2. [Demo](#demo)
3. [Training Models](#training-models)

## Environment setup

- Clone the repository 
```
git clone https://github.com/JaMesLiMers/Hakathon_layout_analysis.git && cd Hakathon_layout_analysis
```

- Setup python environment
```
conda create -n torch python=3.6
source activate torch
pip install -r requirements.txt
```

## Demo

- Some of ouf model structure

<div align="center">
  <img src="https://github.com/JaMesLiMers/Hakathon_layout_analysis/blob/master/test/log/model_2.png" width="300px" />
  <img src="https://github.com/JaMesLiMers/Hakathon_layout_analysis/blob/master/test/log/model_1.png" width="300px" />
</div>

<div align="center">
  <img src="https://github.com/JaMesLiMers/Hakathon_layout_analysis/blob/master/test/Image/XJTLU_d00007.jpg" width="200px" />
  <img src="https://github.com/JaMesLiMers/Hakathon_layout_analysis/blob/master/test/Output/result_6.png" width="200px" />
  <img src="https://github.com/JaMesLiMers/Hakathon_layout_analysis/blob/master/test/Output/result_second_6.png" width="200px" />
</div>

- [Setup](#environment-setup) your environment
- Download the trained model and put into Hakathon_layout_analysis/save/
- trained model avaliable in:
```
https://drive.google.com/open?id=1YSNEL5xzaLlfLiU7t1sEnSGPa3SI9-3O
```
- Put input image into Hakathon_layout_analysis/test/Image('.jpg' file)
- Run `test.py/ test_second.py`
```shell
python ./test.py --resume model_epoch_409.pkl
python ./test_second.py --resume model_second_epoch_409.pkl
```
- Output will be generated in Hakathon_layout_analysis/test/Output

## Training Models
- [Setup](#environment-setup) your environment
- Download test data(Not avaliable now) put into Hakathon_layout_analysis/Data/
- Download pretrained models(alexnet_bn_.pth), avaliable in:
```
https://drive.google.com/open?id=1YSNEL5xzaLlfLiU7t1sEnSGPa3SI9-3O
```
- Run train code:
```shell
python ./train.py
python ./train_second.py
```
- Output will be generated in Hakathon_layout_analysis/save/

