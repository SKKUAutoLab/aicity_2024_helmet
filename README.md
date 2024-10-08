## Automation Lab, Sungkyunkwan University

# A. Run from Docker

---

##### a. Data download

Go to the website of AI-City Challenge to get the dataset.

- https://www.aicitychallenge.org/2024-data-and-evaluation/

Download dataset to the folder **<folder_test_dataset>**

The dataset folder structure should be as following:

```
<folder_test_dataset>
│   ├── videos
│   │   ├── 001
│   │   ├── 002
...
```

##### b. Load Docker

Change the **<folder_test_dataset>** with your path and run:

```shell
docker run  \
    --ipc=host  \
    --gpus all   \
    -v <folder_test_dataset>:/usr/src/aic24-track_5/data   \
    -it supersugar/skku_automation_lab_aic24_track_5:latest
```

##### c. Run inference

And the running script to get the result

```shell
bash script/run_track_5_docker.sh 
```

##### d. Get the result
After more than 2-3 hours, we get the result:
```
<folder_test_dataset>/outputs_s2_v8_det_v8_iden/final_result_s2.txt
```

---

# B. Run from source

---

#### I. Installation

1. Download & install Miniconda or Anaconda from https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html


2. Open new Terminal, create new conda environment named **motorbike_helmet_violation** and activate it with following commands:

```shell
conda create --name motorbike_helmet_violation python=3.10

conda activate motorbike_helmet_violation

pip install -r requirements.txt
```

---


#### II. Data preparation

##### a. Data download

Go to the website of AI-City Challenge to get the dataset.

- https://www.aicitychallenge.org/2024-data-and-evaluation/

##### b. Video data import

Add video files to **data**.
   
The program folder structure should be as following:

```
aicity_2024_helmet
├── data
│   ├──helmet
│   │   ├──videos
│   │   
...
```

---

#### III. Reference

##### a. Download weight 

Download weight from [Release](https://1drv.ms/u/s!Anfa0di4JZJfkxsCCFGv6a9G_Tyg?e=VkgXnF) then put it into **aicity_2024_helmet/model_zoo/aic24/models_zoo**.

The folder structure should be as following:
```

aicity_2024_helmet
├── models_zoo
│   ├──aic24 
│   │   ├── kmeans_cluster
│   │   ├── yolov8x_1536_1cls_track_5_24_v2
│   │   ├── yolov8x_320_9cls_track_5_24_crop_train_equal_val_v4_videos_gr_0
│   │   ├── yolov8x_448_9cls_track_5_24_crop_train_equal_val_v4_videos_gr_0
│   │   ├── yolov8x_320_9cls_track_5_24_crop_train_equal_val_v4_videos_gr_1
│   │   ├── yolov8x_448_9cls_track_5_24_crop_train_equal_val_v4_videos_gr_1
│   │   └── yolov8x_512_9cls_track_5_24_crop_train_equal_val_v4_videos_gr_1
```

##### b. Run inference

And the running script to get the result

```shell
cd aicity_2024_helmet

bash run_track_5_docker.sh 
```

##### c. Get the result
After more than 2-3 hours, we get the result:
```
aicity_2024_helmet/data/output_helmet/final_result.txt
```

---

# C. Training Dataset

---

##### a. Dataset for the Detector

Download dataset for Detector from [link](https://o365skku-my.sharepoint.com/:f:/g/personal/duongtran_o365_skku_edu/Eo2nfe_g62VNocpi_6mOIjsBFPbXaDiVat1C7vaJ6HLJ_g?e=e5tjcB).

##### b. Dataset for the Identifier

Download dataset for Identifier from [link](https://o365skku-my.sharepoint.com/:f:/g/personal/duongtran_o365_skku_edu/Eo2nfe_g62VNocpi_6mOIjsBFPbXaDiVat1C7vaJ6HLJ_g?e=e5tjcB).
