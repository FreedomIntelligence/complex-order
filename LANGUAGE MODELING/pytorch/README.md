
## Prerequisite

- Pytorch 0.4: `conda install pytorch torchvision -c pytorch`


## Data Prepration

`bash getdata.sh`

## Training and Evaluation

- Training

  `bash run_text8_base.sh train --work_dir PATH_TO_WORK_DIR`

- Evaluation

  `bash run_text8_base.sh eval --work_dir PATH_TO_WORK_DIR`