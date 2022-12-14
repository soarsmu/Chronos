# Chronos

This repository contains source code of research paper "Chronos: Zero-Shot Identification of Libraries from Vulnerability Reports", which is submitted to ICSE 2023

```
@inproceedings{lyu2023chronos,
  title={Chronos: Zero-Shot Identification of Libraries from Vulnerability Reports},
  author={Lyu, Yunbo and Le-Cong, Thanh and Kang, Hong Jin and Widyasari, Ratnadira and Zhao, Zhipeng and Le, Xuan-Bach D and Li, Ming and Lo, David},
  booktitle={Proceedings of the 45th IEEE/ACM Internation Conference on Software Engineering},
  year={2023}
}
```
## Dataset
Before using Chronos, please our [data](https://figshare.com/articles/software/Chronos-ICSE23/20787805) from Figshare 

## Structure

The structure of our source code's repository is as follows:
- dataset: contains our dataset for empirical evaluation;
- reference_processing: contains source code for preprocessing reference data;
- zestxml: contains our source code for zero-shot learning model.
- analyze_data.py: contains our source code for analyzing unseen labels and the associated data points
 
## Environment Configuration
For ease of use, we also provide a 
installation package via a [docker image](https://hub.docker.com/repository/docker/chronosicse22/chronos). User can setup AutoPruner's docker step-by-step as follows:

- Pull AutoPruner's docker image: 
```
docker pull chronosicse22/chronos:v1
```
- Run a docker container:
```
docker run --name chronos -it --shm-size 16G --gpus all chronosicse22/chronos:v1
```

## Experiments
### Usage 
```
bash auto_run.sh -d [description data: "merged" or "description_and_reference"]
                 -l [label processing: "splitting" or "none"]
                 -m [the M parameter on Equation (6) for adjustment] 
                 -i [top-i highest labels for adjustment]
```

### Preprocess reference data
If you want to create reference data from scratch, please use the following commands:
```
cd reference_processing
python3 generate_new_csv.py
```

### Replicate our results

To replicate our results for RQ1, please use:
```
python3 analyze_data.py
```

To replicate our results for RQ2, please use:
```
bash auto_run.sh -d 'description_and_reference' -l 'splitting' -m 8 -i 10
```

To replicate our results for RQ3, please use:
- Chronos without adjustment
```
bash auto_run.sh -d 'description_and_reference' -l 'splitting' -m 0 -i 0
```
- Chronos without data enhancement
```
bash auto_run.sh -d 'merged' -l 'none' -m 8 -i 10
```




