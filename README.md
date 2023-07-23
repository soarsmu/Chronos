# Chronos

This repository contains the source code of our research [paper](https://arxiv.org/pdf/2301.03944.pdf) "CHRONOS: Time-Aware Zero-Shot Identification of Libraries from Vulnerability Reports", which is accepted at ICSE 2023

```
@inproceedings{lyu2023chronos,
  title={CHRONOS: Time-Aware Zero-Shot Identification of Libraries from Vulnerability Reports},
  author={Lyu, Yunbo and Le-Cong, Thanh and Kang, Hong Jin and Widyasari, Ratnadira and Zhao, Zhipeng and Le, Xuan-Bach D and Li, Ming and Lo, David},
  booktitle={Proceedings of the 45th IEEE/ACM Internation Conference on Software Engineering},
  year={2023}
}
```

## Note
There is a minor inconsistency between the paper and the code that has been addressed in the latest [arxiv paper](https://arxiv.org/abs/2301.03944). Specifically, the code use normalized P@K to consider the best possible P@K. The paper mentions standard P@K. The arxiv paper has now been updated. The arxiv paper also now includes results of the standard P@K (in the Appendix) and we have similar findings: Chronos outperforms ZestXML by 20%+, and both outperform LightXML by a big margin.


## Dataset
Before using Chronos, please download our [data](https://figshare.com/articles/software/Chronos-ICSE23/22082075) from Figshare.

You should unzip all files in the dataset folder so that you can use Chronos.

## Structure

The structure of our source code's repository is as follows:
- dataset: contains our dataset for empirical evaluation;
- reference_processing: contains source code for preprocessing reference data;
- zestxml: contains our source code for zero-shot learning model.
- analyze_data.py: contains our source code for analyzing unseen labels and the associated data points
 
## Environment Configuration
For ease of use, we also provide a 
installation package via a [docker image](https://hub.docker.com/repository/docker/chronosicse22/chronos). You can set up Chronos's docker step-by-step as follow:

- Pull Chronos's docker image: 
```
docker pull chronosicse22/chronos:v1
```
- Run a docker container:
```
docker run --name chronos -it --shm-size 16G --gpus all chronosicse22/chronos:v1
```
An option command to run a docker container:
```
docker run -it -v </media/Rb/:/workspace/> --name chronos_ae chronosicse22/chronos:v1
```
</media/Rb/:/workspace/> is your workspace path, you need to change it for your usage.

## Experiments
### Path Setting
You need to update the workspace path in auto_run.sh https://github.com/soarsmu/Chronos/blob/07e0a7571a42470a85391b4b921e4ea6f08b0b27/auto_run.sh#L12

The change in the auto_run.sh script is to point to the local directory of the dataset.


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

### Expected Output
For RQ1, after executing the script, you will find the information about seen and unseen labels by years. For RQ2 and RQ3, after executing each script, you will find Precision, Recall, and F1 for each experiment set.

You can get the detailed expected output in this [document](https://github.com/soarsmu/Chronos/blob/main/USAGE%20EXAMPLES.pdf).

# LightXML

Grid search was performed on two hyperparameters: batch size (bs), epochs, and learning rate (lr). 

Particularly, we use batch sizes in {1, 2, 4, 8, 16}; learning rates in {1e-6, 1e-5, 1e-4, 1e-3, and 1e-2}; and epochs in {20, 25, 30, 35, 40}. 

We used the hyperparameters that result in LightXML’s best performance on the validation dataset to evaluate its performance on the testing dataset.

To use LightXML, please refer to our [previous study](https://github.com/soarsmu/ICPC_2022_Automated-Identification-of-Libraries-from-Vulnerability-Data-Can-We-Do-Better).
