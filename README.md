# default_predictor_using_classifier_model_w_zenml_mlflow


![Image Alt Text](test_design.png)


## Run can be found in zemml as below

## Experiment  can be found in mlflow as below



## One time setup (go to project root dir b4 running below)
---
```bash
cd %CD%
conda config --add envs_dirs %CD%
conda create --name default_predictor_zml_mlflow --yes python=3.11
conda activate default_predictor_zml_mlflow
pip install zenml && pip install "zenml[server]"  && pip install "matplotlib" && zenml integration install sklearn -y && pip install pandas && pip install scikit-learn && pip install zenml && pip install mlflow && pip install tensorflow
conda env list
zenml integration install mlflow
zenml integration install sklearn
zenml experiment-tracker register mlflow_tracker --flavor mlflow
zenml stack delete mlflow_stack -y
zenml stack register mlflow_stack -e mlflow_tracker -a default -o default --set

```
zenml stack describe mlflow_stack

## Run mlflow 
mlflow ui --backend-store-uri file:///%APPDATA%\zenml\local_stores\e30b8917-359c-4bbe-9f57-83774ec8ac2d\mlruns

## Run next time
---
```bash
* cd into project directory
* then run >   %CD%\start.bat
* then login into http://127.0.0.1:8237/  as  username : default and no passwored required
```

## Stop and clean up
---
```bash
stop zenml server manually (ctr_c) as its running in blocking mode 
conda deactivate 
zenml clean -y
```

