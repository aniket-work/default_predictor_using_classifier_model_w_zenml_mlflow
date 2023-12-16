# default_predictor_using_classifier_model_w_zenml_mlflow


![Image Alt Text](test_design.png)


## One time setup (go to project root dir b4 running below)
---
```bash
cd %CD%
conda config --add envs_dirs %CD%
conda create --name default_predictor_zml_mlflow --yes python=3.11
conda activate default_predictor_zml_mlflow
pip install -r requirements.txt
conda env list
```


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
stop zenml server manually ( ctr_c) as its running in blocking mode 
conda deactivate 
zenml clean -y
```

