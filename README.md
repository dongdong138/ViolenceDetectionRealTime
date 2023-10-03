<h1 align="center">
Violence Detection Real Time
</h1>

### Installation
1. Clone this repository.
   
    ```
    python train.py
    ```  
3. Create a virtual environment and activate it.
   
    ```
    python3 -m venv .env
    ```  
3. Install Tensorflow with CUDA [follow this Tensorflow installation](https://www.tensorflow.org/install).
4. Install requirement libraries.
   
    ```
    pip install -r requirement.txt
    ```  
         
### Dataset preparation
To get RWF2000 dataset:
1. Go to [RWF2000-Video-Database-for-Violence-Detection](https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection).
2. Folder structure.
```
📦project_directory
  ┣ 📂RWF-2000
    ┣ 📂train
      ┣ 📂Fight
      ┣ 📂NonFight
    ┣ 📂val
      ┣ 📂Fight
      ┣ 📂NonFight
```

### How to run
To process data go to project directory and run *process_data.py* like below.
```
python process_data.py
```
To train models go to project directory and run *train.py* like below.
```
python train.py
```
To test an already trained model, use *test.py* like below.
```
python test.py
```

Reference: [Violence Detection From Videos Captured By CCTV](https://medium.com/@ravinadable16/violence-detection-from-videos-captured-by-cctv-d032a254d489)
