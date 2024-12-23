TASK 1. Containerization

The task is to setup an isolated environment to work on a Pneumonia Detection Problem (https://www.kaggle.com/datasets/iamtapendu/rsna-pneumonia-processed-dataset). The Dockerfile created should build a Docker image with all the dependencies, data files and permissions needed to start Jupyter Notebook and run Experiments.ipynb.  

Commands to execute after pulling the repo:
- cd mle_course/01_containerization
- docker build --no-cache -t pneumonia-image .
- docker run --env-file=.env -p 8888:8888 pneumonia-image
- in the opened browser window run Experiments.ipynb and execute all cells. The last cell should contain 16 images from the validation set with correct labels and labels predicted by the model. 