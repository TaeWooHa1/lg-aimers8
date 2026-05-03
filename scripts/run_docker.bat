docker run -it --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v %cd%:/app hackathon-env
