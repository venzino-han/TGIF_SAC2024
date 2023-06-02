default: build

help:
	@echo 'Management commands for fasion_graph:'
	@echo
	@echo 'Usage:'
	@echo '    make build            Build image'
	@echo '    make pip-sync         Pip sync.'

preprocess:
	@docker 

build:
	@echo "Building Docker image"
	@docker build . -t fasion_graph 

run:
	@echo "Booting up Docker Container"
	@docker run -it --gpus '"device=0"' --ipc=host --name fasion_graph -v `pwd`:/workspace/fasion_graph fasion_graph:latest /bin/bash

up: build run

rm: 
	@docker rm fasion_graph

stop:
	@docker stop fasion_graph

reset: stop rm