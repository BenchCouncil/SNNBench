# DOCKER_IMAGE=bindsnet:0.3.1
DOCKER_IMAGE=bindsnet:0.3.1

set -x
CMD="cd $(pwd); $@"
docker run --rm -it \
	       --gpus all \
	       --ipc=host \
	       --security-opt seccomp=unconfined \
               -v $(pwd):$(pwd) \
	       ${DOCKER_IMAGE} bash -c "${CMD}"
	       # --gpus all \
