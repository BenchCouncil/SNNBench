# DOCKER_IMAGE=snnbench
DOCKER_IMAGE=snnbench

set -x
CMD="cd $(pwd); $@"
docker run --rm -it \
	       --gpus all \
	       --ipc=host \
	       --security-opt seccomp=unconfined \
               -v $(pwd):$(pwd) \
	       ${DOCKER_IMAGE} bash -c "${CMD}"
	       # --gpus all \
