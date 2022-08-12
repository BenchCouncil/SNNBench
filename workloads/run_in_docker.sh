DOCKER_IMAGE="bindsnet:0.1.7"

set -x
CMD="cd $(pwd); $@"
docker run --rm -it \
	       --ipc=host \
	       --security-opt seccomp=unconfined \
               -v $(pwd):$(pwd) \
	       ${DOCKER_IMAGE} bash -c "${CMD}"
	       # --gpus all \
