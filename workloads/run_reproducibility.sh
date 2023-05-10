BASE_DIR=/home/ftang/SNNBench/workloads

RUN_TIMES=1

profile () {
	workload=$1
	image=$2
	workdir=$3
	run_loop=$4
	run_args=$5
	log_dir=${BASE_DIR}/logs/reproducibility/${run_loop}/${workload}

	echo workload: $workload 
	echo docker image: $image 
	echo workdir: $workdir 
	echo log_dir: $log_dir 
	echo run_args: $run_args
	# return 0
	mkdir -p $log_dir
	cd $workdir
	sed -i 's/DOCKER_IMAGE=.*/DOCKER_IMAGE='"${image}"'/g' run_in_docker.sh
	LOG=${log_dir}/${workload}_reproducibility.log
	if test -f ${LOG}; then
		rm ${LOG}
	fi

	# for num_interop_threads in $(seq ${BASE_NUM_THREADS} ${MAX_NUM_THREADS})
	for i in $(seq 1 ${RUN_TIMES})
	do
		./run_in_docker.sh python ${workload}.py ${run_args} | tee -a ${LOG}
	done
}

RUN_TAG=image_stdp_fixed

# profile mnist_stdp_profile bindsnet:0.3.1 ${BASE_DIR} ${RUN_TAG} "--update_interval 100 --n_train 1000 --n_test 100 --n_epochs 1 --gpu"
profile mnist_stdp bindsnet:0.3.1 ${BASE_DIR} ${RUN_TAG} "--update_interval 100 --n_train 1000 --n_test 100 --n_epochs 1 --gpu"
# profile mnist_surrogate snnbench ${BASE_DIR} ${RUN_TAG}
# profile train_mlp bindsnet:0.3.1 ${BASE_DIR}/conversion ${RUN_TAG} "--job-dir logs --gpu"
# profile snn_inference bindsnet:0.3.1 ${BASE_DIR}/conversion ${RUN_TAG} "--job-dir logs --gpu --results-file ann"
# profile train_mlp bindsnet:0.3.1 ${BASE_DIR}/conversion ${RUN_TAG} "--job-dir logs"
# profile snn_inference bindsnet:0.3.1 ${BASE_DIR}/conversion ${RUN_TAG} "--job-dir logs --results-file ann"
# for model in lif lsnn lstm
# do
# 	# profile speech norse ${BASE_DIR}/speech run_1/${model} "--model ${model} --device cpu"
# 	profile speech norse ${BASE_DIR}/speech ${RUN_TAG}/${model} "--model ${model} --device cuda"
# done
