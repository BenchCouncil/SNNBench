BASE_DIR=/home/ftang/SNNBench/workloads

BASE_NUM_THREADS=1
MAX_NUM_THREADS=8

profile () {
	workload=$1
	image=$2
	workdir=$3
	run_loop=$4
	run_args=$5
	log_dir=${BASE_DIR}/logs/diff_threads/${run_loop}/${workload}

	echo workload: $workload 
	echo docker image: $image 
	echo workdir: $workdir 
	echo log_dir: $log_dir 
	echo run_args: $run_args
	# return 0
	mkdir -p $log_dir
	cd $workdir
	sed -i 's/DOCKER_IMAGE=.*/DOCKER_IMAGE='"${image}"'/g' run_in_docker.sh

	# for num_interop_threads in $(seq ${BASE_NUM_THREADS} ${MAX_NUM_THREADS})
	for num_interop_threads in $(seq 1 ${MAX_NUM_THREADS})
	do
		num_threads=${BASE_NUM_THREADS}
		./run_in_docker.sh python ${workload}_profile.py ${run_args} --num_interop_threads ${num_interop_threads} --num_threads ${num_threads} | tee ${log_dir}/${workload}_profile_interop_${num_interop_threads}_intraop_${num_threads}.log
	done

	for num_threads in $(seq 1 ${MAX_NUM_THREADS})
	do
		num_interop_threads=${BASE_NUM_THREADS}
		./run_in_docker.sh python ${workload}_profile.py ${run_args} --num_interop_threads ${num_interop_threads} --num_threads ${num_threads} | tee ${log_dir}/${workload}_profile_interop_${num_interop_threads}_intraop_${num_threads}.log
	done
}

function profile_mnist_stdp {
	cd ${BASE_DIR}
	for num_interop_threads in seq ${BASE_NUM_THREADS} ${MAX_NUM_THREADS}
	do
		num_threads=${BASE_NUM_THREADS}
		./run_in_docker.sh python mnist_stdp_profile.py --num_interop_threads ${num_interop_threads} --num_threads ${num_threads} | tee ${BASE_DIR}/logs/mnist_stdp_diff_threads/mnist_stdp_profile_interop_${num_interop_threads}_intraop_${num_threads}.log
	done

	for num_threads in {${BASE_NUM_THREADS}..${MAX_NUM_THREADS}}
	do
		num_interop_threads=${BASE_NUM_THREADS}
		./run_in_docker.sh python mnist_stdp_profile.py --num_interop_threads ${num_interop_threads} --num_threads ${num_threads} | tee ${BASE_DIR}/logs/mnist_stdp_diff_threads/mnist_stdp_profile_interop_${num_interop_threads}_intraop_${num_threads}.log
	done
}

function profile_mnist_surrogate {
	cd ${BASE_DIR}
	sed -i 's/DOCKER_IMAGE=".*"/DOCKER_IMAGE="snntorch"/g' run_in_docker.sh
	for num_interop_threads in {${BASE_NUM_THREADS}..${MAX_NUM_THREADS}}
	do
		num_threads=${BASE_NUM_THREADS}
		./run_in_docker.sh python mnist_surrogate_profile.py --num_interop_threads ${num_interop_threads} --num_threads ${num_threads} | tee ${BASE_DIR}/logs/mnist_surrogate_diff_threads/mnist_surrogate_profile_interop_${num_interop_threads}_intraop_${num_threads}.log
	done

	for num_threads in {${BASE_NUM_THREADS}..${MAX_NUM_THREADS}}
	do
		num_interop_threads=${BASE_NUM_THREADS}
		./run_in_docker.sh python mnist_surrogate_profile.py --num_interop_threads ${num_interop_threads} --num_threads ${num_threads} | tee ${BASE_DIR}/logs/mnist_surrogate_diff_threads/mnist_surrogate_profile_interop_${num_interop_threads}_intraop_${num_threads}.log
	done
}

function profile_mnist_conversion {
	cd ${BASE_DIR}/conversion
	sed -i 's/DOCKER_IMAGE=".*"/DOCKER_IMAGE="bindsnet:0.3.1"/g' run_in_docker.sh
	for num_interop_threads in {${BASE_NUM_THREADS}..${MAX_NUM_THREADS}}
	do
		num_threads=${BASE_NUM_THREADS}
		./run_in_docker.sh python conversion_profile.py --job-dir logs --num_interop_threads ${num_interop_threads} --num_threads ${num_threads} | tee ${BASE_DIR}/logs/mnist_conversion_diff_threads/mnist_conversion_profile_interop_${num_interop_threads}_intraop_${num_threads}.log
	done

	for num_threads in {${BASE_NUM_THREADS}..${MAX_NUM_THREADS}}
	do
		num_interop_threads=${BASE_NUM_THREADS}
		./run_in_docker.sh python conversion_profile.py --job-dir logs --num_interop_threads ${num_interop_threads} --num_threads ${num_threads} | tee ${BASE_DIR}/logs/mnist_conversion_diff_threads/mnist_conversion_profile_interop_${num_interop_threads}_intraop_${num_threads}.log
	done
}

function profile_speech {
	cd ${BASE_DIR}/speech
	for model in lif lsnn lstm
	do
		for num_interop_threads in {${BASE_NUM_THREADS}..${MAX_NUM_THREADS}}
		do
			num_threads=${BASE_NUM_THREADS}
			./run_in_docker.sh python speech_profile.py --model ${model} --device cpu --num_interop_threads ${num_interop_threads} --num_threads ${num_threads} | tee ${BASE_DIR}/logs/speech_diff_threads/speech_${model}_interop_${num_interop_threads}_intraop_${num_threads}.log
		done

		for num_threads in {${BASE_NUM_THREADS}..${MAX_NUM_THREADS}}
		do
			num_interop_threads=${BASE_NUM_THREADS}
			./run_in_docker.sh python speech_profile.py --model ${model} --device cpu --num_interop_threads ${num_interop_threads} --num_threads ${num_threads} | tee ${BASE_DIR}/logs/speech_diff_threads/speech_${model}_interop_${num_interop_threads}_intraop_${num_threads}.log
		done
	done
}

profile mnist_stdp bindsnet:0.3.1 ${BASE_DIR} run_19
profile mnist_surrogate snntorch ${BASE_DIR} run_19
profile conversion bindsnet:0.3.1 ${BASE_DIR}/conversion run_19 "--job-dir logs"
for model in lif lsnn lstm
do
	profile speech norse ${BASE_DIR}/speech run_19/${model} "--model ${model} --device cpu"
done
