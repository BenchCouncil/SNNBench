for num_interop_threads in {1..12}
do
	./run_in_docker.sh python mnist_stdp_profile.py --num_interop_threads ${num_interop_threads} --num_threads 12 | tee logs/mnist_stdp_diff_threads/mnist_stdp_profile_interop_${num_interop_threads}_intraop_12.log
done

for num_threads in {1..12}
do
	./run_in_docker.sh python mnist_stdp_profile.py --num_interop_threads 12 --num_threads ${num_threads} | tee logs/mnist_stdp_diff_threads/mnist_stdp_profile_interop_12_intraop_${num_threads}.log
done
