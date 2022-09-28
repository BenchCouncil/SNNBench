for device in cpu cuda
do
	for model in lif lsnn lstm
	do
		./run_in_docker.sh python speech_profile.py --device ${device} --model ${model} --log "./${device}_${model}.pt.trace.json"
	done
done
