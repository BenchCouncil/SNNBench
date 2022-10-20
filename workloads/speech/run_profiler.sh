# for device in cpu cuda
for device in cpu
do
	for model in lif lsnn lstm
	do
		./run_in_docker.sh python speech_profile.py --device ${device} --model ${model} | tee $(pwd)/../logs/speech-inference-${model}-${device}.txt
	done
done
