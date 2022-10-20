LOG_FILE=$1
grep 'SNN accuracy\|Test set' ${LOG_FILE} | grep 'SNN' -B 1 | grep -o -E '[0-9][0-9]\.[0-9][0-9]' | awk '{ if (NR % 2 == 0) {snn=$1; snn_sum+=$1; print snn, ann, snn/ann"%", snn-ann"%"; sum+=snn-ann} else {ann=$1; ann_sum+=$1} } END {print snn_sum/NR*2, ann_sum/NR*2, sum/NR*2"%"}'
