# SNNBench: An End-to-End AI-Oriented Spiking Neural Network Benchmark

SNNBench is the first end-to-end AI-oriented Spiking Neural Network (SNN) benchmark that covers the processing stages of training and inference, and includes accuracy information. Focusing on two typical AI applications, image classification and speech recognition, SNNBench provides nine workloads that consider the unique characteristics of both SNNs and AI.

## Features

- Covers the processing stages of training and inference
- Contains accuracy information
- Focuses on two typical AI applications: image classification and speech recognition
- Provides nine workloads considering the characteristics of SNNs and AI
- Evaluates workloads on both CPU and GPU

## Workloads

SNNBench workloads consider the following characteristics:

- Dynamics of spiking neurons
- Learning paradigms, including supervised and unsupervised learning
- Learning rules, including STDP, backpropagation and conversion
- Connection types, like fully connected layers
- Accuracy

## Usage

We recommend using Docker to run these workloads.

### Step 1: Build the Docker Images
Navigate to the `docker` directory and run the following command:
```
./build_docker.sh
```

### Step 2: Run Each Workload

#### In the `workloads` directory:
- For Image-STDP, run:
```
./run_in_docker.sh python mnist_stdp.py
```
- For Image-BackProp, run:
```
./run_in_docker.sh python mnist_surrogate.py
```

#### In the `workloads/conversion` directory:
- For Image-Conversion, run:

  - Train:
  ```
  ./run_in_docker.sh python train_mlp.py --job-dir logs
  ```
  - Infer:
  ```
  ./run_in_docker.sh python snn_inference.py --job-dir logs --results-file ann
  ```

#### In the `workloads/speech` directory:

- For Speech-LIF, run:
```
./run_in_docker.sh python speech.py --model lif
```
- For Speech-LSNN, run:
```
./run_in_docker.sh python speech.py --model lsnn
```
- For Speech-LSTM, run:
```
./run_in_docker.sh python speech.py --model lstm
```

## Publication

Tang, Fei, and Wanling Gao. "SNNBench: End-to-end AI-oriented spiking neural network benchmarking." BenchCouncil Transactions on Benchmarks, Standards and Evaluations (2023): 100108.

If you use SNNBench, please cite:

```
@article{tang2023snnbench,
    title={SNNBench: End-to-end AI-oriented spiking neural network benchmarking},
    author={Tang, Fei and Gao, Wanling},
    journal={BenchCouncil Transactions on Benchmarks, Standards and Evaluations},
    pages={100108},
    year={2023},
    publisher={Elsevier}
}
```

## Contributing

We welcome contributions to SNNBench! Please feel free to submit issues, create pull requests, or get in touch with the maintainers to discuss potential improvements or new features.

## License

SNNBench is released under the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).
