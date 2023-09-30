# Comparison of Quantum Simulators for Variational Quantum Search: A Benchmark Study

### This [benchmarking paper](https://arxiv.org/abs/2309.05924) has been accepted by 27th Annual IEEE High Performance Extreme Computing Conference (HPEC) 2023.

## Description
This code presents a comprehensive performance analysis of [Variational Quantum Search (VQS)](https://arxiv.org/abs/2212.09505), with a specific focus on its quantum simulation. The study explores the impact of different quantum frameworks, backends, and hardware devices on the scalability and efficiency of VQS. By analyzing various metrics such as runtime and memory usage, this research sheds light on the strengths and limitations of each quantum framework. The findings provide valuable insights for researchers and developers working on high-performance extreme computing with variational quantum algorithms.

## Features
- In-depth performance analysis of Variational Quantum Search
- Evaluation of quantum frameworks, backends, and hardware devices
- Analysis of scalability and efficiency
- Assessment of runtime and memory usage
- Insights into the strengths and limitations of each quantum framework

## Results
To access detailed data for the benchmarking graphs, please refer to the Analysis/Tables.docx document within this repository.

All tests were conducted on the [NCSA Delta](https://wiki.ncsa.illinois.edu/display/DSC/Delta+User+Guide) supercomputer. (GPU=A100x8)

![Runtime](https://github.com/natanil-m/benchmark_vqs/raw/main/Analysis/Results/runtime.png)

![Memory Usage](https://github.com/natanil-m/benchmark_vqs/raw/main/Analysis/Results/memory_usage.png)

Time and memory consumed by different simulators to obtain the expectation value of observable 〈Z1〉 in the VQS algorithm for different numbers of qubits. Note: Pennylane (CPU), Pennylane (GPU), and TensorCircuit (CPU) reached their memory limit for more than 29, 30, and 28 qubits, respectively.  Qiskit (CPU) and Cirq encounter errors when calculating the exact expectation value for more than 15 and 28 qubits, respectively.  Qulacs and Project Q encounter time limit when calculating the exact expectation value for more than 30 and 16 qubits, respectively.  


## Contributing
Contributions to the VQS_benchmark project are welcome and encouraged! If you would like to contribute to this project, please follow these guidelines:

1. Fork the repository and create your own branch for your feature or bug fix.
2. Commit your changes and provide clear commit messages.
3. Push your changes to your forked repository.
4. Submit a pull request to the main repository, explaining the changes you have made.

Please ensure that your contributions align with the project's scope and follow the coding conventions and best practices used in this codebase.

By contributing to this project, you agree that your contributions will be licensed under the terms of the [MIT License](LICENSE).

Thank you for considering contributing to VQS_benchmark. Your contributions are highly appreciated!


## License
This project is licensed under the terms of the MIT License. See the [LICENSE](LICENSE) file for more information.

## Contact
For any inquiries or questions, please feel free to reach out to me at [ms69@alfred.edu](mailto:ms69@alfred.edu).

