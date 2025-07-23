# Overview
In this repo, I produced several machine learning algorithms that could run on a quantum kernel.\
The kernel has made via simulating circuits on Aer and Superstaq, Infleqtion's noisy simulator. \
I found that with current quantum computer infrastructure, quantum machine learning is not feasible given the tiny datasets that we could use. For example, on Superstaq's servers, it took 10 minutes to run a training=12, testing=10 shots=1000 to simulate a quantum kernel matrix. Naturally this scales quadratically with the size of the data set, with atleast 100 training samples needed to produce accuracies above 80%. Also the simulators I used were a lot faster than an actual QPU.\
An actual QPU requires qubit setup which on Superstaq's infrastructor results in roughly 10 shots/second, which would severly limit any realistic testing.\

