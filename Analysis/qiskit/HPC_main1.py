from qiskit.algorithms.optimizers import SPSA
import datetime
import numpy as np
from qiskit import QuantumCircuit
from qiskit.opflow import CircuitOp
from qiskit.opflow import CircuitStateFn
from qiskit.opflow.state_fns import StateFn
from qiskit.opflow.expectations import MatrixExpectation
from qiskit.opflow.converters import CircuitSampler
from qiskit.providers.aer import QasmSimulator, AerSimulator
from qiskit import QuantumCircuit, execute
from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit.opflow import PauliExpectation
import math
from qiskit.utils import algorithm_globals
import sys
algorithm_globals.massive = True
# ======================================================
# print('qiskit version = ', qiskit.__version__)

num_of_qubits = int(sys.argv[1])
start_state = None  # None consider 111111...1
max_repeat = 1  # 100
iter_max = 300  # 300
num_of_layers = 3
N = 2**(num_of_qubits-1)  # TODO
normal_val = math.sqrt(1/N)
seed = 80
np.random.seed(seed)
use_sampler = False
n_shots = 6000
# Available methods are: ('automatic', 'statevector', 'density_matrix', 'stabilizer', 'matrix_product_state', 'extended_stabilizer', 'unitary', 'superop','tensor_network')"
method = str(sys.argv[2])
method_result = str(sys.argv[2])
device = 'CPU'
# backend = AerSimulator(method=method, device=device, cuStateVec_enable=True,
#                       batched_shots_gpu=True, blocking_enable=True)
backend = AerSimulator(method=method, device=device)



#===================================================
import subprocess
import tracemalloc

# GPU mem usage
def get_gpu_memory_usage():
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'], capture_output=True)
    output = result.stdout.decode('utf-8').strip()
    memory_usages = output.split('\n')
    total_memory_usage = sum(map(int, memory_usages))
    return total_memory_usage
# CPU mem Usage
tracemalloc.start()




# ======================================================
print(backend.available_devices())
print(backend.available_methods())
print('num_of_qubits =', num_of_qubits)
print("start_state =", start_state)
print("max_repeat =", max_repeat)
print("iter_max =", iter_max)
print("seed =", seed)
print("use_sampler = ", use_sampler)
print("n_shots = ", n_shots)
print('method = ', method)
print('method_result = ', method_result)
print("device =", device)
# ======================================================


def layer_t3_no_HT(circuit, theta, num_of_qubits):
    # type-2 layer
    # length of theta: (num_of_qubits-1)*2
    # number of wires: num_of_qubits
    for i in range(num_of_qubits):
        if(type(theta[i]).__name__ == 'ArrayBox'):
            circuit.ry(float(theta[i]._value), i)  # TODO
        else:
            circuit.ry(float(theta[i]), i)
    for i in np.arange(0, num_of_qubits-1, 2):
        circuit.cnot(i, i+1)  # CCNOT struct3

    for i in range(num_of_qubits):
        if(type(theta[i]).__name__ == 'ArrayBox'):
            circuit.ry(float(theta[i+num_of_qubits]._value), i)  # TODO
        else:
            circuit.ry(float(theta[i+num_of_qubits]), i)  # TODO

    for i in np.arange(1, num_of_qubits-1, 2):
        circuit.cnot(i, i+1)  # CCNOT struct3
    circuit.cnot(num_of_qubits-1, 0)  # CCNOT struct3
# ======================================================


def quantum_circuit_no_Z(num_of_qubits, theta, start_state=None):
    # initiate state vector |phi_1>
    circ_phi = QuantumCircuit(num_of_qubits)
    # circ_phi.initialize(start_state, circ_phi.qubits)
    for i in range(1, num_of_qubits):
        circ_phi.h(i)
    circ_phi.mcx([i for i in range(1, num_of_qubits)], 0)
    circ_phi = CircuitStateFn(circ_phi)
    # operator
    circ_expval_M = QuantumCircuit(num_of_qubits)
    for theta_i in theta:
        layer_t3_no_HT(circ_expval_M, theta_i, num_of_qubits)
    circ_expval_M = CircuitOp(circ_expval_M)

    # define the state to sample
    measurable_expression = StateFn(
        circ_expval_M, is_measurement=True).compose(circ_phi)
    if use_sampler:
        #     simulator = QasmSimulator(method=method)
        #     sampler_exact = CircuitSampler(simulator).convert(measurable_expression)
        #     return sampler_exact.eval().real
        q_instance = QuantumInstance(backend, shots=n_shots)
        # convert to expectation value
        expectation = PauliExpectation().convert(measurable_expression)
        # get state sampler (you can also pass the backend directly)
        sampler = CircuitSampler(q_instance).convert(expectation)
        return sampler.eval().real
    else:
        expectation_exact = MatrixExpectation().convert(measurable_expression)
        return expectation_exact.eval().real


def quantum_circuit_Z(num_of_qubits, theta, start_state=None):
    # initiate state vector |phi_1>
    circ_phi = QuantumCircuit(num_of_qubits)
    # circ_phi.initialize(start_state, circ_phi.qubits)
    for i in range(1, num_of_qubits):
        circ_phi.h(i)
    circ_phi.mcx([i for i in range(1, num_of_qubits)], 0)
    circ_phi = CircuitStateFn(circ_phi)
    # operator
    circ_expval_M = QuantumCircuit(num_of_qubits)
    for theta_i in theta:
        layer_t3_no_HT(circ_expval_M, theta_i, num_of_qubits)
    circ_expval_M.z(0)
    circ_expval_M = CircuitOp(circ_expval_M)

    # define the state to sample
    measurable_expression = StateFn(
        circ_expval_M, is_measurement=True).compose(circ_phi)
    if use_sampler:
        #     simulator = QasmSimulator(method=method)
        #     sampler_exact = CircuitSampler(simulator).convert(measurable_expression)
        #     return sampler_exact.eval().real
        q_instance = QuantumInstance(backend, shots=n_shots)
        # convert to expectation value
        expectation = PauliExpectation().convert(measurable_expression)
        # get state sampler (you can also pass the backend directly)
        sampler = CircuitSampler(q_instance).convert(expectation)
        return sampler.eval().real
    else:
        expectation_exact = MatrixExpectation().convert(measurable_expression)
        return expectation_exact.eval().real
# ======================================================


def quantum_circuit_no_HT_return_state(num_of_qubits, theta, start_state=None):
    # initiate state vector |phi_1>
    circ_phi = QuantumCircuit(num_of_qubits)
    # circ_phi.initialize(start_state, circ_phi.qubits)
    for i in range(1, num_of_qubits):
        circ_phi.h(i)
    circ_phi.mcx([i for i in range(1, num_of_qubits)], 0)

    for theta_i in theta:
        layer_t3_no_HT(circuit=circ_phi, theta=theta_i,
                       num_of_qubits=num_of_qubits)
    circ_phi.save_statevector()
    backend = QasmSimulator(method=method_result)  # statevector_gpu
    job = execute(circ_phi, backend)
    job_result = job.result()

    return job_result


# ======================================================
shape1 = (num_of_layers, 2*(num_of_qubits))


def objective_fn(theta):
    theta = theta.reshape(shape1)
    val1_1 = quantum_circuit_no_Z(
        theta=theta, num_of_qubits=num_of_qubits, start_state=start_state)
    val1_2 = quantum_circuit_Z(
        theta=theta, num_of_qubits=num_of_qubits, start_state=start_state)
    val1_1 = val1_1/normal_val
    val1_2 = val1_2/normal_val
    obj = -0.5*(val1_1 - val1_2)
    return obj

# ======================================================


eps_val_q = 1/math.sqrt(2**num_of_qubits)/100
eps_val = min(1e-10, eps_val_q)
tiny_change_threshold = 1e-4
cnt_threshold_no_change = 5
start_time = datetime.datetime.now()


shape1 = (num_of_layers, 2*(num_of_qubits))
theta = np.random.uniform(0, 2*math.pi, size=shape1)
# theta = theta.reshape(-1)

expval=quantum_circuit_no_Z(
        theta=theta, num_of_qubits=num_of_qubits, start_state=start_state)
print("expval = ",expval)

end_time = datetime.datetime.now()
duration = end_time - start_time
duration_in_s = duration.total_seconds()
print(f'time consumed: {duration_in_s}s')


print('CPU Memory usage :',tracemalloc.get_traced_memory())
tracemalloc.stop()
gpu_memory_usage = get_gpu_memory_usage()
print(f"GPU memory usage: {gpu_memory_usage} MiB")

print('done')












# class TerminationChecker:
#     def __init__(self, N: int):
#         self.N = N
#         self.values = []
#         self.i = 0

#     def __call__(self, nfev, parameters, value, stepsize, accepted) -> bool:

#         self.values.append(value)
#         if(self.i % 10 == 0):
#             print('==================================')
#             temp_theta = np.array(parameters)
#             temp_theta_shaped = temp_theta.reshape(shape1)
#             print('iteration:', self.i, '\n### theta: ',
#                   temp_theta_shaped, '\n### obj=', value)
#             if(value < -0.98):
#                 return True
#         self.i += 1
#         if len(self.values) > self.N:
#             last_values = self.values[-self.N:]
#             pp = np.polyfit(range(self.N), last_values, 1)
#             slope = pp[0] / self.N

#             if slope > 0:
#                 return True

#         return False


# for rep in range(1, max_repeat+1):
#     print(f'\n\nrep={rep}', end=' \n ')
#     print('Wait to initialize ...')
#     spsa = SPSA(maxiter=iter_max, blocking=True,
#                 termination_checker=TerminationChecker(100))
#     shape1 = (num_of_layers, 2*(num_of_qubits))
#     theta = np.random.uniform(0, 2*math.pi, size=shape1)
#     theta = theta.reshape(-1)
#     result = spsa.minimize(x0=theta, fun=objective_fn)
#     best_theta = np.array(result.x)
#     best_theta_shaped = best_theta.reshape(shape1)
#     # break
#     # display the amplified state
#     result = quantum_circuit_no_HT_return_state(
#         theta=best_theta_shaped, num_of_qubits=num_of_qubits, start_state=start_state)
#     # print(result)
#     state_last = result.get_statevector().tolist()[-1]
#     prb_last = np.linalg.norm(state_last)
#     print("++++++++++++++++++++++++++++++++++++++++++++++++")
#     print('prob_good_element = ', prb_last)
#     print("++++++++++++++++++++++++++++++++++++++++++++++++")

# end_time = datetime.datetime.now()
# duration = end_time - start_time
# duration_in_s = duration.total_seconds()
# print(f'time consumed: {duration_in_s}s')
# # ======================================================
