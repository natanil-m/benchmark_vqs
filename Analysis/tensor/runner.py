# 
# !pip install tensorcircuit[tensorflow]
# !pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
# !pip install tensorcircuit[jax]

import tensorcircuit as tc
import math
import numpy as np
import tensorflow as tf
import sys

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

back = str(sys.argv[2])

# K=tc.set_backend("jax")
# K=tc.set_backend("tensorflow")
K=tc.set_backend(back)

# tc.set_dtype("float32")
# tc.set_dtype("complex128")
# tc.set_contractor("greedy")
# print(c.sample(allow_state=True, batch=1024, format="count_dict_bin"))

num_of_qubits=1+int(sys.argv[1])
max_repeat = 1 #100
iter_max = 300  #300
num_of_layers = 3
N = 2**(num_of_qubits-1) #TODO
normal_val = math.sqrt(1/N)

print('backend = ',back)
print('number of qubits = ',num_of_qubits-1)
print('num_of_layers = ',num_of_layers)

N = 2**(num_of_qubits-2)
normal_val = math.sqrt(1/N)
initial_state_phi1 = [math.sqrt(1/N)]*(N-1) + [0]*N + [math.sqrt(1/N)] 
initial_state_0_phi1  = initial_state_phi1 + [0]*len(initial_state_phi1)



"""# New Section

# New Section
"""

#=================================
def mcnot(circuit, control_qubits, target_qubit):
    num_controls = len(control_qubits)

    # Apply Toffoli gates
    for i in range(num_controls - 2):
        circuit.CNOT(control_qubits[i], control_qubits[i + 1])
        circuit.CNOT(control_qubits[i], target_qubit)
        circuit.CNOT(control_qubits[i + 1], target_qubit)
    
    # Apply last Toffoli gate
    circuit.CNOT(control_qubits[num_controls - 2], control_qubits[num_controls - 1])
    circuit.CNOT(control_qubits[num_controls - 2], target_qubit)
#================================================================


eps_val_q = 1/math.sqrt(2**num_of_qubits)/100
eps_val = min(1e-10, eps_val_q)
tiny_change_threshold = 1e-4
cnt_threshold_no_change = 5

def layer_t3_no_HT(circuit,theta,offset, num_of_qubits, qubit_posi):
    # type-2 layer
    # length of theta: (num_of_qubits-1)*2
    # length of qubit_posi: num_of_qubits-1
    # number of wires: num_of_qubits

    for i in range(num_of_qubits-1):
        circuit.ry(qubit_posi[i],theta=theta[offset+i])    
    for i in np.arange(0, num_of_qubits-2, 2):
        circuit.CNOT(qubit_posi[i],qubit_posi[i+1]) # CNOT struct3
    for i in range(num_of_qubits-1):
        circuit.ry(qubit_posi[i],theta=theta[offset+i+num_of_qubits-1])
    for i in np.arange(1, num_of_qubits-2, 2):
        circuit.CNOT(qubit_posi[i],qubit_posi[i+1]) # CNOT struct3
    circuit.CNOT(qubit_posi[-1],qubit_posi[0]) # CNOT struct3

# Tested
# c = tc.Circuit(num_of_qubits-1)
# layer_t3_no_HT(circuit=c,theta=[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],num_of_qubits=num_of_qubits,qubit_posi=list(range(num_of_qubits-1)))
# cq = c.to_qiskit()
# print(cq.draw())

def layer_t3_with_HT(circuit,theta,offset,num_of_qubits):
    # type-2 layer
    # length of theta: (num_of_qubits-1)*2
    # number of wires: num_of_qubits


    for i in range(num_of_qubits-1):
        circuit.cry(0,i+1,theta=theta[offset+i])    
    for i in np.arange(0, num_of_qubits-2, 2):
        circuit.toffoli(0,i+1,i+2) # CCNOT struct3
        
    for i in range(num_of_qubits-1):
        circuit.cry(0, i+1,theta=theta[offset+i+num_of_qubits-1])
    for i in np.arange(1, num_of_qubits-2, 2):
        circuit.toffoli(0,i+1,i+2) # CCNOT struct3
    circuit.toffoli(0,num_of_qubits-1, 1) # CCNOT struct3

# Tested
# c = tc.Circuit(num_of_qubits)
# layer_t3_with_HT(circuit=c,theta=[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],num_of_qubits=num_of_qubits)
# cq = c.to_qiskit()
# print(cq.draw())

def quantum_circuit_no_Z(num_of_qubits,theta,start_state=None):
    # initiate state vector |phi_1>
    circuit = tc.Circuit(num_of_qubits,inputs=start_state)
    # circ_phi.initialize(start_state, circ_phi.qubits)
    circuit.H(0)
    # for i in range(2,num_of_qubits):
    #     circuit.H(i)
    # circuit.CNOT(2,1) #TODO
    # for theta_i in theta:
    #     layer_t3_with_HT(circuit,theta_i, num_of_qubits)   
    mcnot(circuit=circuit,control_qubits=[i for i in range(2, num_of_qubits)],target_qubit=1)
    for i in range(num_of_layers):
        offset = i*2*(num_of_qubits-1)
        layer_t3_with_HT(circuit,theta,offset, num_of_qubits)   
    
    circuit.H(0)
    exp_val=circuit.expectation([tc.gates.z(), [0]])
    return K.real(exp_val)
    # return circuit #for test

# Tested
# c=quantum_circuit_no_Z(theta=[[3.1,0.1,4.5,0.1,8.2,0.2,2.1,0.1]],num_of_qubits=num_of_qubits)
# cq = c.to_qiskit()
# print(cq.draw())

def quantum_circuit_Z(num_of_qubits,theta,start_state=None):
    # initiate state vector |phi_1>
    circuit = tc.Circuit(num_of_qubits,inputs=start_state)
    # circ_phi.initialize(start_state, circ_phi.qubits)
    circuit.H(0)
    # for i in range(2,num_of_qubits):
    #     circuit.H(i)
    # circuit.CNOT(2,1) #TODO
    # for theta_i in theta:
    #     layer_t3_with_HT(circuit,theta_i, num_of_qubits)   
    for i in range(num_of_layers):
        offset = i*2*(num_of_qubits-1)
        layer_t3_with_HT(circuit,theta,offset, num_of_qubits)   

    circuit.cz(0,1)
    circuit.H(0)
    exp_val=circuit.expectation([tc.gates.z(), [0]])
    # print(exp_val)
    return K.real(exp_val)
    # return circuit #for test

# # Tested
# c=quantum_circuit_Z(theta=[[3.1,0.1,2,0.1,8.2,0.2,2.1,0.1]],num_of_qubits=num_of_qubits)
# cq = c.to_qiskit()
# print(cq.draw())

def quantum_circuit_no_HT_return_state(num_of_qubits,theta,start_state=None):
    # initiate state vector |phi_1>
    circuit = tc.Circuit(num_of_qubits-1,inputs=start_state)
    # circ_phi.initialize(start_state, circ_phi.qubits)
    # for i in range(1,num_of_qubits-1):
    #         circuit.H(i)
    # circuit.CNOT(1,0)#TODO
      
    # for theta_i in theta:
    #     layer_t3_no_HT(circuit=circuit,theta=theta_i, num_of_qubits=num_of_qubits,qubit_posi=list(range(num_of_qubits-1)))    
    for i in range(num_of_layers):
      offset = i*2*(num_of_qubits-1)
      layer_t3_no_HT(circuit=circuit,theta=theta,offset=offset, num_of_qubits=num_of_qubits,qubit_posi=list(range(num_of_qubits-1)))   
  
  
    # print(circuit.state())
    return circuit.state()
    # return circuit

# # Tested
# c=quantum_circuit_no_HT_return_state(theta=[[3.1,0.1,2,0.1,8.2,0.2,2.1,0.1]],num_of_qubits=num_of_qubits)
# cq = c.to_qiskit()
# print(cq.draw())

def objective_fn(theta):
    val1_1 = quantum_circuit_no_Z(theta=theta,num_of_qubits=num_of_qubits,start_state=initial_state_0_phi1)
    val1_2 = quantum_circuit_Z(theta=theta,num_of_qubits=num_of_qubits,start_state=initial_state_0_phi1)
    val1_1 = val1_1/normal_val
    val1_2 = val1_2/normal_val
    obj = -0.5*(val1_1 - val1_2)
    return obj

import datetime
# from qiskit.algorithms.optimizers import SPSA
from tensorflow.python.ops.numpy_ops import np_config
# np_config.enable_numpy_behavior()

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import numpy as np
import scipy.optimize as optimize
import tensorflow as tf
import tensorcircuit as tc



eps_val_q = 1/math.sqrt(2**num_of_qubits)/100
eps_val = min(1e-10, eps_val_q)
tiny_change_threshold = 1e-4
cnt_threshold_no_change = 5
start_time = datetime.datetime.now()

shape1 = (num_of_layers, 2*(num_of_qubits))
theta = np.random.uniform(0, 2*math.pi, size=shape1)
theta = theta.reshape(-1)

expval = quantum_circuit_no_Z(theta=theta,num_of_qubits=num_of_qubits,start_state=initial_state_0_phi1)
print('expval = ',expval)


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
#     def __init__(self, N : int):
#         self.N = N
#         self.values = []
#         self.i = 0

#     def __call__(self, nfev, parameters, value, stepsize, accepted) -> bool:
        
#         self.values.append(value)
#         if(self.i%1==0):
#             print('==================================')
#             temp_theta = np.array(parameters)
#             temp_theta_shaped = temp_theta.reshape(shape1)
#             print('iteration:',self.i,'\n### theta: ',temp_theta_shaped, '\n### obj=',value)
#             if(value<-0.98):
#                 return True
#         self.i+=1
#         if len(self.values) > self.N:
#             last_values = self.values[-self.N:]
#             pp = np.polyfit(range(self.N), last_values, 1)
#             slope = pp[0] / self.N

#             if slope > 0:
#                 return True
        
#         return False

# for rep in range(1,max_repeat+1):
#     print(f'\n\nrep={rep}', end=' \n ')
#     print('Wait to initialize ...')


#     shape2 = [num_of_layers* 2*(num_of_qubits-1)]
#     theta = np.random.uniform(0, 2*math.pi, size=shape2)

#     result = quantum_circuit_no_HT_return_state(theta=theta,num_of_qubits=num_of_qubits,start_state=initial_state_phi1)
#     # print(result)
#     # state_last = result.get_statevector().tolist()[4]
#     state_last = result[-1]
#     prb_last = np.linalg.norm(state_last)
#     print("++++++++++++++++++++++++++++++++++++++++++++++++")
#     print('prob_good_element = ',prb_last)
#     print("++++++++++++++++++++++++++++++++++++++++++++++++")
#     end_time = datetime.datetime.now()
#     duration = end_time - start_time
#     duration_in_s = duration.total_seconds()
#     print(f'time consumed: {duration_in_s}s')
#     raise ValueError("Finished")


#     # spsa = SPSA(maxiter=iter_max,blocking=True, termination_checker=TerminationChecker(100))
#     # shape1 = (num_of_layers, 2*(num_of_qubits))
#     # theta = np.random.uniform(0, 2*math.pi, size=shape1)
#     # print(theta)
#     # theta = theta.reshape(-1)
#     # result = spsa.minimize(x0=theta,fun=objective_fn)

#     # spsa = SPSA(maxiter=iter_max,blocking=True, termination_checker=TerminationChecker(100))
#     # shape1 = (num_of_layers, 2*(num_of_qubits))
#     # theta = tf.random.uniform(shape=shape1, minval=0, maxval=2*math.pi)
#     # print(theta)
#     # theta = tf.reshape(theta, shape=(-1,))
#     # result = spsa.minimize(x0=theta, fun=objective_fn)



#     # # energy_val_grad_jit = K.jit(objective_fn)
#     # energy_val_grad = K.value_and_grad(objective_fn)
#     # # learning_rate = 2e-2
#     # learning_rate = 0.02
#     # opt = K.optimizer(tf.keras.optimizers.SGD(learning_rate))
#     # def grad_descent(params, i):
#     #     # val, grad = energy_val_grad_jit(params)
#     #     val, grad = energy_val_grad(params)
#     #     params = opt.update(grad, params)
#     #     if i % 1 == 0:
#     #         print(f"i={i}, energy={val}")
            
#     #     if i % 5 == 0:
#     #         ## display the amplified state
#     #         result = quantum_circuit_no_HT_return_state(theta=params,num_of_qubits=num_of_qubits,start_state=start_state)
#     #         # print(result)
#     #         # state_last = result.get_statevector().tolist()[4]
#     #         state_last = result[-1]
#     #         prb_last = np.linalg.norm(state_last)
#     #         print("++++++++++++++++++++++++++++++++++++++++++++++++")
#     #         print('prob_good_element = ',prb_last)
#     #         print("++++++++++++++++++++++++++++++++++++++++++++++++")

#     #     return params,val



#     global itr
#     itr = 0
#     def callback(theta):
#       global itr
#       if(itr%1==0):
#             print('iteration: ',itr)
#             result = quantum_circuit_no_HT_return_state(theta=theta,num_of_qubits=num_of_qubits,start_state=initial_state_phi1)
#             state_last = result[-1]
#             prb_last = np.linalg.norm(state_last)
#             print("++++++++++++++++++++++++++++++++++++++++++++++++")
#             print('prob_good_element = ',prb_last)
#             print("++++++++++++++++++++++++++++++++++++++++++++++++")
#             if prb_last>0.98:
#               end_time = datetime.datetime.now()
#               duration = end_time - start_time
#               duration_in_s = duration.total_seconds()
#               print(f'time consumed: {duration_in_s}s')
#               raise ValueError("Finished")
#       itr+=1

       
  

#     shape1 = (num_of_layers, 2*(num_of_qubits-1))
#     shape2 = [num_of_layers* 2*(num_of_qubits-1)]
#     # f_scipy = tc.interfaces.scipy_interface(objective_fn, shape=shape2, jit=True)
#     f_scipy = tc.interfaces.scipy_interface(objective_fn, shape=shape2, jit=True)
#     params = K.implicit_randn(shape=shape2)
#     r = optimize.minimize(f_scipy,params, method="L-BFGS-B",callback=callback, jac=True)
#     print(r)
#     best_theta = r.x

  
#     # best_theta = np.array(result.x)
#     # best_theta_shaped = best_theta.reshape(shape1)
#     # best_theta_shaped = params

#     ## display the amplified state
#     result = quantum_circuit_no_HT_return_state(theta=best_theta,num_of_qubits=num_of_qubits,start_state=initial_state_phi1)
#     # print(result)
#     # state_last = result.get_statevector().tolist()[4]
#     state_last = result[-1]
#     prb_last = np.linalg.norm(state_last)
#     print("++++++++++++++++++++++++++++++++++++++++++++++++")
#     print('prob_good_element = ',prb_last)
#     print("++++++++++++++++++++++++++++++++++++++++++++++++")







