import random
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DQN(nn.Module):
    
    

    def __init__(self, n_observations, n_actions):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128, device=self.device)
        self.layer2 = nn.Linear(128, 128, device=self.device)
        self.layer3 = nn.Linear(128, n_actions, device=self.device)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
class DQN3(nn.Module):

    def __init__(self, n_observations, n_actions, layers, activation):
        # layers = int(layers)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(layers * 2)
        super(DQN3, self).__init__()
        self.list_layers = []
        self.layer1 = nn.Linear(n_observations, 128, device=self.device)
        for i in range(layers):
            
            layer = nn.Linear(128, 128, device=self.device)
            self.list_layers.append(layer)
            
        self.last_layer = nn.Linear(128, n_actions, device=self.device)
        self.activation = activation

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        
        # self.activation = "lrelu"
        if self.activation == "lrelu":
            for layer in self.list_layers:
                x = F.leaky_relu(layer(x))
        else:
            for layer in self.list_layers:
                x = F.relu(layer(x))
            
        return self.last_layer(x)
    
    
class Test(nn.Module):
    
    

    def __init__(self, n_observations, hidden_size, n_actions):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super(Test, self).__init__()
        self.layer1 = nn.Linear(n_observations, hidden_size, device=self.device)
        self.layer2 = nn.Linear(hidden_size, n_actions, device=self.device)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        return self.layer2(x)
    
# Python program to find 
# maximal Bipartite matching.
 
class GFG:
    def __init__(self,graph):
         
        # residual graph
        self.graph = graph 
        self.ppl = len(graph)
        self.jobs = len(graph[0])
 
    # A DFS based recursive function
    # that returns true if a matching 
    # for vertex u is possible
    def bpm(self, u, matchR, seen):
 
        # Try every job one by one
        for v in range(self.jobs):
 
            # If applicant u is interested 
            # in job v and v is not seen
            if self.graph[u][v] and seen[v] == False:
                 
                # Mark v as visited
                seen[v] = True
 
                '''If job 'v' is not assigned to
                   an applicant OR previously assigned 
                   applicant for job v (which is matchR[v]) 
                   has an alternate job available. 
                   Since v is marked as visited in the 
                   above line, matchR[v]  in the following
                   recursive call will not get job 'v' again'''
                if matchR[v] == -1 or self.bpm(matchR[v], 
                                               matchR, seen):
                    matchR[v] = u
                    print(matchR)
                    return True
                
        
        return False
 
    # Returns maximum number of matching 
    def maxBPM(self):
        '''An array to keep track of the 
           applicants assigned to jobs. 
           The value of matchR[i] is the 
           applicant number assigned to job i, 
           the value -1 indicates nobody is assigned.'''
        matchR = [-1] * self.jobs
         
        # Count of jobs assigned to applicants
        result = 0
        for i in range(self.ppl):
             
            # Mark all jobs as not seen for next applicant.
            seen = [False] * self.jobs
             
            # Find if the applicant 'u' can get a job
            if self.bpm(i, matchR, seen):
                result += 1
        return result
    
    
    
net1 = Test(2,4,2)
net2 = Test(2,4,2)



arrThress = torch.ones((4,2), device=net1.device) * 3

arrFours = torch.ones((4,2), device=net1.device) * 4
cut_location = 3

layer = 1

i = 0
# for key in [f"layer{layer}.weight", f"layer{layer}.bias"]:
i +=1 
state_dict1 =arrThress
state_dict2 = arrFours

nn_layer1 = state_dict1
nn_layer2 = state_dict2

print(nn_layer1)
print(nn_layer2)

shape = nn_layer1.shape
    
zeros_first = torch.zeros(shape, device=net1.device)  # Shape: (4, 2)
ones_first = torch.ones(shape, device=net2.device)

ones_first[cut_location:] = 0
zeros_first[cut_location:] = 1
    

first_half1 = nn_layer1 * ones_first
second_half1 = nn_layer1 * zeros_first

first_half2 = nn_layer2 * ones_first
second_half2 = nn_layer2 * zeros_first

state_dict1 = first_half1 + second_half2
state_dict2 = first_half2 + second_half1


print( state_dict1)
    
# n1_l1 = net1.state_dict()["layer1.weight"].shape
# # print()

# # input()

# # n1 = np.zeros((4, 2))
# # print(n1)
# # # n2 = np.zeros((128, 128))
# # input()

# arrThress = torch.ones(n1_l1) * 3

# print(arrThress)
# arrFours = torch.ones(4,2) * 4


# # Example array
# arr = torch.zeros(4, 2)  # Shape: (4, 2)
# arr2 = torch.ones(4,2)



# # Determine the index where the second half starts
# half_index = arr.shape[0] // 2

# half_index = 0

# # Add 1s to the second half of the array
# arr[half_index:] = 1
# arr2[half_index:] = 0

# print("Original array:")
# print(arr)
# print(arr2)

# ar3 = arrThress * arr + arrFours * arr2
# print(ar3)

# input()

# input()

# random_inputs = []
# for i in range(128):
#     random_tensor = torch.randn(80, device="cuda", dtype=torch.float32).cuda()
#     n1_tensor = F.relu(net1.layer1(random_tensor))
#     n2_tensor = F.relu(net2.layer1(random_tensor))
    
#     n1[i] = n1_tensor.detach().cpu().numpy()
#     n2[i] = n2_tensor.detach().cpu().numpy()


# # print(n1)
# # print(n2)

# # def standardize_data(data):
# #         mean = np.mean(data, axis=0)
# #         std_dev = np.std(data, axis=0)
# #         standardized_data = (data - mean) / std_dev
# #         return standardized_data

# # n1_data = standardize_data(n1)
# # n2_data = standardize_data(n2)

# # # n1_data.reshape(1, -1)

# # print("datas")
# # print(n1_data)
# # print(n2_data)

# scaler = StandardScaler()# Fit your data on the scaler object
# relu_layer_one = scaler.fit_transform(n1)
# relu_layer_two = scaler.fit_transform(n2)

# # print(8423942378947283)
# # print(relu_layer_one)
# # print(relu_layer_two)


    
    
# # print(standardize_data(n1_data))

# from sklearn.cross_decomposition import CCA
# # X = [[0., 0., 1.], [1.,0.,0.], [2.,2.,2.], [3.,5.,4.]]
# # Y = [[0.1, -0.2], [0.9, 1.1], [6.2, 5.9], [11.9, 12.3]]
# cca = CCA(n_components=128)
# cca.fit(relu_layer_two, relu_layer_two)
# X_c, Y_c = cca.transform(relu_layer_one, relu_layer_two)

# print(f"X_c {X_c}")


# # Sample array
# my_array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

# # List of indices to set to a certain value
# indices_to_set = [1, 3, 5]

# # Value to set the elements to
# value_to_set = 0

# # Set the elements at the specified indices to the desired value using NumPy indexing
# my_array[indices_to_set] = value_to_set

# def get_ordered_indices(wa, wb):

#     la = []
#     lb = []
    
#     la_indicies  = []
#     lb_indicies = []

#     for k in range(min(len(wa), len(wb))):
        
#         wa_mean = np.mean(wa[k])
#         wb_mean = np.mean(wb[k])
        
#         print(wa_mean)

#         wa[k][la_indicies] = wa_mean
#         wb[k][lb_indicies] = wb_mean
        
#         # print(wa[k])
        
        

        
#         sk_minus = abs(min(wa[k]) + min(wb[k]))
#         sk_plus = abs(max(wa[k]) + max(wb[k]))

#         la_index = lb_index = 0 

#         if sk_plus > sk_minus:
#             wa[k][la_indicies] = -np.inf
#             wb[k][lb_indicies] = -np.inf
#             la_index = np.argmax(wa[k])
#             lb_index = np.argmax(wb[k])
#         else:
#             wa[k][la_indicies] = np.inf
#             wb[k][lb_indicies] = np.inf
#             la_index = np.argmin(wa[k])
#             lb_index = np.argmin(wb[k])
            
            
#         la.append(la_index)
#         la_indicies.append(la_index)
#         lb.append(lb_index)
#         lb_indicies.append(lb_index)
        
#     # print(wa)
#     return la, lb

# la, lb = get_ordered_indices(X_c, Y_c)


# has_duplicates = len(la) != len(np.unique(la))
# hasd = len(lb) != len(np.unique(lb))

# print("Has duplicates:", has_duplicates)

# print("Has duplicates:", hasd)


                    
# def get_corr( relu_original_one, relu_original_two):
    
#         axis_number = 0
#         semi_matching = False
#         n = relu_original_one.shape[1]
#         # n = len(relu_original_one)
#         list_neurons_x = []
#         list_neurons_y = []
        
#         # scaler = StandardScaler()# Fit your data on the scaler object
#         # relu_layer_one = scaler.fit_transform(relu_original_one)
#         # relu_layer_two = scaler.fit_transform(relu_original_two)

#         corr_matrix_nn = np.empty((n,n))

#         for i in range(n):
#             for j in range(n):
#                 corr = np.corrcoef(relu_layer_one[:,i], relu_layer_two[:,j])[0,1]
#                 corr_matrix_nn[i,j] = corr

#         corr_matrix_nn[np.isnan(corr_matrix_nn)] = -1


#         # print(corr_matrix_nn)
#         corr_matrix_nn[np.isnan(corr_matrix_nn)] = -1
        
#         # print(corr_matrix_nn)
        
#         #argmax_columns = np.argmax(corr_matrix_nn, axis=axis_number)
#         argmax_columns = np.flip(np.argsort(corr_matrix_nn, axis=axis_number), axis=axis_number)
#         dead_neurons = np.sum(corr_matrix_nn, axis=axis_number) == n*(-1) # these are neurons that always output 0 (dead relu)
#         for index in range(n):
#             if dead_neurons[index] == False:
#                 if semi_matching:
#                     if axis_number == 0:
#                         list_neurons_y.append(index)
#                         list_neurons_x.append(argmax_columns[0,index])
#                     elif axis_number == 1:
#                         list_neurons_x.append(index)
#                         list_neurons_y.append(argmax_columns[index,0])
                        
#                 elif semi_matching == False:
                    
#                 # do not allow same matching
#                     for count in range(n):

#                         if axis_number == 0:
#                             if argmax_columns[count,index] not in list_neurons_x:
#                                 list_neurons_y.append(index)
#                                 list_neurons_x.append(argmax_columns[count,index])
#                                 break
#                         elif axis_number == 1:
#                             if argmax_columns[index,count] not in list_neurons_y:
#                                 list_neurons_x.append(index)
#                                 list_neurons_y.append(argmax_columns[index,count])
#                                 break
        
#         # randomly pair the unpaired neurons
#         for index in range(n):
#             if index not in list_neurons_x and len(list_neurons_x) < n:
#                 list_neurons_x.append(index)
#             if index not in list_neurons_y and len(list_neurons_y) < n:
#                 list_neurons_y.append(index)
        
#         return list_neurons_x, list_neurons_y
    

# # print(get_corr(n1_data, n2_data))

# state_dict = net1.state_dict()

# # print(state_dict)


# def swap_nodes(swap1, swap2, state_dict, layer):
    
#     # for keys in state_dict.keys():
#     #     print(keys)
#     # input()
    
#     if f"layer{layer + 1}.weight" in state_dict.keys():
#         keys = [f"layer{layer}.weight", f"layer{layer}.bias",f"layer{layer + 1}.weight"]
#     else:
#         keys = [f"layer{layer}.weight", f"layer{layer}.bias",f"last_layer.weight"]
        
#     # print(keys)
    
#     # print((swap1, swap2))

#     for key in keys:
#         tor = state_dict[key]
#         if key != keys[2]:
#             try:
#                 tor[[swap1,swap2]] = tor[[swap2,swap1]]
#             except:
#                 tor[:, [swap1,swap2]] = tor[:, [swap2,swap1]]
#         else:
#             for out in state_dict[keys[2]]:
#                 out[[swap1,swap2]] = out[[swap2,swap1]]
#     return state_dict


# def permute_layers(swaps, state_dict, layers):
#     for layer in layers:
#         already_swapped = []
#         to_change = {}
#         for index1, index2 in swaps:
#             if index1 in to_change.keys():
#                 index1 = to_change[index1]
#             if index1 != index2:
#                 if (index1, index2) not in already_swapped:
#                     print(index1, index2)
#                     if index1 != index2:
#                         if (index1, index2) not in already_swapped:
#                             state_dict = swap_nodes(index1, index2, state_dict, layer)
#                             to_change[index2] = index1
#                             already_swapped.append((index2, index1))
#     return state_dict

# swaps = ([(1,4), (2,3), (0,0)])

# state_dict = torch.tensor([0,4,3,2,1])

# state_dict2 = torch.tensor([[0,  0],
#         [4, 4],
#         [ 3,  3],
#         [2,2],
#         [1,1]])

# state_dict3 = torch.tensor([[0,4,3,2,1],
#        [0,4,3,2,1]])


# already_swapped = []
# to_change = {}
# for index1, index2 in swaps:
#     if index1 in to_change.keys():
#         index1 = to_change[index1]
#     if index1 != index2:
#         if (index1, index2) not in already_swapped:
#             print(index1, index2)
#             if index1 != index2:
#                 if (index1, index2) not in already_swapped:
                    
#                     tor = state_dict2
#                     for out in state_dict3:
#                         out[[index1,index2]] = out[[index2,index1]]
                
#                     to_change[index2] = index1
#                     already_swapped.append((index2, index1))
                    
                    
# # print(state_dict3)






# # Create a sample PyTorch tensor
# # tensor = torch.tensor([[1, 2, 3],
# #                        [4, 5, 6],
# #                        [7, 8, 9]], device="cuda")

# # # Define the indices of the rows you want to swap
# # row_index1 = 0  # Index of the first row to swap
# # row_index2 = 2  # Index of the second row to swap

# # tensor[[row_index1,row_index2]] = tensor[[row_index2,row_index1]]

# # print(tensor)


# # Create a sample NumPy array
# array = np.array([1, 2, 3, 4, 5, 6])

# indicies = []

# # Define your filter condition (e.g., ignore values less than 3)
# filter_condition = array >= 3


# # Find the index of the minimum value in the original array that satisfies the filter condition
# min_index = np.where(filter_condition, array, np.inf).argmin()

# print("Index of the minimum value satisfying the filter condition:", min_index)
# print("Minimum value:", array[min_index])


sav_cba =  -253.76126098632812
sav = np.array([-253.76126,     -111111111111111111,  -49.51717,   -891.3944,    -433.52487, -83.9612,    -323.76242,   -433.39606,   -706.8675,     -56.09465  ])
  
pi_div_a = sav_cba / sav

# pi_div_a =  [ 1,     94.68476,    5.1247125,  0.284679,   0.5853442,  3.0223637, 0.7837885,  0.5855182,  0.3589941,  4.523805 ]

top = np.exp(pi_div_a)



# # probability = top/bottom

infs = []
inf_count = 0
for i, el in enumerate(top): 
    if el == np.inf:
        print("inf")
        infs.append(i)
        top[i] = 0
        
if len(infs) != 0:
    inf_prob = 0.8 / len(infs)
    bottom = sum(top)

    probability = np.empty(top.shape)
    print(probability)
    for i, el in enumerate(top):
        if i in infs:
            probability[i] = (inf_prob)
        else:
            probability[i] = (top[i] * 0.2/bottom)
else:
    bottom = np.sum(top)
    probability = top/bottom
# top [  2.718282          inf 168.1258      1.3293353   1.795609   20.539783
#    2.1897523   1.7959214   1.4318882  92.18571  ]
# bottom inf