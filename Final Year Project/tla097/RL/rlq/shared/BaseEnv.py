import numpy as np
from sklearn import preprocessing


class BaseEnv():
    def __init__(self) -> None:
        self.test_length = 100
        
        
    def calculate_accuracy(self, env):
        produced_list = [d['ID'] for d in env.arr]
        optimal_list = [d['ID'] for d in env.optimal]
        
        print(produced_list)
        print(optimal_list)
        
        produced_list = np.array(produced_list)
        optimal_list = np.array(optimal_list)
        
        produced_list = produced_list.reshape(-1, 1)
        optimal_list = optimal_list.reshape(-1, 1)
        
        scaler = preprocessing.MinMaxScaler()
        
        normalised_p_l = scaler.fit_transform(produced_list)
        normalised_o_l = scaler.fit_transform(optimal_list)
        # Calculate the mean squared error
        mae = np.mean(abs(normalised_p_l - normalised_o_l))

        print("Mean Absolute Error:", mae)
        return mae
    
    
    def calc_rank_point_average(self, optimal, lst):
        produced_list = [d['ID'] for d in lst]
        optimal_list = [d['ID'] for d in optimal]
        produced_list = np.array(produced_list)
        optimal_list = np.array(optimal_list)
        
        length = len(produced_list)
        total  = 0
        for el_index in range(length):
            res = 0
            for i in range(el_index, length):
                index_in_opt = produced_list[el_index]
                res += length - index_in_opt + 1
            total += res
            
        bottom = length**2 * (length + 1)/2
        print(total/bottom)
        return total/bottom
    
    def calc_normalised_rank_point_average(self, env):
        RPA_produced = self.calc_rank_point_average(env.optimal, env.arr)
        RPA_optimal = self.calc_rank_point_average(env.optimal, env.optimal)
        NRPA = RPA_produced/RPA_optimal
        return NRPA
    
    def calc_APFD(be, self):
        length = len(self.arr)
        result = 0
        for el_index in range(length):
            result += el_index * self.arr[el_index]["Verdict"]
        bottom = length * self.num_fails
        right = 1/(2*length)
        
        if bottom == 0:
            return 0
        
        
        
        
        return 1 - result/bottom + right
            