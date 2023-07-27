import numpy as np
import pandas as pd


class Client:
    input_data = None
    number_of_samples = None
    local_sum = None
    local_mean = None
    global_mean = None

    def __init__(self):
        pass

    def read_input(self, input_path):
        try:
            self.input_data = pd.read_csv(input_path, header=None, delimiter=',')
        except FileNotFoundError:
            print(f'File {input_path} could not be found.', flush=True)
            exit()
        except Exception as e:
            print(f'File could not be parsed: {e}', flush=True)
            exit()

    def compute_local_mean(self):
        self.local_mean = self.input_data.to_numpy().mean()
        print(f'Local mean: {self.local_mean}')
        self.weight = self.input_data.shape[1]
        self.weighted_mean = self.local_mean * self.weight
        print(f'Local weighted mean: {self.weighted_mean}')

    def set_global_mean(self, global_mean):
        self.global_mean = global_mean

    def write_results(self, output_path):
        f = open(output_path, "a")
        f.write(str(self.global_mean))
        f.close()


class Coordinator(Client):

    def compute_global_mean(self, weigthed_means, weights):
        return np.sum(weigthed_means) / np.sum(weights)
