import pandas as pd
from .noiser import laplace_noise
class sampler:

    def __init__(self,data:pd.DataFrame=None, sample_fn=None) -> None:
        if not data:
            self._init_defualt_data()
        else:
            self._data = data

        self._sample_fn = sample_fn


    def _init_defualt_data(self):
        self._data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
                 header=None, names=['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                                     'marital-status', 'occupation', 'relationship', 'race', 'sex',
                                     'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
                                     'income'])
        
    def sample(self, noise_type='laplace',**dp_param):
        
        true_result = self._sample_fn(self._data)
        if noise_type == 'laplace':
            noised_result = true_result + self._laplace(**dp_param)
        elif noise_type == 'exponential':
            raise NotImplementedError
        else:
            raise NotImplementedError
        return noised_result
    
    def _laplace(self,**kwargs):
        noise = laplace_noise(**kwargs)
        return noise

    def _exponential(self,**kwargs):
        raise NotImplementedError