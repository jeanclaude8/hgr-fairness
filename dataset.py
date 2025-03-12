from causalgen import Generator
import numpy as np
import torch
from torch.utils.data import Dataset

class DataGenerator():
    def __init__(self, num_train, num_test):
        self.num_train = num_train
        self.num_test = num_test

    def generate_not_binormial(self, seed, hidden, f_e, f_i, f_y):
        #TrainSet
        dg = Generator(seed=seed)   
        UY = dg.uniform(low=0, high=1, name='UY', hidden=True)
        UE = dg.normal(mu = 0, sigma = 1, name='UE', hidden = True)
        UI = dg.normal(mu = 0, sigma = 1, name='UI', hidden = True)
        B = dg.custom(lambda s : np.random.binomial(n = 1, p=0.5, size = s), name='B')
        E = dg.descendant(lambda B, UE: f_e(UE, B), name='E')  
        I = dg.descendant(lambda B, UI: f_i(UI, B), name='I')
        Y = dg.descendant(lambda UY, I, E : f_y(UY,I, E), name='Y')
        Y_hat = dg.descendant(lambda UY, UI, UE : f_y(UY,UI, UE), name='Y-hat', hidden = True)

        df_train = dg.generate(num=self.num_train, hidden=hidden)
        df_train.to_csv('data_train.csv', index=False)

        dg.visualize()

        #TestSet
        dg = Generator(seed=seed)   
        UY = dg.uniform(low=0, high=1, name='UY', hidden=True)
        UE = dg.normal(mu = 0, sigma = 1, name='UE', hidden = True)
        UI = dg.normal(mu = 0, sigma = 1, name='UI', hidden = True)
        B = dg.custom(lambda s : np.random.binomial(n = 1, p=0.5, size = s), name='B', hidden=True)
        E = dg.descendant(lambda B, UE: f_e(UE, B), name='E')  
        I = dg.descendant(lambda B, UI: f_i(UI, B), name='I')
        Y = dg.descendant(lambda UY, I, E : f_y(UY,I, E), name='Y')
        Y_hat = dg.descendant(lambda UY, UI, UE : f_y(UY,UI, UE), name='Y-hat', hidden = True)
    
        df_test = dg.generate(num=self.num_test, hidden=hidden)
        df_test.to_csv('data_test.csv', index=False)

        dg.visualize()

        return df_train, df_test
    
    def generate_binormial(self, seed, hidden, f_e, f_i, f_y):
         #TrainSet
        dg = Generator(seed=seed)   
        UY = dg.uniform(low=0, high=1, name='UY', hidden=True)
        UE = dg.normal(mu = 0, sigma = 1, name='UE', hidden = True)
        UI = dg.normal(mu = 0, sigma = 1, name='UI', hidden = True)
        B = dg.normal(mu = 0, sigma = 1, name='B')
        E = dg.descendant(lambda B, UE: f_e(UE, B), name='E')  
        I = dg.descendant(lambda B, UI: f_i(UI, B), name='I')
        Y = dg.descendant(lambda UY, I, E : f_y(UY,I, E), name='Y')
        Y_hat = dg.descendant(lambda UY, UI, UE : f_y(UY,UI, UE), name='Y-hat', hidden = True)

        df_train = dg.generate(num=self.num_train, hidden=hidden)
        df_train.to_csv('data_train_normal.csv', index=False)

        dg.visualize()

        #TestSet
        dg = Generator(seed=seed)   
        UY = dg.uniform(low=0, high=1, name='UY', hidden=True)
        UE = dg.normal(mu = 0, sigma = 1, name='UE', hidden = True)
        UI = dg.normal(mu = 0, sigma = 1, name='UI', hidden = True)
        B = dg.normal(mu = 0, sigma = 1, name='B', hidden = True)
        E = dg.descendant(lambda B, UE: f_e(UE, B), name='E')  
        I = dg.descendant(lambda B, UI: f_i(UI, B), name='I')
        Y = dg.descendant(lambda UY, I, E : f_y(UY,I, E), name='Y')
        Y_hat = dg.descendant(lambda UY, UI, UE : f_y(UY,UI, UE), name='Y-hat', hidden = True)
    
        df_test = dg.generate(num=self.num_test, hidden=hidden)
        df_test.to_csv('data_test_normal.csv', index=False)

        dg.visualize()

        return df_train, df_test

    def generate_one_feature(self, seed, hidden, f_a, f_y):
        #TrainSet
        dg = Generator(seed=seed)   
        UY = dg.uniform(low=0, high=1, name='UY', hidden=True)
        UA = dg.normal(mu = 0, sigma = 1, name='UA', hidden = True)
        B = dg.custom(lambda s : np.random.binomial(n = 1, p=0.5, size = s), name='B')
        A = dg.descendant(lambda B, UA: f_a(UA, B), name='A')  
        Y = dg.descendant(lambda UY, A : f_y(UY,A), name='Y')
        Y_hat = dg.descendant(lambda UY, UA : f_y(UY,UA), name='Y-hat', hidden = True)

        df_train = dg.generate(num=self.num_train, hidden=hidden)
        df_train.to_csv('data_train_one.csv', index=False)

        dg.visualize()

        #TestSet
        dg = Generator(seed=seed)   
        UY = dg.uniform(low=0, high=1, name='UY', hidden=True)
        UA = dg.normal(mu = 0, sigma = 1, name='UA', hidden = True)
        B = dg.custom(lambda s : np.random.binomial(n = 1, p=0.5, size = s), name='B', hidden=True)
        A = dg.descendant(lambda B, UA: f_a(UA, B), name='A')  
        Y = dg.descendant(lambda UY, A : f_y(UY,A), name='Y')
        Y_hat = dg.descendant(lambda UY, UA : f_y(UY,UA), name='Y-hat', hidden = True)

        df_test = dg.generate(num=self.num_test, hidden=hidden)
        df_test.to_csv('data_test_one.csv', index=False)

        dg.visualize()

        return df_train, df_test
    

class SynteticDataset(Dataset):
    def __init__(self, df):
        self.x = torch.tensor(
            np.array([(df['E'].values - df['E'].mean())/df['E'].std(), (df['I'].values - df['I'].mean())/df['I'].std()]).T,  
            dtype=torch.float32,
        )
        self.z = torch.tensor(df['B'].values, dtype=torch.float32).unsqueeze(1)  
        self.y = torch.tensor(df['Y'].values, dtype=torch.long).unsqueeze(1)  
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.z[idx], self.x[idx], self.y[idx]


class SynteticDatasetOne(Dataset):
    def __init__(self, df):
        self.x = torch.tensor(
            np.array((df['A'].values - df['A'].mean())/df['A'].std()),  
            dtype=torch.float32,
        ).unsqueeze(1)
        self.z = torch.tensor(df['B'].values, dtype=torch.float32).unsqueeze(1)  
        self.y = torch.tensor(df['Y'].values, dtype=torch.long).unsqueeze(1)  
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.z[idx], self.x[idx], self.y[idx]


class AdultDataset(Dataset):
    def __init__(self,df):
        self.x = torch.tensor(
            np.array(df.drop(columns=['race', 'sex', 'age', 'native-country', 'income_>50K']).values),  
            dtype=torch.float32,
        )
        self.z = torch.tensor(df['sex'].values, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(df['income_>50K'].values, dtype=torch.long).unsqueeze(1)

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.z[idx], self.x[idx], self.y[idx]
    

class AdultDatasetTarget(Dataset):
    def __init__(self,df):
        self.x = torch.tensor(
            np.array(df.drop(columns=['race', 'sex', 'age', 'native-country']).values),  
            dtype=torch.float32,
        )
        self.z = torch.tensor(df['sex'].values, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(df['income_>50K'].values, dtype=torch.long).unsqueeze(1)

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.z[idx], self.x[idx], self.y[idx]

    
class NewDataset(Dataset):
    def __init__(self, df, x_new):
        self.z =  torch.tensor(df['B'].values, dtype=torch.float32).unsqueeze(1) 
        self.x = (x_new - torch.mean(x_new, dim = 0))/torch.std(x_new, dim=0)
        self.y = torch.tensor(df['Y'].values, dtype=torch.long).unsqueeze(1)

    def __len__(self):
        return len(self.z)
    
    def __getitem__(self, idx):
        return self.z[idx], self.x[idx], self.y[idx] 