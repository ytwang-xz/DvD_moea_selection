import numpy as np
import pandas as pd

class BayesianBandits(object):
    
    def __init__(self, 
                 arms=[0,0.5],
                 window=50,
                 rolling=1,
                 prev=5):
        
        self.arms = arms
        self.a = {i:1 for i in range(len(self.arms))}
        self.b = {i:1 for i in range(len(self.arms))}
        self.window = window
        self.rolling = rolling
        self.arm = 0   #当前选择的arm
        self.value = self.arms[self.arm]    #当前选择的arm的值
        self.prev = prev
        
        self.choices = [self.arm]  #所有选过的arm
        self.data = [-99999]
        
    def sample(self):
        #先随机生成prev+1个
        
        if len(self.choices) > self.prev:
            samples = []
            #i对应于第i个arm
            for i in range(len(self.arms)):
                samples.append(np.random.beta(self.a[i], self.b[i]))
            
            best = np.argmax(samples)
            self.arm = best
        else:
            self.arm = int(np.random.uniform() * len(self.arms))

        self.value = self.arms[self.arm]
        self.choices.append(self.arm)
        
        return(self.value)
        
    def update_dists(self, prev):
        
        self.data.append(prev)
        
        choices = self.choices.copy()
        choices.append(9999)
        
        if len(choices) > self.prev:
            df = pd.DataFrame({'Choice': choices, 'Reward': self.data})
        
            df['Max_0'] =  df.Reward.rolling(5).max().shift(1)
            df['Max_Ahead'] = df.Reward.rolling(self.rolling).max().shift(-self.rolling)
            df['Feedback'] = df['Max_Ahead'] - df['Max_0']
            df = df[~df['Feedback'].isna()].reset_index(drop=True)
            df['Bern'] = (np.sign(df.Feedback)+1)/2
            df = df.iloc[1:, :].reset_index(drop=True)
            df = df.iloc[-self.window:].reset_index(drop=True)
            
            self.a = {i:(1+df[df['Choice'] ==i].sum()['Bern']) for i in range(len(self.arms))}
            self.b = {i:(1+df[df['Choice'] ==i].count()['Bern'] - df[df['Choice'] ==i].sum()['Bern']) for i in range(len(self.arms))}
       
if __name__ == '__main__':
    BB = BayesianBandits()
    samples = []
    for i in range(10):
        samples.append(BB.sample())
        BB.update_dists(5)


        
