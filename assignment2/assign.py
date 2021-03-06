import pandas as pd
import numpy as mp

#Reading CSV into a DataFrame
trainDf = pd. read_csv('test.csv');

#Removing unneeded columns
idDf = trainDf[['id']];

#Generating Random Number for each row id
idDf.insert(1, "target",0);
idDf['target'] = mp.random.rand (700000,1);
#Writing to CSV
print(idDf);
idDf.to_csv('out.csv', index=False);