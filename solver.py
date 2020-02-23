import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile

df = pd.read_excel('CE_Europe.xlsx')

print("Column headings:")
print(df.columns)