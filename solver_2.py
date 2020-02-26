import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import numpy as np
from sklearn.linear_model import LinearRegression


eps = 1e-6
r_f = 0.5

def checkifnan(num):
	if num is np.nan:
		return 0
	else:
		return num

def calculateWeightedParam(companies,listofcompanies):
	totalsum = 0
	capsum = 0

	for company_name in listofcompanies:
		totalsum = totalsum + companies[company_name]['CAP']*companies[company_name]['ROI']
		capsum = totalsum + companies[company_name]['ROI']

	return (totalsum/capsum)

def calculateNormalParam(companies,listofcompanies,x):
	returnarray = []

	for company_name in listofcompanies:
		returnarray.append(companies[company_name]['ROI'][x])

	return np.average(returnarray)


# def find3factors(time_row):
	


df = pd.read_excel('CE_Europe.xlsx')
columns = list(df.loc[2])[1:]
# First five rows have no data
row_count = df.shape[0] - 5
companies = dict()
j = 0
company_name = ""
prev_err = False
for i,col in enumerate(columns):
	# Calculate the actual offset
	idx = i + 1
	if(col == "#ERROR"):
		prev_err = True
		j = (j+1)%3
		continue
	elif(j == 0):
		# New Company
		company_name = col.split('-')[0].strip()
		companies[company_name] = dict()
		# CAP
		a = np.nan_to_num(np.array(df.iloc[4:(4+row_count),idx]).astype(float))
		# print(a)
		companies[company_name]['CAP'] = a
		
	elif(j == 1):
		# ROI
		if(prev_err == True):
			company_name = col.split('-')[0].strip()
			if(company_name not in companies):
				companies[company_name] = dict()
		a = np.nan_to_num(np.array(df.iloc[4:(4+row_count),idx]).astype(float))
		companies[company_name]['ROI'] = a
	elif(j == 2):
		# PI
		if(prev_err == True):
			company_name = col.split('-')[0].strip()
			if(company_name not in companies):
				companies[company_name] = dict()
		a = np.nan_to_num(np.array(df.iloc[4:(4+row_count),idx]).astype(float))
		companies[company_name]['PI'] = a

	j = (j+1)%3
	prev_err = False
for company_name in companies:
	if 'PI' not in companies[company_name]:
		companies[company_name]['PI'] = np.array([0.]*row_count)
	if 'ROI' not in companies[company_name]:
		companies[company_name]['ROI'] = np.array([0.]*row_count)
	if 'CAP' not in companies[company_name]:
		# print(company_name)
		companies[company_name]['CAP'] = np.array([0.]*row_count)

marketcaparr = []
marketcap = []
priceindexarr = []
priceindex = []
growtharr = []
growth = []
booktomarketarr = []
booktomarket = []
companylist =[]

for company_name in companies:
	companylist.append(company_name)

three_factors = []
four_factors = []

for x in range(row_count):
	for company_name in companies:
		marketcap.append(companies[company_name]['CAP'][x])
		priceindex.append(companies[company_name]['PI'][x])
		growth.append(companies[company_name]['ROI'][x])
		booktomarket.append(companies[company_name]['CAP'][x]/(companies[company_name]['PI'][x] + eps))
	
	marketcaparr.append(marketcap)
	priceindexarr.append(priceindex)
	growtharr.append(growth)
	booktomarketarr.append(booktomarket)

	high = np.percentile(booktomarket,70)
	low = np.percentile(booktomarket,30)
	small = np.percentile(marketcap,30)
	big = np.percentile(marketcap,70)
	win = np.percentile(growth,70)
	lose = np.percentile(growth,30)

	valuefirm = []
	growthfirm = []
	neutralfirm = []
	smallfirm = []
	bigfirm = []
	winnerfirm = []
	loserfirm = []

	for y in range(len(companylist)):
		if marketcap[y] > big:
			bigfirm.append(companylist[y])
		elif marketcap[y] < small:
			smallfirm.append(companylist[y])
		if booktomarket[y] < low:
			growthfirm.append(companylist[y])
		elif booktomarket[y] > low and booktomarket[y] < high:
			neutralfirm.append(companylist[y])
		elif booktomarket[y] > high:
			valuefirm.append(companylist[y])
		if growth[y] > win:
			winnerfirm.append(companylist[y])
		elif growth[y] < lose:
			loserfirm.append(companylist[y])


	BV = list(set(bigfirm) & set(valuefirm))
	BN = list(set(bigfirm) & set(neutralfirm))
	BG = list(set(bigfirm) & set(growthfirm))
	SV = list(set(smallfirm) & set(valuefirm))
	SN = list(set(smallfirm) & set(neutralfirm))
	SG = list(set(smallfirm) & set(growthfirm))
	WB = list(set(winnerfirm) & set(bigfirm))
	WS = list(set(winnerfirm) & set(smallfirm))
	LB = list(set(loserfirm) & set(bigfirm))
	LS = list(set(loserfirm) & set(smallfirm))

	if len(BV)> 0:
		bvreturn = calculateNormalParam(companies,BV,x)
	else:
		bvreturn = 0

	if len(BN)> 0:
		bnreturn = calculateNormalParam(companies,BN,x)
	else:
		bnreturn = 0

	if len(BG)> 0:
		bgreturn = calculateNormalParam(companies,BG,x)
	else:
		bgreturn = 0


	if len(SV)> 0:
		svreturn = calculateNormalParam(companies,SV,x)
	else:
		svreturn = 0

	if len(SN)> 0:
		snreturn = calculateNormalParam(companies,SN,x)
	else:
		snreturn = 0

	if len(SG)> 0:
		sgreturn = calculateNormalParam(companies,SG,x)
	else:
		sgreturn = 0

	if len(WB)>0:
		wbreturn = calculateNormalParam(companies,WB,x)
	else:
		wbreturn = 0

	if len(WS)>0:
		wsreturn = calculateNormalParam(companies,WS,x)
	else:
		wsreturn = 0

	if len(LB)>0:
		lbreturn = calculateNormalParam(companies,LB,x)
	else:
		lbreturn = 0

	if len(LS)>0:
		lsreturn = calculateNormalParam(companies,LS,x)
	else:
		lsreturn = 0

	smb = (svreturn+snreturn+sgreturn)/3 - (bvreturn+bnreturn+bgreturn)/3 

	hml = (svreturn+bvreturn)/2 - (sgreturn+bgreturn)/2

	f = np.average(growth)

	mom = (wsreturn+wbreturn)/2 - (lsreturn+lbreturn)/2

	three_factors.append([1,f-r_f,hml,smb])

	four_factors.append([1,f-r_f,hml,smb,mom])

Z = np.array(four_factors, np.float32)
# Now three_factors is a list of list
X = np.array(three_factors, np.float32)
# Xw = y regression
# Now apply linear regression for each company
for company_name in companies:
	y = companies[company_name]['ROI']
	y = y - r_f
	model = LinearRegression().fit(X, y)
	r_sq = model.score(X, y)
	print(company_name,model.coef_,r_sq)


for company_name in companies:
	y = companies[company_name]['ROI']
	y = y - r_f
	model = LinearRegression().fit(Z, y)
	r_sq = model.score(Z, y)
	print(company_name,model.coef_,r_sq)