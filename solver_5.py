import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import random as rand
from datetime import datetime
eps = 1
r_f = 0.5

def checkifnan(num):
	if num is np.nan:
		return 0
	else:
		return num

def calculateWeightedParam(companies,listofcompanies,x):
	totalsum = 0.0
	capsum = 0.0
	for company_name in listofcompanies:
		# print(companies[company_name]['PIreturn'][x])
		totalsum += companies[company_name]['CAP'][x]*companies[company_name]['PIreturn'][x]
		capsum += companies[company_name]['CAP'][x]
	# print("totalsum: ",totalsum ," capsum: ",capsum)
	if capsum == 0:
		return 0
	return (totalsum/(capsum))

# def calculateNormalParam(companies,listofcompanies,x):
# 	returnarray = []

# 	for company_name in listofcompanies:
# 		# returnarray.append(companies[company_name]['ROI'][x])
# 		returnarray.append(priceindexreturn[x])

# 	return np.average(returnarray)


# def find3factors(time_row):
	


df = pd.read_excel('CE_Europe.xlsx')
columns = list(df.loc[2])[1:]
# First five rows have no data
row_count = df.shape[0] - 5
companies = dict()
j = 0
company_name = ""
prev_err = False

date=[]
l=df.iloc[4:(4+row_count),0].tolist()
for x in range(row_count):
	date.append(str(l[x]).split()[0])


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
# marketcap = []
priceindexarr = []
# priceindex = []
growtharr = []
# growth = []
booktomarketarr = []
# booktomarket = []
companylist =[]
smblist = []
hmllist = []
flist=[]
momlist=[]
t=[]
y_r=[]

for company_name  in companies:
	companies[company_name]['YR'] = []
	for x in range(row_count):
		companies[company_name]['YR'].append(0.0)

for x in range(0,row_count,12):
	for company_name in companies:
		# companies[company_name]['YR'] = []
		# for  iters in range(0,row_count):
		# 	companies[company_name]['YR'].append(0.0)
		for y in range(12):
			if y == 0:
				if x == 0:
					if companies[company_name]['PI'][x] > 0:
						companies[company_name]['YR'][x+y] = 100.00*(companies[company_name]['PI'][x+11]-companies[company_name]['PI'][x])/(companies[company_name]['PI'][x])
				elif x > 0:
					if companies[company_name]['PI'][x-11] > 0:
						companies[company_name]['YR'][x+y] = 100.00*(companies[company_name]['PI'][x]-companies[company_name]['PI'][x-11])/(companies[company_name]['PI'][x-11])
			elif y > 0:
				if x == 0:
					if companies[company_name]['PI'][x] > 0:
						companies[company_name]['YR'][x+y] = 100.00*(companies[company_name]['PI'][x+11]-companies[company_name]['PI'][x])/(companies[company_name]['PI'][x])
				elif x > 0:
					if companies[company_name]['PI'][x-11] > 0:
						companies[company_name]['YR'][x+y] = 100.00*(companies[company_name]['PI'][x]-companies[company_name]['PI'][x-11])/(companies[company_name]['PI'][x-11])
			# 	pireturn = 100.00*(companies[company_name]['PI'][x]-companies[company_name]['PI'][x-11])/(companies[company_name]['PI'][x-11])
			# y_r.append(pireturn)
		# print(companies[company_name]['YR'])



for company_name in companies:
	# print(companies[company_name]['YR'])
	companylist.append(company_name)

three_factors = []
four_factors = []
new_mom = []
capm=[]
# print(row_count)

for x in range(row_count):
	# marketcap.append([])
	# priceindex.append([])
	# .append([])
	marketcap = []
	priceindex = []
	growth = []
	booktomarket = []
	pireturnarr = []
	new_mom_arr = []

	for company_name in companies:
		if(x == 0):
			companies[company_name]['PIreturn'] = []
			companies[company_name]['PIreturn'].append(0.0)

		marketcap.append(companies[company_name]['CAP'][x])
		# priceindex.append(companies[company_name]['PI'][x])
		growth.append(companies[company_name]['ROI'][x])
		if x > 0:
			if companies[company_name]['PI'][x-1] ==0:
				pireturn=0
			else:
				pireturn = 100.00*(companies[company_name]['PI'][x]-companies[company_name]['PI'][x-1])/(companies[company_name]['PI'][x-1])
			companies[company_name]['PIreturn'].append(pireturn)
			# print(pireturn)
		# booktomarket.append(companies[company_name]['CAP'][x]/(companies[company_name]['PI'][x] + eps))
		# booktomarket.append(priceindexreturn[x])
		pireturnarr.append(companies[company_name]['PIreturn'][x])
		booktomarket.append(companies[company_name]['ROI'][x])
		new_mom_arr.append(companies[company_name]['YR'][x])
	
	# marketcaparr.append(marketcap)
	# # priceindexarr.append(priceindex)
	# growtharr.append(growth)
	# booktomarketarr.append(booktomarket)

	high = np.percentile(booktomarket,70)
	# print(high)
	low = np.percentile(booktomarket,30)
	small = np.percentile(marketcap,30)
	big = np.percentile(marketcap,70)
	# print("----\n\n",big)
	# win = np.percentile(pireturnarr,70)
	# lose = np.percentile(pireturnarr,30)

	win = np.percentile(new_mom_arr,70)
	lose = np.percentile(new_mom_arr,30)

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
		# if pireturnarr[y] > win:
		# 	winnerfirm.append(companylist[y])
		# elif pireturnarr[y] < lose:
		# 	loserfirm.append(companylist[y])

		if new_mom_arr[y] > win:
			winnerfirm.append(companylist[y])
		elif new_mom_arr[y] < lose:
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

	# print("===================================================")
	# print("			V    	   N    	G")
	# print("S  		%d    	%d  	%d"%(len(SV),len(SN),len(SG)))
	# print("B  		%d    	%d  	%d"%(len(BV),len(BN),len(BG)))
	# print("====================================================")

	# print("===================================================")
	# print("			W    	L")
	# print("S  		%d    	%d"%(len(WS),len(LS)))
	# print("B  		%d    	%d"%(len(WB),len(LB)))
	# print("====================================================")

	# exit()
	if len(BV)> 0:
		bvreturn = calculateWeightedParam(companies,BV,x)
		# print("bvreturn",bvreturn,x)
		# exit()
	else:
		bvreturn = 0

	if len(BN)> 0:
		bnreturn = calculateWeightedParam(companies,BN,x)
	else:
		bnreturn = 0

	if len(BG)> 0:
		bgreturn = calculateWeightedParam(companies,BG,x)
	else:
		bgreturn = 0


	if len(SV)> 0:
		svreturn = calculateWeightedParam(companies,SV,x)
	else:
		svreturn = 0

	if len(SN)> 0:
		snreturn = calculateWeightedParam(companies,SN,x)
	else:
		snreturn = 0

	if len(SG)> 0:
		sgreturn = calculateWeightedParam(companies,SG,x)
	else:
		sgreturn = 0

	if len(WB)>0:
		wbreturn = calculateWeightedParam(companies,WB,x)
	else:
		wbreturn = 0

	if len(WS)>0:
		wsreturn = calculateWeightedParam(companies,WS,x)
	else:
		wsreturn = 0

	if len(LB)>0:
		lbreturn = calculateWeightedParam(companies,LB,x)
	else:
		lbreturn = 0

	if len(LS)>0:
		lsreturn = calculateWeightedParam(companies,LS,x)
	else:
		lsreturn = 0

	smb = (svreturn+snreturn+sgreturn)/3 - (bvreturn+bnreturn+bgreturn)/3 
	smblist.append(smb)

	hml = (svreturn+bvreturn)/2 - (sgreturn+bgreturn)/2
	hmllist.append(hml)

	f = calculateWeightedParam(companies,companylist,x)
	flist.append(f)

	mom = (wsreturn+wbreturn)/2 - (lsreturn+lbreturn)/2
	momlist.append(mom)

	three_factors.append([f-r_f,hml,smb])
	capm.append([f-r_f])
	four_factors.append([f-r_f,hml,smb,mom])

	t.append(x)


C = np.array(capm, np.float32)
Z = np.array(four_factors, np.float32)
# Now three_factors is a list of list
X = np.array(three_factors, np.float32)
# Xw = y regression
# Now apply4 linear regression for each company
# print(hmllist)
# print(smblist)
# print(t)
# print(momlist)
# print(flist)

smblist[0]=1
for i in range(1,len(smblist)):
	smblist[i]=smblist[i]/100+smblist[i-1]

hmllist[0]=1
for i in range(1,len(hmllist)):
	hmllist[i]=hmllist[i]/100+hmllist[i-1]

flist[0]=1
for i in range(1,len(flist)):
	flist[i]=flist[i]/100+flist[i-1]

momlist[0]=1
for i in range(1,len(momlist)):
	momlist[i]=momlist[i]/100+momlist[i-1]

# print(smblist)
# print(momlist)
# print(hmllist)
for i in range(len(date)):
	date[i] = datetime.strptime(date[i], '%Y-%m-%d')
#print(date)
plt.plot(date,smblist,label='SMB')
plt.plot(date,hmllist,label='HML')
plt.plot(date,flist,label='F')
#print(date)
# plt.plot(date,momlist,label='MOM')
plt.title('SMB+HML+F vs date')
plt.xlabel('Date')
plt.ylabel('Commulative return')
plt.locator_params(axis='x', nbins=10)
plt.legend(loc=2)
plt.show()

# print("company_name & model.coef & r_sq")
for company_name in companies:
	y = np.array(companies[company_name]['PIreturn'])
	y = y - r_f
	# print(y)
	model = LinearRegression().fit(X, y)
	r_sq = model.score(X, y)
	# print(company_name," & ",round(model.intercept_,3)," & ",round(model.coef_[0],3)," & ",round(model.coef_[1],3)," & ",round(model.coef_[2],3)," \\\\")
	# print(y,model.intercept_,model.coef_)

# print("3factor = ",np.average(r_sq))
# print("==============================================================================================================================================")
# print("==============================================================================================================================================")
# print("==============================================================================================================================================")
# print("==============================================================================================================================================")
# print("==============================================================================================================================================")

for company_name in companies:
	y = np.array(companies[company_name]['PIreturn'])
	y = y - r_f
	model = LinearRegression().fit(Z, y)
	r_sq = model.score(Z, y)
	# print(company_name," & ",round(model.intercept_,3)," & ",round(model.coef_[0],3)," & ",round(model.coef_[1],3)," & ",round(model.coef_[2],3)," & ",round(model.coef_[3],3)," \\\\")
# print("4factor = ",np.average(r_sq))


for company_name in companies:
	y = np.array(companies[company_name]['PIreturn'])
	y = y - r_f
	# print(y)
	model = LinearRegression().fit(C, y)
	r_sq = model.score(C, y)
	# print(company_name," & ",round(model.intercept_,3)," & ",round(model.coef_[0],3)," \\\\")
	# print(y,model.intercept_,model.coef_)
# print("CAPM = ",np.average(r_sq))
