import matplotlib.pyplot as plt
import random, math
import numpy as np

# The following two functions work recursively in order to compute infection rates from Eq. (3), for each state (i_1,...,i_M), by using matrix V
def InfRates(Infection_rates):
	a=[0]*M
	for I in range(1,N+1):
		l=1
		for i1 in range(max(0,I-sum(a)-sum(Ni[l:M])),min(Ni[l-1]+1,I+1)):
			a[l-1]=i1
			Rates(2,a,I,M)
			a[l-1]=0
			
def Rates(l,a,I,M):
	if l<M:
		for il in range(max(0,I-sum(a)-sum(Ni[l:M])),min(Ni[l-1],I-sum(a))+1):
			a[l-1]=il
			Rates(l+1,a,I,M)
			a[l-1]=0
	else:
		il=I-sum(a)
		a[l-1]=il
		infected=np.asmatrix(np.zeros((M,1)))
		for j in range(M):
			infected[j,0]=a[j]*q
		Solution=np.zeros((M,1))
		Solution=invV*infected
		for j in range(M):
			Infection_rates[j][tuple(a)]=-Solution[j,0]*p
		a[l-1]=0

# Analogous functions to those above, but for the partial derivatives instead
def InfRates_theta(Infection_rates_theta):
	a=[0]*M
	for I in range(1,N+1):
		l=1
		for i1 in range(max(0,I-sum(a)-sum(Ni[l:M])),min(Ni[l-1]+1,I+1)):
			a[l-1]=i1
			Rates_theta(2,a,I,M)
			a[l-1]=0
			
def Rates_theta(l,a,I,M):
	if l<M:
		for il in range(max(0,I-sum(a)-sum(Ni[l:M])),min(Ni[l-1],I-sum(a))+1):
			a[l-1]=il
			Rates_theta(l+1,a,I,M)
			a[l-1]=0
	else:
		il=I-sum(a)
		a[l-1]=il
		infected=np.asmatrix(np.zeros((M,1)))
		for j in range(M):
			infected[j,0]=a[j]*q
		Solution=np.zeros((M,1))
		Solution=-invV*V_theta*invV*infected
		for j in range(M):
			Infection_rates_theta[j][tuple(a)]=-Solution[j,0]*p
		a[l-1]=0

# The following two functions are a recursive scheme to implement Algorithm 1
def My_function(l,a,I,M,d,D):
	if l<M:
		for il in range(max(0,I-sum(a)-sum(Ni[l:M])),min(Ni[l-1],I-sum(a))+1):
			a[l-1]=il
			My_function(l+1,a,I,M,d,D)
			a[l-1]=0
	else:
		il=I-sum(a)
		a[l-1]=il
		denominator=0
		numerator=0
		for j in range(M):
			denominator+=Infection_rates[j][tuple(a)]*(Ni[j]-a[j])+(Gi[j]+Di[j])*a[j]
			if a[j]>0:
				new=np.append([],a)
				new[j]-=1
				numerator+=Gi[j]*a[j]*P[d][0][tuple(new)]
				if d==D-1:
					numerator+=Di[j]*a[j]
				else:
					numerator+=Di[j]*a[j]*P[d+1][n][tuple(a)]
		P[d][0][tuple(a)]=numerator/denominator
		a[l-1]=0

def My_function_n(l,a,I,M,n,d,D):
	if l<M:
		for il in range(max(0,I-sum(a)-sum(Ni[l:M])),min(Ni[l-1],I-sum(a))+1):
			a[l-1]=il
			My_function_n(l+1,a,I,M,n,d,D)
			a[l-1]=0
	else:
		il=I-sum(a)
		a[l-1]=il
		denominator=0
		numerator=0
		for j in range(M):
			denominator+=Infection_rates[j][tuple(a)]*(Ni[j]-a[j])+(Gi[j]+Di[j])*a[j]
			if a[j]>0:
				new=np.append([],a)
				new[j]-=1
				numerator+=Gi[j]*a[j]*P[d][n][tuple(new)]
				if d<D-1:
					numerator+=Di[j]*a[j]*P[d+1][n][tuple(a)]
			if a[j]<Ni[j]:
				new2=np.append([],a)
				new2[j]+=1
				numerator+=Infection_rates[j][tuple(a)]*(Ni[j]-a[j])*P[d][n-1][tuple(new2)]
		P[d][n][tuple(a)]=numerator/denominator
		a[l-1]=0

# The following two functions implement the same recursive scheme, but for the partial derivatives
def My_function_theta(l,a,I,M):
	if l<M:
		for il in range(max(0,I-sum(a)-sum(Ni[l:M])),min(Ni[l-1],I-sum(a))+1):
			a[l-1]=il
			My_function_theta(l+1,a,I,M)
			a[l-1]=0
	else:
		il=I-sum(a)
		a[l-1]=il
		denominator=0
		numerator=0
		for j in range(M):
			denominator+=Infection_rates[j][tuple(a)]*(Ni[j]-a[j])+(Gi[j]+Di[j])*a[j]
			if a[j]>0:
				new=np.append([],a)
				new[j]-=1
				numerator+=Gi[j]*a[j]*Ptheta[0][tuple(new)]
			numerator-=Infection_rates_theta[j][tuple(a)]*(Ni[j]-a[j])*P[0][tuple(a)]
		Ptheta[0][tuple(a)]=numerator/denominator
		a[l-1]=0
		
def My_function_n_theta(l,a,I,M,n):
	if l<M:
		for il in range(max(0,I-sum(a)-sum(Ni[l:M])),min(Ni[l-1],I-sum(a))+1):
			a[l-1]=il
			My_function_n_theta(l+1,a,I,M,n)
			a[l-1]=0
	else:
		il=I-sum(a)
		a[l-1]=il
		denominator=0
		numerator=0
		for j in range(M):
			denominator+=Infection_rates[j][tuple(a)]*(Ni[j]-a[j])+(Gi[j]+Di[j])*a[j]
			if a[j]>0:
				new=np.append([],a)
				new[j]-=1
				numerator+=Gi[j]*a[j]*Ptheta[n][tuple(new)]
			if a[j]<Ni[j]:
				new2=np.append([],a)
				new2[j]+=1
				numerator+=Infection_rates[j][tuple(a)]*(Ni[j]-a[j])*Ptheta[n-1][tuple(new2)]
				numerator+=Infection_rates_theta[j][tuple(a)]*(Ni[j]-a[j])*P[n-1][tuple(new2)]
			numerator-=Infection_rates_theta[j][tuple(a)]*(Ni[j]-a[j])*P[n][tuple(a)]
		Ptheta[n][tuple(a)]=numerator/denominator
		a[l-1]=0

# This is a function to construct matrix V for each ventilation regime
def construct_V(V,regime,beta0):
	if(beta0==9):
		if(regime=='A'):
			V[0,0]=-12
			V[0,1]=9
			
			V[1,0]=9
			V[1,1]=-21
			V[1,2]=9

			V[2,1]=9
			V[2,2]=-21
			V[2,5]=9
			
			V[3,3]=-12
			V[3,4]=9
			
			V[4,3]=9
			V[4,4]=-21
			V[4,5]=9
			
			V[5,2]=9
			V[5,4]=9
			V[5,5]=-30
			V[5,8]=9
			
			V[6,6]=-12
			V[6,7]=9

			V[7,6]=9
			V[7,7]=-21
			V[7,8]=9
			
			V[8,5]=9
			V[8,7]=9
			V[8,8]=-21

		if(regime=='C'):
			V[0,0]=-18
			V[0,1]=18
			
			V[1,0]=9
			V[1,1]=-27
			V[1,2]=18

			V[2,1]=9
			V[2,2]=-27
			V[2,5]=9
			
			V[3,3]=-18
			V[3,4]=18
			
			V[4,3]=9
			V[4,4]=-27
			V[4,5]=18
			
			V[5,2]=9
			V[5,4]=9
			V[5,5]=-36
			V[5,8]=9
			
			V[6,6]=-18
			V[6,7]=18

			V[7,6]=9
			V[7,7]=-27
			V[7,8]=18
			
			V[8,5]=9
			V[8,7]=9
			V[8,8]=-27

		if(regime=='B'):
			V[0,0]=-18
			V[0,1]=9
			
			V[1,0]=18
			V[1,1]=-27
			V[1,2]=9

			V[2,1]=18
			V[2,2]=-27
			V[2,5]=9
			
			V[3,3]=-18
			V[3,4]=9
			
			V[4,3]=18
			V[4,4]=-27
			V[4,5]=9
			
			V[5,2]=9
			V[5,4]=18
			V[5,5]=-36
			V[5,8]=9
			
			V[6,6]=-18
			V[6,7]=9

			V[7,6]=18
			V[7,7]=-27
			V[7,8]=9
			
			V[8,5]=9
			V[8,7]=18
			V[8,8]=-27
			
		if(regime=='D'):
			V[0,0]=-15
			V[0,1]=9
			
			V[1,0]=9
			V[1,1]=-18
			V[1,2]=9

			V[2,1]=9
			V[2,2]=-21
			V[2,5]=9
			
			V[3,3]=-15
			V[3,4]=9
			
			V[4,3]=9
			V[4,4]=-18
			V[4,5]=9
			
			V[5,2]=9
			V[5,4]=9
			V[5,5]=-30
			V[5,8]=9
			
			V[6,6]=-15
			V[6,7]=9

			V[7,6]=9
			V[7,7]=-18
			V[7,8]=9
			
			V[8,5]=9
			V[8,7]=9
			V[8,8]=-21
			
		if(regime=='F'):
			V[0,0]=-15
			V[0,1]=15
			
			V[1,0]=9
			V[1,1]=-24
			V[1,2]=9

			V[2,1]=9
			V[2,2]=-21
			V[2,5]=9
			
			V[3,3]=-15
			V[3,4]=15
			
			V[4,3]=9
			V[4,4]=-24
			V[4,5]=9
			
			V[5,2]=9
			V[5,4]=9
			V[5,5]=-30
			V[5,8]=9
			
			V[6,6]=-15
			V[6,7]=15

			V[7,6]=9
			V[7,7]=-24
			V[7,8]=9
			
			V[8,5]=9
			V[8,7]=9
			V[8,8]=-21
			
		if(regime=='E'):
			V[0,0]=-15
			V[0,1]=9
			
			V[1,0]=15
			V[1,1]=-24
			V[1,2]=9

			V[2,1]=9
			V[2,2]=-21
			V[2,5]=9
			
			V[3,3]=-15
			V[3,4]=9
			
			V[4,3]=15
			V[4,4]=-24
			V[4,5]=9
			
			V[5,2]=9
			V[5,4]=9
			V[5,5]=-30
			V[5,8]=9
			
			V[6,6]=-15
			V[6,7]=9

			V[7,6]=15
			V[7,7]=-24
			V[7,8]=9
			
			V[8,5]=9
			V[8,7]=9
			V[8,8]=-21
	if(beta0==27):
		if(regime=='A'):
			V[0,0]=-30
			V[0,1]=27
			
			V[1,0]=27
			V[1,1]=-57
			V[1,2]=27

			V[2,1]=27
			V[2,2]=-57
			V[2,5]=27
			
			V[3,3]=-30
			V[3,4]=27
			
			V[4,3]=27
			V[4,4]=-57
			V[4,5]=27
			
			V[5,2]=27
			V[5,4]=27
			V[5,5]=-84
			V[5,8]=27
			
			V[6,6]=-30
			V[6,7]=27

			V[7,6]=27
			V[7,7]=-57
			V[7,8]=27
			
			V[8,5]=27
			V[8,7]=27
			V[8,8]=-57

		if(regime=='C'):
			V[0,0]=-36
			V[0,1]=36
			
			V[1,0]=27
			V[1,1]=-63
			V[1,2]=36

			V[2,1]=27
			V[2,2]=-63
			V[2,5]=27
			
			V[3,3]=-36
			V[3,4]=36
			
			V[4,3]=27
			V[4,4]=-63
			V[4,5]=36
			
			V[5,2]=27
			V[5,4]=27
			V[5,5]=-90
			V[5,8]=27
			
			V[6,6]=-36
			V[6,7]=36

			V[7,6]=27
			V[7,7]=-63
			V[7,8]=36
			
			V[8,5]=27
			V[8,7]=27
			V[8,8]=-63

		if(regime=='B'):
			V[0,0]=-36
			V[0,1]=27
			
			V[1,0]=36
			V[1,1]=-63
			V[1,2]=27

			V[2,1]=36
			V[2,2]=-63
			V[2,5]=27
			
			V[3,3]=-36
			V[3,4]=27
			
			V[4,3]=36
			V[4,4]=-63
			V[4,5]=27
			
			V[5,2]=27
			V[5,4]=36
			V[5,5]=-90
			V[5,8]=27
			
			V[6,6]=-36
			V[6,7]=27

			V[7,6]=36
			V[7,7]=-63
			V[7,8]=27
			
			V[8,5]=27
			V[8,7]=36
			V[8,8]=-63
			
		if(regime=='D'):
			V[0,0]=-33
			V[0,1]=27
			
			V[1,0]=27
			V[1,1]=-54
			V[1,2]=27

			V[2,1]=27
			V[2,2]=-57
			V[2,5]=27
			
			V[3,3]=-33
			V[3,4]=27
			
			V[4,3]=27
			V[4,4]=-54
			V[4,5]=27
			
			V[5,2]=27
			V[5,4]=27
			V[5,5]=-84
			V[5,8]=27
			
			V[6,6]=-33
			V[6,7]=27

			V[7,6]=27
			V[7,7]=-54
			V[7,8]=27
			
			V[8,5]=27
			V[8,7]=27
			V[8,8]=-57
			
		if(regime=='F'):
			V[0,0]=-33
			V[0,1]=33
			
			V[1,0]=27
			V[1,1]=-60
			V[1,2]=27

			V[2,1]=27
			V[2,2]=-57
			V[2,5]=27
			
			V[3,3]=-33
			V[3,4]=33
			
			V[4,3]=27
			V[4,4]=-60
			V[4,5]=27
			
			V[5,2]=27
			V[5,4]=27
			V[5,5]=-84
			V[5,8]=27
			
			V[6,6]=-33
			V[6,7]=33

			V[7,6]=27
			V[7,7]=-60
			V[7,8]=27
			
			V[8,5]=27
			V[8,7]=27
			V[8,8]=-57
			
		if(regime=='E'):
			V[0,0]=-33
			V[0,1]=27
			
			V[1,0]=33
			V[1,1]=-60
			V[1,2]=27

			V[2,1]=27
			V[2,2]=-57
			V[2,5]=27
			
			V[3,3]=-33
			V[3,4]=27
			
			V[4,3]=33
			V[4,4]=-60
			V[4,5]=27
			
			V[5,2]=27
			V[5,4]=27
			V[5,5]=-84
			V[5,8]=27
			
			V[6,6]=-33
			V[6,7]=27

			V[7,6]=33
			V[7,7]=-60
			V[7,8]=27
			
			V[8,5]=27
			V[8,7]=27
			V[8,8]=-57

# Number of individual detections until outbreak detection
D=2

# Hospital wards in Figure 7
N=18
# There are 9 ventilation zones
M=9
# But, from an epidemiological perspective, there are 6 compartments in the model (since there are 6 zones with patients)
Mreal=6

# Pulmonary rate
p=0.01
# Quanta rate
q=0.5

# Ni is a vector with the number of patients in each of the 6 compartments
Ni=np.zeros((M,1))
# Gi is a vector with discharge rates for patients in each of the 6 compartments
Gi=np.zeros((M,1))
# Di is a vector with detection rate for patients in each of the 6 compartments
Di=np.zeros((M,1))

# Patient in zone 1a starts the outbreak
init_state=[0]*M
init_state[0]=1

# Patients leave the ward in an average time of 7 days
for i in range(M):
	Ni[i]=N/Mreal
	Gi[i]=1.0/(7*24*60)

Ni[2]=0
Ni[5]=0
Ni[8]=0

regimes=['A','B','C','D','E','F']

# We explore two values of delta in Figure 4
deltas=[12,48]

# And two values of beta0
betas0=[9,27]

plt.figure(1)

for beta0_index in range(2):
	beta0=betas0[beta0_index]

	for delta_index in range(2):
		plt.subplot(2,2,delta_index*2+beta0_index+1)

		# Average time until each patient shows symptoms: delta^{-1}
		for i in range(M):
			Di[i]=1.0/(deltas[delta_index]*60)

		# We consider the 6 ventilation regimes in Figure 7
		for regime_index in range(6):
	
			regime=regimes[regime_index]
			print("Regime: ",regime)
			
			# We construct matrix V
			V=np.asmatrix(np.zeros((M,M)))
			construct_V(V,regime,beta0)

			# And compute its inverse
			invV=np.asmatrix(np.zeros((M,M)))
			invV=V.I

			# The number of states (i_1,...,i_M) in the Markov chain is stored in 'dimensions'
			dimensions=[0]*M
			for i in range(M):
				dimensions[i]=Ni[i]+1

			# We compute all infection rates for all states (i_1,...,i_M) from Eq. (3)
			Infection_rates=[np.ndarray(shape=(tuple(dimensions)), dtype=float) for n in range(M)]
			InfRates(Infection_rates)

			a=[0]*M
			a[0]=1

			nmax=20
			nmax_mean=40
			P=[[np.ndarray(shape=(tuple(dimensions)), dtype=float) for n in range(nmax_mean+1)] for d in range(D)]
			mean_R0=0

			d=D-1
			
			# We apply a recursive scheme to implement (an adapted version of) Algorithm 1, for n=0
			n=0
			a=[0]*M
			P[d][0][tuple(a)]=1
			for I in range(1,N+1):
				l=1
				for i1 in range(max(0,I-sum(a)-sum(Ni[l:M])),min(Ni[l-1]+1,I+1)):
					a[l-1]=i1
					My_function(2,a,I,M,d,D)
					a[l-1]=0

			# And a recursive scheme for n>0
			for n in range(1,nmax_mean+1):
				a=[0]*M
				P[d][n][tuple(a)]=0
				for I in range(1,N+1):
					for i1 in range(max(0,I-sum(a)-sum(Ni[l:M])),min(Ni[l-1],I-sum(a))+1):
						l=1
						a[l-1]=i1
						My_function_n(2,a,I,M,n,d,D)
						a[l-1]=0

			# We repete the recursive scheme above, for subsequent values of d. We started with d=D-1, and decrease d until d=1. Note that in Figure 4, states 
			# are of the form (i_1,...,i_M,d), for d=0,...,D-1
			for d in reversed(range(1,D-1)):
				n=0
				a=[0]*M
				P[d][0][tuple(a)]=1
				for I in range(1,N+1):
					l=1
					for i1 in range(max(0,I-sum(a)-sum(Ni[l:M])),min(Ni[l-1]+1,I+1)):
						a[l-1]=i1
						My_function(2,a,I,M,d,D)
						a[l-1]=0
				
				for n in range(1,nmax_mean+1):
					a=[0]*M
					P[d][n][tuple(a)]=0
					for I in range(1,N+1):
						for i1 in range(max(0,I-sum(a)-sum(Ni[l:M])),min(Ni[l-1],I-sum(a))+1):
							l=1
							a[l-1]=i1
							My_function_n(2,a,I,M,n,d,D)
							a[l-1]=0

			# We implement the recursive scheme for d=0
			d=0
			n=0
			a=[0]*M

			P[d][0][tuple(a)]=1
			for I in range(1,N+1):
				l=1
				for i1 in range(max(0,I-sum(a)-sum(Ni[l:M])),min(Ni[l-1]+1,I+1)):
					a[l-1]=i1
					My_function(2,a,I,M,d,D)
					a[l-1]=0

			probabilities_plot=np.append([],P[d][0][tuple(init_state)])
			probabilities_mean=np.append([],P[d][0][tuple(init_state)])
			
			for n in range(1,nmax_mean+1):
				a=[0]*M
				P[d][n][tuple(a)]=0
				for I in range(1,N+1):
					for i1 in range(max(0,I-sum(a)-sum(Ni[l:M])),min(Ni[l-1],I-sum(a))+1):
						l=1
						a[l-1]=i1
						My_function_n(2,a,I,M,n,d,D)
						a[l-1]=0

				probabilities_mean=np.append(probabilities_mean,P[d][n][tuple(init_state)])
				if n<=nmax:
					probabilities_plot=np.append(probabilities_plot,P[d][n][tuple(init_state)])
				mean_R0+=n*P[d][n][tuple(init_state)]

			print("E[R_0]: ",mean_R0)
			
			if regime=='A':
				plt.bar(np.arange(0, nmax+1, 1)-0.5, probabilities_plot, width=0.15, color="black",label='A')
			if regime=='B':
				plt.bar(np.arange(0, nmax+1, 1)-0.35, probabilities_plot, width=0.15, color="red",label='B')
			if regime=='C':
				plt.bar(np.arange(0, nmax+1, 1)-0.2, probabilities_plot, width=0.15, color="blue",label='C')
			if regime=='D':
				plt.bar(np.arange(0, nmax+1, 1)-0.05, probabilities_plot, width=0.15, color="orange",label='D')
			if regime=='E':
				plt.bar(np.arange(0, nmax+1, 1)+0.1, probabilities_plot, width=0.15, color="green",label='E')
			if regime=='F':
				plt.bar(np.arange(0, nmax+1, 1)+0.25, probabilities_plot, width=0.15, color="brown",label='F')

			plt.xlabel("n")
			plt.ylabel(r'$P(R_{0}=n)$')
			plt.xlim(-0.5,nmax+0.5)
			plt.ylim(0,0.5)
			plt.legend()

plt.show()

