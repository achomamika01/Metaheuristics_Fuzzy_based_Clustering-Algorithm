# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 07:31:51 2022

@author: Shyamchand Salam
"""

#function import
import numpy as np, random, math, operator, time
from alive_progress import alive_bar
#from scipy import spatial
#from sklearn.metrics import silhouette_score # silhouette score
from sklearn.metrics import silhouette_samples, silhouette_score, calinski_harabasz_score, davies_bouldin_score
#sklearn.metrics.calinski_harabasz_score(X, labels)
import matplotlib.pyplot as plt
import matplotlib.cm as cm
start_time = time.time()

# reading dataset
dataset = []
data = open('Iries_No_label.txt')
for line in data:
    line = line.strip()
    dataset.append(line.split(','))
dataset = np.array(dataset, dtype = ('float64'))
#pop_min = 2; pop_max = 6# int(np.sqrt(len(dataset)))
pop_min = 3; pop_max = int(np.sqrt(len(dataset)))
if pop_max >= 13:
    pop_max = 13
    
#User input variable lenght population
import winsound
duration = 1000  # milliseconds
freq = 440  # Hz
winsound.Beep(freq, duration)
population = []; pop_temp =[];
pop_size = input("\n\nEnter the Population size : ")
population_size = int(pop_size)

#Generating varaible length population according to given population size
def variable_population(dataset, population_size):
    #pop_min = 2; pop_max = 6#int(np.sqrt(len(dataset)))
    print('Minimum possible cluster size = ',pop_min)
    print('Maximum possible cluster size = ',pop_max)
    population = []; pop_temp = []; pop_weight = []; weight = []; wei_temp = []
    for pop_counter in range(0, population_size):
        pop_position = int((random.randint(0, len(dataset)))%(pop_max-1))+pop_min
        for pop_1 in range(0, pop_position):
            ran_value = random.randint(0, len(dataset)-1)
            pop_temp.append(dataset[ran_value])
        population.append(pop_temp)
        #Generating weight of the population
        for wei_counter in range(0, len(dataset[0])):
            wei_temp.append(random.random())
        #Normalization of the weight
        for count in wei_temp:
            temp = count/sum(wei_temp)
            weight.append(temp)
        pop_weight.append([pop_temp,weight])
        wei_temp = []
        weight = []
        pop_temp = []
    return population

#Variable population
population = variable_population(dataset, population_size) 

'''
#Printing the varialble population    
print("\nVariable population :\n") 
counter = 0  
for aa in population:
    print("Pop",counter,": Size of Pop :",len(aa)," : ",aa)
    counter += 1
    '''

# Python code to find Euclidean distance 
# using linalg.norm()
def eucl_distance(point1, point2):
    dist = np.linalg.norm(point1 - point2)
    return dist

def distance_cal(a,b):
    from scipy.spatial import distance
    #a = (1, 2, 3)
    #b = (4, 5, 6)
    dst = distance.euclidean(a, b)
    return dst  

#Initiating Membership Matrix
def init_membership_matrix(dataset, population):
    temp_dist_list_0 = []; temp_dist_list_1 = []; temp_dist = [] 
    for b in range(0, len(dataset)):
        for a in range(0, len(population)):
            dist = eucl_distance(dataset[b],population[a]) 
            dist = abs(dist)
            temp_dist.append(dist)
        summation = sum(temp_dist)  
        replace_min = [x / summation for x in temp_dist]
        temp_dist = []
        temp_dist_list_0.append(replace_min)
    temp_dist_list_1.append(temp_dist_list_0)
    return temp_dist_list_1

#Finding the cluster Center.
def CalculateClusterCenter(dataset, membership_matrix):#, possible_cluster_count):
    clusterCenter_test = []
    clusterCenter = []
    for i in range(0, len(membership_matrix[0][0])):
        for j in range(0, len(dataset[0])):
            numerator = 0
            denominator = 0
            for k in range(0, len(dataset)):
                numerator = numerator + math.pow(membership_matrix[0][k][i], 2) * dataset[k][j]
                denominator = denominator + math.pow(membership_matrix[0][k][i], 2)
            try:
                value = numerator/denominator
            except:
                value = 0 
            clusterCenter_test.append(value)
        clusterCenter.append(clusterCenter_test) 
        clusterCenter_test = []
    return clusterCenter

def update_membership_matrix(updated_center, dataset):
    temp_dist_list = []; update_membership = []; temp_dist_list = []; 
    temp_up_member_1 = []
    for b in range(0,len(dataset)):
        temp_up_member = []
        for c in range(0,len(updated_center)):
            temp_dist = eucl_distance(dataset[b], updated_center[c])
            temp_dist = abs(temp_dist)
            temp_dist_list.append(temp_dist)
        for d in range(0, len(temp_dist_list)):
            denominator = sum(math.pow(float(temp_dist_list[d] / temp_dist_list[e]), q) for e in range(0, len(temp_dist_list)))
            temp_val = 1 / denominator
            temp_up_member.append(temp_val)
        temp_dist_list = []
        temp_up_member_1.append(temp_up_member)
    update_membership.append(temp_up_member_1)
    temp_up_member_1 = []
    return update_membership

#lebeling of dataset
def labeler(solution,datset):
    # /labeling of the dataset with the help of the euclidean distance
    labeling_data_set = list()
    for i in range(len(dataset)):
        min_euc_dist = 999999
        label = 0
        for c in range(len(solution)):
            temp_euc_dist = euclidean_distance(list(map(operator.sub, dataset[i], solution[c])))
            if temp_euc_dist < min_euc_dist:
                min_euc_dist = temp_euc_dist
                label = c# + 1
        labeling_data_set.append(label)
    return labeling_data_set

def euclidean_distance(distance_matrix):
    euc_dist = np.linalg.norm(distance_matrix)
    return euc_dist

# jmeasure value
def j_measure(u_membership_matrix, updated_cluster_centers):#, pop_size, K, size_dataset):
    jmeasure = list()
    #for p in range(pop_size):
    outer_sum = 0.0
    for i in range(len(dataset)):
        for j in range(len(updated_cluster_centers)):
            temp_u = math.pow(u_membership_matrix[0][i][j], m)
            temp_norm = math.pow(
                np.linalg.norm((list(map(operator.sub, dataset[i], updated_cluster_centers[j])))), 2)
            outer_sum = outer_sum + (temp_norm * temp_u)
    jmeasure.append(outer_sum)
    #jmeasure = outer_sum
    return jmeasure

# Finding Xie-Beni index
def xie_beni_index(clusterCenter, jindexSum):  
    distanceBetweenClsuterCenter = []; c = 0
    centre_temp = clusterCenter
    numcols = len(clusterCenter)
    pixel = len(dataset)
    for i in range(0, numcols):
        while c < population_size:
            if i == (numcols - 1):
                distanceBetweenClsuterCenter.append(distance_cal(centre_temp[i],centre_temp[0])) #,weigh_te[i]))
            else:
                distanceBetweenClsuterCenter.append(distance_cal(centre_temp[i],centre_temp[i+1])) #,weigh_te[i]))
            c+=1
    try:
        xiebeni = jindexSum[0]/(pixel*min(distanceBetweenClsuterCenter))
    except:
        xiebeni = 0
    return xiebeni

def compute():
    for i in range(1000):
        ... # process items as usual.
        yield  # insert this :)
        
        
#Entering FCM main program

m = 2 #FCM parameter and predefine number
solution = []; jmeasure_list = []; cluster_label = []; silhouette_list = []; xieindex_list = []; center_table = []
q = float(2 / (m - 1))
print('\n\nFCM parameter m = ',m)
print('\n\nPlease wait, finding the best solution vector....')

with alive_bar(population_size) as bar:
    #for i in compute():
    for a in range(0, len(population)):
        epsilon = 0.01
        ep = 99999
        membership_matrix = init_membership_matrix(dataset, population[a])
        while ep > epsilon:
            update_cluster_center = CalculateClusterCenter(dataset, membership_matrix)#, len(population[a]))
            update_member = update_membership_matrix(update_cluster_center, dataset)
            ep = np.array(membership_matrix) - np.array(update_member)
            ep = np.amax(ep)
            membership_matrix = update_member
            temp_cluster_label = labeler(update_cluster_center, dataset)
            jmeasure = j_measure(membership_matrix, update_cluster_center)#, pop_size, K, size_dataset)
            xieindex = xie_beni_index(update_cluster_center,jmeasure)
            try:
                silhouette_avg = silhouette_score(dataset, temp_cluster_label)
            except:
                silhouette_avg = 0
        solution.append(update_cluster_center)
        jmeasure_list.append(jmeasure)
        xieindex_list.append(xieindex)
        center_table.append([jmeasure, xieindex, update_cluster_center])
        cluster_label.append(temp_cluster_label)
        silhouette_list.append(silhouette_avg)
        bar()

'''
print("\nRefine Cluster Center :\n") 
counter = 0 
for aa in range(0, len(solution)):
    print("Solution",counter,": Count of cluster :",len(solution[aa]),"wrt silhoute :",silhouette_list[aa]," : ",solution[aa])
    counter += 1
'''    
#Printing the result of FCM
print("\n\nThe optimum solution vector according to FCM is :")
best_silhoutte = max(silhouette_list)
#result = np.where(silhouette_list == np.amax(silhouette_list))
ind = silhouette_list.index(best_silhoutte)
#result = result.tolist()
#result = result[0]
print(solution[ind])
print("\n\nBest silhoutte score :",best_silhoutte)


winsound.Beep(freq, duration)  
#Entering Classical GWO
print("\n\n\nPeforming Non-dominating sorting using crowding distance")
#maxinteration_string = input("\n\nPlease enter the MAXIMUM NUMBER OF ITERATION : ")
#print("\n\nPlease wait for few seconds....")
maxinteration = 10#int(maxinteration_string)

# Non-dominating sorting
def NonDominatedSort(values1, values2):
    S=[[] for i in range(0,len(values1))]
    front = [[]]
    n=[0 for i in range(0,len(values1))]
    rank = [0 for i in range(0, len(values1))]

    for p in range(0,len(values1)):
        S[p]=[]
        n[p]=0
        for q in range(0, len(values1)):
            if (values1[p] < values1[q] and values2[p] < values2[q]) or (values1[p] <= values1[q] and values2[p] < values2[q]) or (values1[p] < values1[q] and values2[p] <= values2[q]):
                if q not in S[p]:
                    S[p].append(q)
            elif (values1[q] < values1[p] and values2[q] < values2[p]) or (values1[q] <= values1[p] and values2[q] < values2[p]) or (values1[q] < values1[p] and values2[q] <= values2[p]):
                n[p] = n[p] + 1
        if n[p]==0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)
    i = 0
    while(front[i] != []):
        Q=[]
        for p in front[i]:
            for q in S[p]:
                n[q] =n[q] - 1
                if( n[q]==0):
                    rank[q]=i+1
                    if q not in Q:
                        Q.append(q)
        i = i+1
        front.append(Q)

    del front[len(front)-1]
    return front

sortedIndex = NonDominatedSort(jmeasure_list, xieindex_list)
print("\n\nThe Pareto optimal Front are (Non-dominated sorted): \n")
print(sortedIndex)


def calculate_crowding(sortedIndex):
    crowding_list = []; temp_jindex_list = []; temp_xieindex_list = []
    length_nds = len(sortedIndex)
    for i in range(0,length_nds):
        if len(sortedIndex[i]) >= 3:
            #print("Main Crowding distance code initiate from here\n=========\n=======\n=============")
            temp_sortedIndex = sortedIndex[i]
            #print("temp_sortedIndex : ",temp_sortedIndex)
            temp_length = len(temp_sortedIndex)
            temp_crowding_dist = (temp_length)*[0]
            global_jindex_max = max(jmeasure_list)
            global_jindex_min = min(jmeasure_list)
            global_xieindex_max = max(xieindex_list)
            global_xieindex_min = min(xieindex_list)
            
            #print("temp_length : ",temp_length)
            for p in range(0,temp_length):
                #print("value of P : ",p)
                temp_jindex_list.append(jmeasure_list[temp_sortedIndex[p]])
                temp_xieindex_list.append(xieindex_list[temp_sortedIndex[p]])
            sorted_jindex_list = sorted(temp_jindex_list)
            sorted_xiedex_list = sorted(temp_xieindex_list)
            #print("Shyam testing sorted jindex list\n",sorted_jindex_list)
            #jindex_min = sorted_jindex_list[0]
            #jindex_max = sorted_jindex_list[temp_length-1]
            for j in range(0,temp_length):
                #print("j value is : ",j)
                #break
                if j == 0 or j == (temp_length-1):
                    if j == 0:
                        temp_dist = 998
                    elif j == (temp_length-1):
                        temp_dist = 999
                    temp_crowding_dist[j] = temp_dist
                else: #j != 0 or j <= (temp_length-1):
                    temp_dist = temp_crowding_dist[j] + (sorted_jindex_list[j+1][0] - 
                                              sorted_jindex_list[j-1][0])/(global_jindex_max[0] - 
                                                    global_jindex_min[0])
                    temp_dist_xie = temp_dist + (sorted_xiedex_list[j+1] - 
                                                 sorted_xiedex_list[j-1])/(global_xieindex_max -
                                                                   global_xieindex_min)
                    temp_crowding_dist[j] = temp_dist_xie
            crowding_list.append([i+1,sortedIndex[i],temp_crowding_dist])
            #break
        elif len(sortedIndex[i]) == 1:
            sing_inf = []
            cd = 999
            sing_inf.append(cd)
            crowding_list.append([i+1,sortedIndex[i],sing_inf])
            sing_inf = []
        elif len(sortedIndex[i]) == 2:
            infinit_val = []
            cd_first = 998
            cd_last = 999
            infinit_val.append(cd_first)
            infinit_val.append(cd_last)
            crowding_list.append([i+1,sortedIndex[i],infinit_val])
            infinit_val = []
    #crowding_distance = crowding_list
    return crowding_list

crowding_distance = calculate_crowding(sortedIndex)

print("\n\nThe fitness table assignment under NSGA-II")
print("\nFront\tSolutions\tCrowding Distance")
for crowd in crowding_distance:
    print(crowd)
    
# ************* Ranking the crowding distance *******
aa = 0; rank_counter = []
while aa < len(crowding_distance):
    temp_sort = sorted(crowding_distance[aa][2])
    len_temp_sort = len(temp_sort)
    for bb in range(0,len_temp_sort):
        indexing = crowding_distance[aa][2].index(temp_sort[bb])
        rank_counter.append(crowding_distance[aa][1][indexing])
    aa+=1
print("\nRanking process completed...........")
rank_table = []
#print("\n\nRanking the solution wrt crowding distance\n\n")
#print("\nRank \t \t Solution_Index \t \t Cluster_Center")
for cc in range(0,len(rank_counter)):
    rank_table.append([rank_counter[cc],center_table[rank_counter[cc]][2]])
    #print("Rank :",cc+1,"\t \t",rank_counter[cc],"\t \t",center_table[cc][2])

winsound.Beep(freq, duration)
#Entering Classical GWO
print("\n\n\nPlease wait.... grey wolf optimization algorithm is running:")
#maxinteration_string = input("\n\nPlease enter the MAXIMUM NUMBER OF ITERATION : ")
#print("\n\nPlease wait for few seconds....")
maxinteration = 10#int(maxinteration_string)


def A(a):
    r1 = random.uniform(0,1) # r1 and r2 are uniform vector in the range 0 and 1
    A = (2*a*r1)-a
    return A

def C():
    r2 = random.uniform(0,1)
    C = 2*r2  # Control parameter
    return C

gwo_solution =[]; best_sil_clus = []; j_s_center = []
with alive_bar((pop_max+1)-pop_min) as bar:
    for bb in range(pop_min, pop_max+1):
        #print("Checking for Cluster Count = ",bb)
        gwo_population = [item[1] for item in rank_table if len(item[1])==bb]
        if len(gwo_population) > 3:
            gwo_jmeasure_list = []; jmeasure_center = []
            for cc in gwo_population:
                gwo_jm = solution.index(cc)
                gwo_jmeasure_list.append(jmeasure_list[gwo_jm])
            #Entering GWO
            for cc in range(0, len(gwo_population)):
                jmeasure_center.append([gwo_jmeasure_list[cc],gwo_population[cc]])
            sorted_jmeasure_center = sorted(jmeasure_center,key=lambda x: (x[0]))#,reverse=True)    
            #main GWO checking...........
            t = 0 # iterator is assign to zero
            max_sil = 0
            while t < maxinteration: #this part of module can be define as function to call multiple times
                a = 2-(t*(t/maxinteration))
                temp_sol = []; temp_sol_sil=[]
                for i in range(0, len(sorted_jmeasure_center)):
                    alpha = np.array(sorted_jmeasure_center[0][1])
                    beta = np.array(sorted_jmeasure_center[1][1])
                    delta = np.array(sorted_jmeasure_center[2][1])
                    A1 = A(a); A2 = A(a); A3 = A(a)
                    C1 = C(); C2 = C(); C3 = C()
                    D_alpha = abs(C1*alpha - sorted_jmeasure_center[i][1])
                    D_beta = abs(C2*beta - sorted_jmeasure_center[i][1])
                    D_delta = abs(C3*delta - sorted_jmeasure_center[i][1])
                    X1 = alpha - (A1*D_alpha)
                    X2 = beta - (A2*D_beta)
                    X3 = delta - (A3*D_delta)
                    gwo_sol = (X1+X2+X3)/3
                    gwo_membership = update_membership_matrix(gwo_sol, dataset)
                    gjm = j_measure(gwo_membership, gwo_sol)
                    gxi = xie_beni_index(gwo_sol, gjm)
                    temp_sol.append([gjm,gwo_sol])
                    #print("i value : ",i)
                    gwo_cluster_label = labeler(gwo_sol, dataset)
                    try:
                        gwo_silhouette_avg = silhouette_score(dataset, gwo_cluster_label)
                        #print(bb," gwo_silhoute : ",gwo_silhouette_avg)
                        if gwo_silhouette_avg > max_sil:
                            max_sil = gwo_silhouette_avg
                            best_clus = gwo_sol
                            best_bb = bb
                            best_tt = t
                            best_ii = i
                    except:
                        gwo_silhouette_avg = 0 
                    temp_sol_sil.append([gjm, gxi, gwo_silhouette_avg, gwo_sol])
                sorted_jmeasure_center = sorted(temp_sol,key=lambda x: (x[0]))#,reverse=True)
                sorted_jmeasure_center_sil = sorted(temp_sol_sil,key=lambda x: (x[0]))#,reverse=True)
                j_s_center.append([bb,sorted_jmeasure_center_sil[0][0],sorted_jmeasure_center_sil[0][1],sorted_jmeasure_center_sil[0][2],sorted_jmeasure_center_sil[0][3]])
                t =t+1
                #print("T - ",t)
           #AMIKA: best_sil_clus.append([best_bb, best_tt, best_ii, max_sil, best_clus])
            gwo_solution.append(sorted_jmeasure_center[0])
            #print("bb - ",bb)
        bar()

gwo_solution = sorted(gwo_solution, key=lambda x: (x[0]))
#Finding the silhoutte score
silhoutte_index = []; simple_silhoutte = []
for tt in gwo_solution:
    gw_label = labeler(tt[1], dataset)
    #sample_silhouette_values = silhouette_samples(dataset,gw_label)
    try:
        gw_silhouette = silhouette_score(dataset,gw_label,metric='euclidean')
        #sample_silhouette_values = silhouette_samples(dataset,gw_label)
    except:
        gw_silhouette = 0
    silhoutte_index.append([gw_silhouette, tt[1]])
    #simple_silhoutte.append([sample_silhouette_values,tt[1]])
    
print("\n\nThe Silhoueete Score wrt center")
for ee in silhoutte_index:
    print(ee)
    

print("\n\nTime taken in the execution is %s Seconds" % (time.time() - start_time))
winsound.Beep(freq, duration)

import pandas as pd
pd.DataFrame(j_s_center).to_csv('Testis_Output_JM_XI_Sihoute_Center_220.csv')   

def pca_convertor(my_data_list):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    #import numpy as np
    #Xm = np.array(my_data_list)
    Xm = my_data_list
    XX = StandardScaler().fit_transform(Xm)
    #alpha_array = np.array(alpha)
    #alp_alp = StandardScaler().fit_transform(alpha_array)

    # Create a PCA that will retain 99% of the variance
    pca = PCA(n_components=2, whiten=True)
    X_pca = pca.fit_transform(XX)
    return X_pca

winsound.Beep(freq, duration)
#Plotting to verifty the real cluster count and data distribution
print("\n\nPlotting the data with their silhouette score\n")
checking = input("\n\nPlease press ENTER key :")
for aa in silhoutte_index:
    X = dataset
    center = aa[1]
    center = np.array(center)
    n_clusters = len(center)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
    cluster_labels = np.array(labeler(center, dataset))
    try:
        silhouette_avg = silhouette_score(X, cluster_labels)
    except:
        silhouette_avg = 0
    try:
        ch_score = calinski_harabasz_score(X, cluster_labels) 
    except:
        chi = 0
    try:
        db_score = davies_bouldin_score(X, cluster_labels)
    except:
        db_score = 1
        
    if silhouette_avg != 0:
    #silhouette_avg = silhouette_score(X, cluster_labels)
        print("For n_clusters =",n_clusters,"The Calinski-Harabasz_score is :",ch_score)
        print("For n_clusters =",n_clusters,"The Davies Bouldin_score is :",db_score)
        print("For n_clusters =",n_clusters,"The average silhouette_score is :",silhouette_avg)
        sample_silhouette_values = silhouette_samples(X, cluster_labels)
        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        
            ith_cluster_silhouette_values.sort()
        
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
        
            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )
            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        
            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples
        
        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")
        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
        
        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        
        # 2nd Plot showing the actual clusters formed
        X = pca_convertor(X)
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(
            X[:, 0], X[:, 1], marker="o", s=40, lw=0, alpha=0.7, c=colors, edgecolor="k"
        )
    
        centers = pca_convertor(center)
        # Draw white circles at cluster centers
        
        ax2.scatter(
            centers[:, 0],
            centers[:, 1],
            marker="o",
            c="white",
            alpha=1,
            s=200,
            edgecolor="k",
        )
        
        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")
         
        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")
        
        plt.suptitle(
            "Silhouette analysis for GWO clustering on sample data with n_clusters = %d"
            % n_clusters,
            fontsize=14,
            fontweight="bold",
        )
    plt.show()


#TO DO LIST : PRINT Jmeasure, Silhouette sscore and center a ever iteration and write in a csv

