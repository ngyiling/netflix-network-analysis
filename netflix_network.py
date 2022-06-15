# %% Install package
!pip install wordcloud

#%% import libraries
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import itertools
from collections import Counter
from networkx.classes.function import degree
from networkx.algorithms.community import girvan_newman, modularity
from pprint import pprint as pp
from wordcloud import WordCloud


#%% import data
df = pd.read_csv("netflix_titles.csv")

#%% only look at 2019-2021 movie releases & United Kingdom
df = df.loc[(df["release_year"]>=2019) & (df["country"]=='United Kingdom') & (df["type"] == "Movie"), :]
# cleaning the data to remove rows with missing director or cast information
df1=df[['show_id', 'director', 'cast']].dropna()

#%% create a list of directors and a list of actors, and assign ID
# explode the director and cast columns
df_1 = df1.assign(**{"director":df["director"].str.split(',')})
df_1 = df_1.assign(**{"cast":df["cast"].str.split(',')})
df_1.reset_index(drop = True, inplace = True)
df_explode = df_1.explode("director").explode("cast")
# creating a new DF for all directors
df_director = pd.DataFrame(df_explode["director"].str.strip().unique())
df_director.rename(columns={0:"Name"}, inplace = True)
# creating a new DF for all cast
df_cast = pd.DataFrame(df_explode["cast"].str.strip().unique())
df_cast.rename(columns={0:"Name"}, inplace = True)
# create a DF for all directors and cast
df_names = df_director.append(df_cast).drop_duplicates().reset_index(drop = True)

# tagging the names with 'D' = director, 'C' = cast, 'B' = Both
df_names["Role"] = None
director = df_director["Name"].to_list()
cast = df_cast["Name"].to_list()
for i in range(len(df_names)):
    if df_names["Name"][i] in director and df_names["Name"][i] in cast:
        df_names["Role"][i] = "B"
    elif df_names["Name"][i] in director:
        df_names["Role"][i] = "D"
    elif df_names["Name"][i] in cast:
        df_names["Role"][i] = "C"

# giving an ID to each node
df_names["id"] = df_names.groupby(["Name"]).ngroup()

#%% get the weight of connections
df_1["names"] = df_1["director"]+df_1["cast"]
for i in range(len(df_1["names"])):
    for j in range(len(df_1["names"][i])):
        df_1["names"][i][j] = df_1["names"][i][j].strip()
df_temp = df_1[["show_id", "names"]].reset_index(drop = True)

# replacing names with their respective IDs
names_temp = df_temp["names"]
df_name_temp = df_names.set_index("Name")
for i in range(len(names_temp)):
    for n in range(len(names_temp[i])):
        names_temp[i][n] = df_name_temp.loc[df_temp["names"][i][n], "id"]

# calculate the weight of each connection
combinations = []
for i in range(len(df_temp)):
    n1 = set(df_temp['names'][i])
    for j in itertools.combinations(n1, 2):
        combinations.append(j)

df_network = pd.DataFrame(combinations, columns = ["n0", "n1"]).drop_duplicates().reset_index(drop=True)
df_network["weight"] = Counter(combinations).values()



# %% assign values to the network
G = nx.from_pandas_edgelist(df_network,                
                            source='n0',        
                            target='n1',        
                            edge_attr='weight')

#%% plot network
# colour mapping for nodes
df_names['colour'] = None
df_names.loc[df_names['Role'] == 'D', 'colour'] = 'darkslategrey'
df_names.loc[df_names['Role'] == 'C', 'colour'] = 'teal'
df_names.loc[df_names['Role'] == 'B', 'colour'] = 'turquoise'
colour_map_node = df_names[['id', 'colour']]
df_nodes = pd.DataFrame(G.nodes, columns=['id'])
colour_map_node = colour_map_node.merge(df_nodes, how = 'right', left_on = 'id', right_on = 'id').set_index('id')

#plot the fig
fig = plt.figure(figsize=(20, 15))
#set the position
pos = nx.layout.spring_layout(G, k = 0.5, seed = 000)
nx.draw_networkx_nodes(G,
                        pos,
                        node_color=colour_map_node['colour'],
                        node_size=100,
                        alpha=1)
nx.draw_networkx_edges(G,
                        pos,
                        edge_color='grey',
                        alpha=0.7)

plt.savefig("network_plot.png",
            transparent=True,
            bbox_inches='tight',
            pad_inches=0,
            dpi=600)
            
plt.show()

# %% degree distribution
l = sorted([d for k,d in G.degree()])
k = Counter(l)
average_degree=np.mean(l)
print('the average_degree =',average_degree)
#  draw the degree distribution
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(12,6))
ax1.bar(k.keys(),[i for i in k.values()])
ax1.axvline(x=average_degree, color='grey', ls='--')
#annotate mean on the graph
ax1.annotate('Mean = 8.31', xy=(2,70),fontsize=12,color='grey')
#set spines
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
#set labels
ax1.set_xlabel('Degree ($k$)')
ax1.set_ylabel('Counts')
#set title
ax1.set_title('Degree Distribution')

# initialize a new figure and plot the data contestually
x = sorted([d for n, d in G.degree()], reverse=True)
ax2.loglog(x, marker="o")
# axes properties
ax2.set_title("Degree rank plot")
ax2.set_ylabel("degree")
ax2.set_xlabel("rank")

# set spines
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
plt.show()
 

# %% node centrality
#compute centrality measures
import seaborn as sns
from networkx.algorithms import centrality, degree_centrality
from networkx.algorithms import betweenness_centrality
from networkx.algorithms import eigenvector_centrality
# --> degree centrality
dg = degree_centrality(G)
# --> betweenness centrality
bc = betweenness_centrality(G)
# --> eigenvector centrality
ec = eigenvector_centrality(G)

# %% degree centrality
name = []
centrality = []
for key, value in dg.items():
    name.append(key)
    centrality.append(value)

cent = pd.DataFrame()    
cent['name'] = name
cent['centrality'] = centrality
cent = cent.sort_values(by='centrality', ascending=False)

#plot the figure
plt.figure(figsize=(10, 25))
#pick the top 10 degree centrality
fig_centrality = sns.barplot(x='centrality', y='name', data=cent[:10], orient='h')
#set labels and title
fig_centrality = plt.xlabel('Degree Centrality')
fig_centrality = plt.ylabel('Directors or Casts')
fig_centrality = plt.title('Top 10 Degree Centrality Scores in netflix Network')
plt.show()
# %% betweeness centrality
name = []
betweenness = []
for key, value in bc.items():
    name.append(key)
    betweenness.append(value)

cent = pd.DataFrame()    
cent['name'] = name
cent['betweenness'] = betweenness
cent = cent.sort_values(by='betweenness', ascending=False)

#plot the figure
plt.figure(figsize=(10, 25))
#pick the top 10 betweenness centrality
fig_betweenness = sns.barplot(x='betweenness', y='name', data=cent[:10], orient='h')
#set labels and title
fig_betweenness = plt.xlabel('betweenness Centrality')
fig_betweenness = plt.ylabel('Directors or Casts')
fig_betweenness = plt.title('Top 10 betweenness centrality Scores in netflix Network')
plt.show()
# %% eigenvector_centrality
name = []
eigenvector = []
for key, value in ec.items():
    name.append(key)
    eigenvector.append(value)

cent = pd.DataFrame()    
cent['name'] = name
cent['eigenvector'] = eigenvector
cent = cent.sort_values(by='eigenvector', ascending=False)

#plot the figure
plt.figure(figsize=(10, 25))
#pick the top 10 eigenvector centrality
fig_eigenvector = sns.barplot(x='eigenvector', y='name', data=cent[:10], orient='h')
#set labels and title
fig_eigenvector = plt.xlabel('eigenvector Centrality')
fig_eigenvector = plt.ylabel('Directors or Casts')
fig_eigenvector = plt.title('Top 10 Eigenvector centrality Scores in netflix Network')

# %% draw the correlation among three centralities
#set share y axis into false to keep their orginal ticks
fig, axs = plt.subplots(3, 3, figsize=(12, 12),sharex='col', sharey=False)
n = G.number_of_nodes() 
# degree centrality
axs[0, 0].hist(centrality) #calculate the percentage of distribution
axs[0, 0].set_ylabel("degree centrality", size=15)
axs[0,0].tick_params(axis='both',labelsize=10)
#degree-betweenness
axs[0, 1].scatter(betweenness, centrality, s=13, marker='o')
axs[0,1].set_title('Correlation among different centralities and its distribution',size=20,pad=20)
axs[0,1].tick_params(axis='both',labelsize=10)
#degree-eigen
axs[0, 2].scatter(eigenvector, centrality, s=13,marker='o')
axs[0,2].tick_params(axis='both',labelsize=10)
#betweenness-degree
axs[1, 0].scatter(centrality, betweenness, s=13,marker='o')
axs[1, 0].set_ylabel("betweenness", size=15)
axs[1,0].tick_params(axis='both',labelsize=10)
#betweenness
axs[1, 1].hist(betweenness,weights= np.ones(n)/n)
axs[1,1].tick_params(axis='both',labelsize=10)
#betweenness-eigen
axs[1, 2].scatter(eigenvector, betweenness, s=13,marker='o')
axs[1,2].tick_params(axis='both',labelsize=10)
#eigen-degree
axs[2, 0].scatter(centrality,eigenvector,s=13,marker='o')
axs[2, 0].set_ylabel("eigenvector", size=15)
axs[2, 0].set_xlabel("degree centrality", size=15)
axs[2,0].tick_params(axis='both',labelsize=10)
#eigen-betweenness
axs[2, 1].scatter(betweenness,eigenvector,s=13,marker='o')
axs[2, 1].set_xlabel("betweenness", size=15)
axs[2,1].tick_params(axis='both',labelsize=10)
#eigen centrality
axs[2, 2].hist(eigenvector,weights= np.ones(n)/n)
axs[2, 2].set_xlabel("eigenvector", size=15)
axs[2,2].tick_params(axis='both',labelsize=10)

plt.tight_layout()
plt.show()
# %% community detection
# use the modularity index to appreciate the quality of alternative
# paritioning solutions
# fit
solutions = girvan_newman(G)
# get all segmentations for the network
k = 35
# register modularity scores
modularity_scores = dict()
comm = []
# iterate over solutions
for community in itertools.islice(solutions, k):
    solution = list(sorted(c) for c in community)
    comm.append(solution)
    score = modularity(G, solution)
    modularity_scores[len(solution)] = score
# print(comm)
# plot modularity data
fig = plt.figure(figsize=(10,8))
pos = list(modularity_scores.keys())
values = list(modularity_scores.values())
ax = fig.add_subplot(1, 1, 1)
ax.stem(pos, values)
ax.set_xticks(pos)
ax.set_xlabel(r'Number of communities detected')
ax.set_ylabel(r'Modularity score')
plt.show()
#%% get the highest modularity score segmentation
comm[1]

#get all shortest path among nodes
n = range(1,199)
x = []
for i in n:
    for j in n:
        try:
            a= nx.shortest_path(G,i,j)
            x.append(a)
        except nx.NetworkXNoPath:
            x = x
list = []
for i in range(len(x)): #get the pathlength which is over 3
    if len(x[i])>3:
        list.append(x[i])

# %% shortest path example
example= nx.shortest_path(G,51,128)
print(example)
# to emphasise the shortest paths example
comm27 = comm[1]
for i in range(len(comm27)):
    comm27[i].insert(0,i)
dict_com = {}
for l2 in comm27:
    dict_com[l2[0]]=l2[1:]

#%% plot network graph
nodes = G.nodes()
import random
numcolor = 27
colorlist = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
for k in range(numcolor)]
nodecolor = []
for n in nodes:
    for i in range(len(dict_com)):
        if n in dict_com[i]:
            nodecolor.append(colorlist[i])

# assign labels to items in the path
labels = {}    
for n in nodes:
    if n in example:
        labels[n] = n
#labels

# assign node sizes to items in the path
size = []
for n in nodes:
    if n in example:
        size.append(300)
    else: size.append(100)

# draw the graph
fig = plt.figure(figsize=(20, 15))
pos = nx.layout.spring_layout(G, k = 0.5, seed = 000)
nx.draw_networkx_nodes(G,
pos,
node_color=nodecolor,
node_size=size,
alpha=1)
nx.draw_networkx_edges(G,
pos,
edge_color='grey',
alpha=0.7)
nx.draw_networkx_labels(G,
pos,
labels,
font_size=30,
font_color='k')

# circle one of the communities for report
ax = plt.gca()
ellipse = Ellipse(xy=(0.6, 0.35), width=0.3, height=0.7, 
                        edgecolor='green', angle = 310, fc='None', lw=2)
ax.add_patch(ellipse)                
plt.show()

# %% name cloud
df_namecloud = df_network
df_namecloud = df_network.n0.map(df_names.set_index('id').Name)
df_namecloud = pd.DataFrame(df_namecloud)

# draw the wordclour fig
# plot the fig
fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(111)
# get name counts
text = df_namecloud['n0'].value_counts().to_dict()
# draw wordcloud, the scale of the word is decided by counts
wordcloud = WordCloud(background_color="white", relative_scaling = 1).generate_from_frequencies(text)
ax.imshow(wordcloud,interpolation='bilinear')
ax.set_title('Top Directors and Casts', size=20, pad=20)
ax.axis("off")
plt.show()

# %% homophily based on movie genres
# set-up the data
df_2 = df.copy()
# group the genres into 2 main categories - Fiction, Non-Fiction 
df_2["genre"] = 'F'
df_2.loc[df_2['listed_in'].str.contains("Documentaries"), 'genre'] = 'NF'
df_2.loc[df_2['listed_in'].str.contains("Documentariess"), 'genre'] = 'NF'
df_2.loc[df_2['listed_in'].str.contains("Stand-Up Comedy"), 'genre'] = 'NF'
df_3 = df_2[["show_id", "genre"]]
# merge with the existing df_explode to get the director and cast names for each show id
df_genre = df_explode.merge(df_3, left_on = "show_id", right_on = "show_id" )
# strip the names of cast and director in the dataframe
df_g_director = df_genre[["director", "genre"]].reset_index(drop=True).rename(columns={"director": "Name"})
df_g_director["Name"] = df_g_director["Name"].str.strip().drop_duplicates()
df_g_cast = df_genre[["cast", "genre"]].reset_index(drop=True).rename(columns={"cast": "Name"})
df_g_cast["Name"] = df_g_cast["Name"].str.strip().drop_duplicates()
#put cast and director in one column
df_mix = df_g_director.append(df_g_cast)
#merge the cast and director with their respective id
df_merge = df_mix.merge(df_names, on= "Name")

#%% get counts of Fictions and non-Fictions
counts1 = df_merge['genre'].value_counts().to_dict()
print("the number of each type of genre within the network:")
print(counts1)

#get counts of diffrent ties
Category1 = df_network.n0.map(df_merge.set_index('id').genre)
Category2 =df_network.n1.map(df_merge.set_index('id').genre)
df_homo = pd.DataFrame({'n1':Category1,'n2':Category2})
df_homo['names'] = df_homo['n1']+df_homo['n2']
counts2 = df_homo['names'].value_counts().to_dict()
print("the number of ties between genres within the network: " )
print(counts2)

c_ff = 792
c_nfnf = 68
c_fnf = 0

#probability of each type of genre within the network
pr_f = 178/(178+29)
pr_nf = 29/(178+29)

#probability of each tie formation
p_fnf = round((0 / (c_ff + c_nfnf)), 2)
p_ff = round((pr_f * pr_f), 2)
p_nfnf = round((pr_nf * pr_nf), 2)

print("probability of Fiction to Fiction: ", p_ff)
print("probability of Non-Fiction to Non-Fiction: ",  p_nfnf)
print("probability of Fiction to Non-Fiction:", p_fnf )
# %%

# %%
