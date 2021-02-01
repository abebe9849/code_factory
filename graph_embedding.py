import pandas as pd 
import collections
import numpy as np
import gensim
import time,random,sys
from tqdm import tqdm
import networkx as nx

df_network = pd.DataFrame()
df_network["col1"] = ["a","c","c","a","x","x","t","x","x","t","b","b","b","a","e","c","c","d","d","e","e","d","a","d"]
df_network["col2"] = ["x","x","t","x","x","t","a","c","c","a","a","c","c","a","b","b","b","e","c","c","e","c","c","t"]
df_network["date"] = 1 ##ここはdatetime型ですでに時系列順に並んでいるものとする

df_train = pd.DataFrame()
df_train["col"] = list(set([*df_network["col1"].unique(),*df_network["col2"].unique()]))

#自己回帰(col1==col2)を除く
df_network["id1=id2"] = df_network["col1"]==df_network["col2"]
df_network = df_network[df_network["id1=id2"]==False].drop("id1=id2",axis=1)
#col1基準、時系列に並べておく
df_network = df_network.sort_values(["col1","date"])

#col1→col2の有向グラフ、向き指定はcreate_using=nx.Graph()にて
G = nx.from_pandas_edgelist(df_network, "col1", "col2", edge_attr=True, create_using=nx.Graph())

### ①[node_id1,node_id2]をtextとみる =================
#sizeは生成される埋め込みベクトルの長さ、windowは周辺何単語まで考慮するか、iterjはどれだけ学習してw2vするか
w2v_model_1 = word2vec.Word2Vec(df_network[['col1', 'col2']].values.tolist(), size=50,min_count=1,window=1,iter=100)

##ほぞんしたいならここで　~~~~~
w2v_model_2.wv.save_word2vec_format(f"./network.vec.pt", binary=True)
model_2 = KeyedVectors.load_word2vec_format(f'./network.vec.pt', binary=True)
##>>vec2 = model_2[word2]でvecとりだし
# ~~~

###  =================

### ②col2を時系列で =================
##ログデータに対して各人のイベントを時系列に並べてtext化,w2vでイベントの埋め込みベクトルを得る　　というのがあったそう(Riiid)
col1 = df_network["col1"].values.tolist()
col2 = df_network["col2"].values
c = collections.Counter(col1)
c = np.array(list(c.values()))
c_1 = np.cumsum(c)
col2_new = np.split(col2, c_1[:-1])
df_renew = pd.DataFrame()
df_renew["col1"] = df_network["col1"].unique()
##col2を時系列で並べた
df_renew["col2_not_dupli"] = [list(c) for c in col2_new]
##col2を時系列で並べた(重複col2は削除) 
def remove_duplicate(x):
    t_new = []
    for i in t:
        if i not in t_new:
            t_new.append(i)
    return t_new
    
df_renew["col2"] = [remove_duplicate(c) for c in col2_new]
##col2を時系列で並べたものの先頭にclo1をつけた
df_renew["col1_2"] = [[c1,*list(c)] for c,c1 in zip(col2_new,df_network["col1"].unique())]

seq = list(df_renew["col1_2"].values)
w2v_model_2 = word2vec.Word2Vec(seq, size=100,min_count=1,window=1,iter=100)
###  =================

### ③ =================
### w2vで[col1,col2]としたものは最近傍だけをみていたが、n次近傍までをみる。num_walkは試行回数で計算量↑精度↑
def make_random_walks(G, num_walk, length_of_walk):
    ## num_walkは何周試行するか,データ数が大きくなる
    #ランダムウォークで歩いたノードを入れるlistを生成
    paths = list()
    #node_list = list(G.nodes())
    node_list = df_network["corporation_id1"].unique()
    #ランダムウォークを擬似的に行う
    for i in range(num_walk):
        for node in node_list:
            now_node = node
            ### スタートとlength_of_walk歩　歩いたとき　到着したノード(length_of_walk+1)がpathに入る==========
            #到達したノードを追加する用のリストを用意する
            path = list()
            path.append(str(now_node))
            for j in range(length_of_walk):
                #次に到達するノードを選択する
                next_node = random.choice(list(G.neighbors(now_node)))
                #リストに到達したノードをリストに追加する
                path.append(str(next_node))
                #今いるノードを「現在地」とする
                now_node = next_node
            #ランダムウォークしたノードをリストに追加
            paths.append(path)

            # ==========
    #訪れたノード群を返す
    return paths

random_walk = make_random_walks(G, 30, 3)
w2v_model_3 = word2vec.Word2Vec(random_walk, size=100,min_count=1,window=1,iter=10)

###  =================

### ④ =================

###　random_warkの遷移の確率をパラメータp,qによって制御する。
### 小q大でbfs(周辺みる)、逆の時dfs(深く繋がりを追う)  
def make_node2vec(G, num_walk, length_of_walk,p=1,q=1):
    ## num_walkは何周試行するか,データ数が大きくなる
    ## p,qはnode2vecのパラメータ　https://recruit.gmo.jp/engineer/jisedai/blog/node2vec/
    ##p小q大でbfs(周辺みる)、逆の時dfs(深く繋がりを追う)
    #ランダムウォークで歩いたノードを入れるlistを生成
    paths = list()
    #node_list = list(G.nodes())
    node_list = df_network["corporation_id1"].unique()
    #ランダムウォークを擬似的に行う
    for i in range(num_walk):
        for node in node_list:
            now_node = node
            ### スタートとlength_of_walk歩　歩いたとき　到着したノード(length_of_walk+1)がpathに入る==========
            #到達したノードを追加する用のリストを用意する
            path = list()
            path.append(str(now_node))
            for j in range(length_of_walk):
                #次に到達するノードを選択する
                if j ==0:
                    next_node = random.choice(list(G.neighbors(now_node)))
                    path.append(str(next_node))
                    pre_node = now_node
                    now_node = next_node
                else:
                    pre = list(G.neighbors(pre_node))
                    now = list(G.neighbors(now_node))
                    dtx_0 = list(set(pre_node)& set(now))
                    dtx_1 = list(set(pre)& set(now))
                    dtx_2 = list(set(now) -set(pre)& set(now)-set(pre_node))
                    choice_prob = np.array([*[1/p]*len(dtx_0),*[1]*len(dtx_1),*[1/q]*len(dtx_2)])
                    choice = [*dtx_0,*dtx_2,*dtx_1]
                    next_node = np.random.choice(choice, p=choice_prob/sum(choice_prob))
                    pre_node = now_node
                    now_node = next_node      
                    path.append(str(next_node))   
            #ランダムウォークしたノードをリストに追加
            paths.append(path)

            # ==========
    #訪れたノード群を返す
    return paths

node2vec_wark = make_node2vec(G, 50, 3)
w2v_model_4 = word2vec.Word2Vec(node2vec_wark, size=100,min_count=1,window=10,iter=100)

###  =================

#######~~~~~~~~~~~~~~


def func1(x):
    return w2v_model_1.wv[x]
def func2(x):
    return w2v_model_2.wv[x]
def func3(x):
    return w2v_model_3.wv[x]
def func4(x):
    return w2v_model_4.wv[x]


train["w2v"] = train["col"].apply(func1)
train["timeseries"] = train["col"].apply(func2)
train["deepwark"] = train["col"].apply(func1)
train["node2vec"] = train["col"].apply(func2)
#https://speakerdeck.com/sansandsoc/sansanxatmacup-number-6-1st-place-solution?slide=11
#https://speakerdeck.com/sansandsoc/overview-of-graph-data-analysis-and-dsoc-initiatives?slide=21

