import numpy as np

class KMeans():
    
    def __init__(self, n_clusters=3, max_iter=300, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter=max_iter
        self.random_state=random_state
        self.cluster_centers_ = None # numpy array # see create_random_centroids for more info
        self.labels_ = None # predictions # numpy array of size len(input)
        self.data_ = None

    def fit(self, input: np.ndarray) -> np.array: 
        """
            Fitting a model means to train the model on some data using your specific algorithm. 
            Typically, you also provide a set of a labels along with your input.
            However, this is an unsupervised algorithm so we don't have y (or labels) to consider! 
                If you're not convinced, look up any supervised learning algorithm on sklearn: https://scikit-learn.org/stable/supervised_learning.html
                If you can explain the difference between the fit function of this unsupervised algorithm and any other supervised algorith, you get 5 extra credit points towards this assignment. 
            This function will simply return the cluster centers, but it will also update your cluster centers and predictions.
        """
        # YOUR CODE HERE
        self.init_centroids(len(input[0]))
        
        self.labels_ = np.full(len(input),0)
        
        self.data_ = input
        
        #np.full(len(input), 0)
        #looks at each item (a list of n_FEATURES), 
        for i in range(self.max_iter):
            min_dist=10000
            count = 0
            
            for item in input:
                clust_count = -1
                label_distances = np.full((len(self.data_),len(input[0])),1000) #size: [num_data][num_features]
                label = 0
                for cluster in self.cluster_centers_:
                    clust_count+=1
                    dist = self.calculate_distance(item, cluster) #which cluster are we closest to?
                    if(dist < min_dist):
                        min_dist = dist
                        label_distances[count][clust_count] = min_dist
                        min_label = clust_count
                self.labels_[count] = min_label
                count+=1

               
               # print("min_dist ", min_dist)
                
            
                #print("labels: ", self.labels_)
            self.recenter_centroids(self.cluster_centers_)  
       
        
            
        
        return self.cluster_centers_
    
    
    def init_centroids(self, num_features: int) -> np.array:
        """
            To initialize the classifier, you will create random starting points for your centroids. 
            You will have n_cluster (N) amounts of centroids and the dimension of your centroids depends on the shape of your data.
            For example, your data may have 100 rows and 5 features (M). Therefore, your centroids will need 5 dimensions as well. 
            This function will return nothing, but it will initialize your cluster centers. 
            cluster_centers_ is an attribute that you will update. 
            It has a specific shape of (N, M) where 
                N = n_clusters 
                M = number of features
        """
        # YOUR CODE HERE 
        self.cluster_centers_=np.random.rand(self.n_clusters, num_features)
        
#         self.recenter_centroids(self.cluster_centers_)
        
        return None

    def calculate_distance(self, d_features, c_features) -> int:
        """
            Calculates the Euclidean distance between point A and point B. 
            Recall that a Euclidean distance can be expanded such that D^2 = A^2 + B^2 + ... Z^2. 
        """
        # YOUR CODE HERE 
        #takes the sum of the (subtraction of two vectors) and uses the distance formula to find the distance between the two coordinates   
        return (np.sum((d_features - c_features)**2)**(1/2))
        
        
    def recenter_centroids(self, input: np.array) -> None:
        """
            This function recenters the centroid to the average distance of all its datapoints.
            Returns nothing, but updates cluster centers 
        """
        #print("clusters: ", len(input)) 
        #step 1: look at all items of label n (ex: all items under label 0)
        #step 2: add all nums and get average
        #step 3: assign average to cluster_centers_[n]
        
        sums = np.zeros(len(self.cluster_centers_), dtype="float64")
        print(sums)
        
        clusterCount = 0
        x = 0
        ave = np.full_like(self.cluster_centers_, 0)
        #print("ave", ave)
        #for each cluster center add associated points
        for cluster in input:
            itemCount = 0
            for item in self.data_:
                #print("item[clusterCount]: ", item[clusterCount],"sums[clusterCount]: ", sums[clusterCount])
                sums[clusterCount] = np.add(np.sum(item[clusterCount]), sums[clusterCount])
                itemCount+=1
            #get the average of the sums and apply to current cluster
            sums[clusterCount] /= itemCount
            self.cluster_centers_[clusterCount] = sums[clusterCount]
            clusterCount += 1
       