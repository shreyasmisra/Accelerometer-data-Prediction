import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

class Support_Vector_Machine:
    def __init__(self,visualization=True):
        self.visualization = visualization
        self.colors = {1:'r',-1:'b'}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111)
    
    
    def fit(self,data):
        ''' Fit the training data '''
        self.data = data
        opt_dict = {}
        transforms = [[1,1,1],[-1,1,1],[-1,1,-1],[-1,-1,-1],[-1,-1,1],[1,-1,1],[1,-1,-1],[1,1,-1]]
        # transforms = [[1,1],[1,-1],[-1,1],[-1,-1]]
        # inefficient
        all_data = []
        for yi,featureset in list(self.data.items()):
            for feature in featureset:
                all_data.append(feature)
        
        self.max_feature_val = np.max(all_data)
        self.min_feature_val = np.min(all_data)
        all_data = None
        
        step_sizes = [self.max_feature_val * 0.1,
                      self.max_feature_val * 0.01,
                      self.max_feature_val * 0.001,
                      self.max_feature_val * 0.0001# expensive 
                        ]
        # extremely expensive
        b_range_multipe = 5
        # we dont need to take as small steps with b as with w
        b_multiple = 5
        
        # w matrix will be the same number from min to max, still accurate
        latest_optimum = self.max_feature_val*10
        
        for step in step_sizes:
            w = np.array([latest_optimum,latest_optimum,latest_optimum])
            # it will stay False until its optimized because its convex
            optimized = False
            
            while not optimized:
                for b in np.arange(-1*(self.max_feature_val * b_range_multipe), self.max_feature_val * b_range_multipe, b_multiple ):
                    for transfomation in transforms:
                        w_t = w * transfomation
                        # print(w.shape)
                        found_option = True
                        # weakest link in SVM as it has to go through all the data
                        for y in self.data:
                            for xi in self.data[y]:
                                # print(np.transpose(xi).shape)
                                if not y*(np.dot(w_t,xi) + b)>=1 : 
                                    found_option = False
                                    break
                            if not found_option:
                                break 
                        if found_option:
                            opt_dict[np.linalg.norm(w_t)] = [w_t,b]
                if w[0]<0:
                    optimized = True
                    print("optimized a step.")
                
                else:
                    w = w - step
            norms = sorted([n for n in opt_dict])
            opt_choice = opt_dict[norms[0]] # min magnitude of w 
            
            self.w = opt_choice[0]
            self.b = opt_choice[1]
            latest_optimum = opt_choice[0][0] + step*2
                                    
    
    def predict(self,features):
        ''' Function used to predict the data using the fit on the training data'''
        classification = np.sign(np.dot(np.array(features), self.w) + self.b )
        
        if classification !=0 and self.visualization:
            self.ax.scatter(features[0], features[1], marker ='*',s=200, c=self.colors[classification])
        
        return classification
    
    def visualize(self,data):
        ''' visualize the data including the hyperplanes '''
        self.data = data
        [[self.ax.scatter(x[0],x[1],s=100,c=self.colors[i]) for x in self.data[i]] for i in self.data]
        
        
        def hyperplane(x,w,b,v):
            ''' visualize hyperplane'''
            return (-w[0]*x-b+v) / w[1]
        
        data_range = (self.min_feature_val*0.9, self.max_feature_val*1.1)
        hyp_x_min = data_range[0]
        hyp_x_max = data_range[1]
        
        # positive support vector 1
        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1) 
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        self.ax.plot([hyp_x_min,hyp_x_max],[psv1,psv2])
        
        # negative support vector -1
        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1) 
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min,hyp_x_max],[nsv1,nsv2])
        
        # zero support vector 0
        db1 = hyperplane(hyp_x_min, self.w, self.b, 0) 
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min,hyp_x_max],[db1,db2])
    
        plt.show()

### TEST SAMPLE #########

if __name__=='__main__':
    data_dict = {-1: np.array([ [1,7],
                            [2,8],
                            [3,8] ]),
            
            1:np.array([ [5,1],
                          [6,-1],
                          [7,3] ])
            }
    
    predictors = [[0,10],[1,3],[3,4],[3,5],[5,5],[5,6],[6,-5],[5,8],[4,2.5],[7,4],[3,1],[2,1]]
    
    svm = Support_Vector_Machine()
    svm.fit(data_dict)
    
    for p in predictors:
        svm.predict(p)
    svm.visualize(data_dict)
    
    








