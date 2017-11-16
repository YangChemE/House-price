import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

##read the data set
HOUSE = pd.read_csv("https://raw.githubusercontent.com/YangChemE/House-price/master/data",sep = '\t')
##encoding city names into a vector of numerical values
enc = LabelEncoder()
enc.fit(HOUSE['City'])
HOUSE['City'] = enc.transform(HOUSE['City'])

#store the numeric city vector temperorily
temp_city = HOUSE['City']

#generate the 1-hot vector for the city
a = HOUSE['City']
b = np.zeros((len(temp_city), (max(temp_city)+1)))
b[np.arange(len(temp_city)), a] = 1


##split the data set into train set and test set
train, test = train_test_split(HOUSE, train_size = 0.7)
train_city, test_city = train_test_split(b, train_size = 0.7)

##store values for features that are going to be trained with
##sqft is in unit k and price is in unit of million
sqft = np.array(train["sqft"]/1000)

city = np.array(train_city)
price = np.array(train["price"]/1000000)
beds = np.array(train["bedrooms"])
baths = np.array(train["baths"])
X0=[]
for i in range(58):
    X0.append(1)

##store values for features that are going to be tested with
##sqft is in unit k and price is in unit of million
sqft_test = np.array(test["sqft"]/1000)

city_test = np.array(test_city)
price_test = np.array(test["price"]/1000000)
beds_test = np.array(test["bedrooms"])
baths_test = np.array(test["baths"])

X0_test=[]
for i in range(25):
    X0_test.append(1)

##declare the weights
w_sqft = 1

w_city = []
for i in range(48):
    w_city.append(1)
w_city = np.array(w_city)
w_beds = 1
w_baths = 1
w_0 = 1

##encoding datas and weights into matrices
Features = np.mat(np.column_stack((X0,sqft,beds,baths,city)))

##encoding datas and weights into matrices
TESTS = np.mat(np.column_stack((X0_test,sqft_test,beds_test,baths_test,city_test)))




W=np.mat(np.hstack((w_0, w_sqft, w_beds, w_baths, w_city))).T




##declare boolean function and error tolerance
converge = False
tol = 0.0000001
R = 0.0001
E_temp = 0
i = 0

while converge == False :
    
    
    price_model = Features*W
  

    error = np.sum(np.square(np.array(price_model - np.mat(price).T)))/(len(train))
    
    GRAD = ((Features.T)*(price_model - np.mat(price).T))/(len(train))
    
    W = W - (R*GRAD)
    if (abs(error - E_temp) <= tol):
        converge = True
        print ("Converged ! Final error is " , error)
        
        break
    
    E_temp = error
    i=i+1 
    if (i%10000 ==0):
        print (i , " iteration, the error is " , error)


price_new = TESTS*W
error_test = np.sum(np.square(np.array(price_new - np.mat(price_test).T)))/(len(test))

print ("Test error for this model is ", error_test)


print ("Now Perceptron: ")
#Perceptron
W = []
for i in range(52):    
    W.append(0)
W = (np.mat(W)).T
price_model = Features*W
price_new = TESTS*W
r = 0.001  
Price_TH = np.mean(HOUSE["price"]/1000000)# Threshold
class_train_true = (np.mat(np.sign(price - Price_TH))).T
class_train_model = np.sign(price_model)
accuracy_train = ((len(train))-np.count_nonzero(class_train_true - class_train_model))/(len(train))

class_test_true = (np.mat(np.sign(price_test - Price_TH))).T
class_test_model = np.mat(np.sign(price_new)).T
accuracy_test = ((len(test))-np.count_nonzero(class_test_true - class_test_model))/(len(test))



for i in range (20000):
    for j in range (len(train)):
         if (class_train_true[j] != class_train_model[j]):
            W = W + r*(class_train_true.item(j)*(Features[j].T))
            price_model = Features*W
            class_train_model = np.sign(price_model) #modeled classification for train set
            accuracy_train = ((len(train))-np.count_nonzero(class_train_true - class_train_model))/(len(train))
            if(i%1000==0):
                print("The model can now classify the price of houses with accuracy of ",
                    accuracy_train, " for the training set.")
    
            
price_new = TESTS*W,

class_test_true = (np.mat(np.sign(price_test - Price_TH))).T
class_test_model = np.mat(np.sign(price_new)).T
accuracy_test = ((len(test))-np.count_nonzero(class_test_true - class_test_model))/(len(test))


    
    
print("The model can classify the price of houses with accuracy of ", accuracy_train, " for the training set, and ",
    accuracy_test, " for the test set")


