'''
@Author : Sumitraj Patil
'''

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.feature_extraction.text import  TfidfTransformer
from sklearn.pipeline import Pipeline

yelp = pd.read_csv('yelp.csv')
yelp.head()
yelp.info()
yelp.describe()


'''Create a new column called "text length" which is the number of words in the text column'''
yelp['text length'] = yelp['text'].apply(len)


sns.set_style('white')
get_ipython().run_line_magic('matplotlib', 'inline')


'''Use FacetGrid from the seaborn library to create a grid of 5 histograms of text length based off of the star ratings. Reference the seaborn documentation for hints on this'''
g = sns.FacetGrid(yelp,col='stars')
g.map(plt.hist,'text length')


'''Create a boxplot of text length for each star category.'''
sns.boxplot(x='stars',y='text length',data=yelp,palette='rainbow')


'''Create a countplot of the number of occurrences for each type of star rating'''
sns.countplot(x='stars',data=yelp,palette='rainbow')


'''Use groupby to get the mean values of the numerical columns, you should be able to create this dataframe with the operation'''
stars = yelp.groupby('stars').mean()
print(stars)
'''Use the corr() method on that groupby dataframe to produce this dataframe:'''
stars.corr()


'''Then use seaborn to create a heatmap based off that .corr() dataframe:'''
sns.heatmap(stars.corr(),cmap='coolwarm',annot=True)

'''Create a dataframe called yelp_class that contains the columns of yelp dataframe but for only the 1 or 5 star reviews.'''
yelp_class = yelp[(yelp.stars==1) | (yelp.stars==5)]


'''Create two objects X and y. X will be the 'text' column of yelp_class and y will be the 'stars' column of yelp_class. (Your features and target/labels)'''
X = yelp_class['text']
y = yelp_class['stars']
cv = CountVectorizer()


''' Use the fit_transform method on the CountVectorizer object and pass in X (the 'text' column). Save this result by overwriting X'''
X = cv.fit_transform(X)

# Let's split our data into training and testing data.
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=101)
nb = MultinomialNB()
nb.fit(X_train,y_train)

'''Predictions and Evaluations'''
predictions = nb.predict(X_test)
print(confusion_matrix(y_test,predictions))
print('\n')
print(classification_report(y_test,predictions))
