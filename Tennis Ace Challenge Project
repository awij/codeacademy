import codecademylib3_seaborn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# load and investigate the data here:
df = pd.read_csv('tennis_stats.csv')
print(df.head())

# perform exploratory analysis here:
plt.scatter(df['BreakPointsFaced'], df['Winnings'])
plt.xlabel('BreakPointsFaced')
plt.ylabel('Earnings')
plt.title('BreakPointsFaced vs winnings')
plt.show()
plt.clf()

plt.scatter(df['BreakPointsOpportunities'], df['Winnings'])
plt.xlabel('BreakPointsOpportunities')
plt.ylabel('Earnings')
plt.title(' BreakPointsOpportunities vs Winnings')
plt.show()
plt.clf()

## perform single feature linear regressions here:

X = df[['BreakPointsFaced']]
y = df[['Winnings']]
xtrain,xtest,ytrain,ytest = train_test_split(X,y, train_size=0.8)
lr = LinearRegression()
lr.fit(xtrain,ytrain)
print(lr.score(xtest,ytest))
predict1 = lr.predict(xtest)
#
plt.scatter(ytest,predict1, alpha=0.4)
plt.xlabel('Y Test')
plt.ylabel('X Test Prediction')
plt.title('X Test Prediction Winnings vs Y test Actual Winnings')
plt.show()
plt.clf()

## perform two feature linear regressions here:

x2 = df[['BreakPointsOpportunities', 'DoubleFaults']] #DoubleFaults
y2 = df[['Winnings']]
xxtrain, xxtest, yytrain, yytest = train_test_split(x2,y2, train_size=0.8)
lr2 = LinearRegression()
lr2.fit(xxtrain, yytrain)
print(lr2.score(xxtest, yytest))
predict2 = lr2.predict(xxtest)
plt.scatter(yytest, predict2, alpha=0.4)
plt.xlabel('Y Test 2')
plt.ylabel('X Test Prediction 2')
plt.title('X Test Prediction Winnings 2 vs Y test Actual Winnings 2')
plt.show()
plt.clf()

## perform multiple feature linear regressions here:

elements = ['FirstServe','FirstServePointsWon','FirstServeReturnPointsWon',
'SecondServePointsWon','SecondServeReturnPointsWon','Aces',
'BreakPointsConverted','BreakPointsFaced','BreakPointsOpportunities',
'BreakPointsSaved','DoubleFaults','ReturnGamesPlayed','ReturnGamesWon',
'ReturnPointsWon','ServiceGamesPlayed','ServiceGamesWon','TotalPointsWon', 'TotalServicePointsWon']
x3 = df[elements] 
y3 = df[['Winnings']]
x3train, x3test, y3train, y3test = train_test_split(x3,y3, train_size=0.8)
lr3 = LinearRegression()
lr3.fit(x3train, y3train)
print(lr3.score(x3test, y3test))
predict3 = lr3.predict(x3test)
plt.scatter(y3test, predict3, alpha=0.4)
plt.xlabel('Y Test 3')
plt.ylabel('X Test Prediction 3')
plt.title('X Test Prediction Winnings 3 vs Y test Actual Winnings 3')
plt.show()
plt.clf()


































