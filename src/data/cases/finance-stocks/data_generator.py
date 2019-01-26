import numpy as np
import pandas as pd
import csv

NUM_OF_PEOPLE = 100
NUMBER_OF_STOCKS = 3

np.random.seed(444)
# age, gender, number of children
userDf = pd.DataFrame(index=range(0, NUM_OF_PEOPLE))

numberOfChildren = np.random.normal(1, 1, 2*NUM_OF_PEOPLE)
realisticNOfChildren = [round(child) for child in numberOfChildren if (child > -0.5)]

randomDoubleAges = np.random.normal(30, 10, 2*NUM_OF_PEOPLE)
limitedIntAges = [round(age) for age in randomDoubleAges if (age >18) and (age < 65)]

gender = ['M', 'F']
userDf['gender'] = [gender[np.random.randint(0, 2)] for _ in range(len(userDf.index))]
userDf['age'] = [limitedIntAges[i] for i in range(len(userDf.index))]
userDf['numberOfChildren'] = [realisticNOfChildren[i] for i in range(len(userDf.index))]

'''
Open the stocks CSV file and assign the labels and stocks lists
'''

# get CSV into useful format
with open('stocks.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=';')
    stocks = []
    for index, row in enumerate(reader):
        if index == 0:
            labels = row[2:-3]
        else:
            specificTrends = []
            for tr in range(2, len(row)-3):
                if row[tr] == 'yes':
                    specificTrends.append(labels[tr-2])

            stocks.append({'name': row[0], 'tag': row[1], 'trends': specificTrends})


trends = []
for label in labels:
    stocksArr = []
    for stock in stocks:
        if label in stock['trends']:
            stocksArr.append(stock['tag'])
    trends.append({'name': label, 'stocks': stocksArr})

# Assign top 3 of liked and disliked trends
def findTrends(trends, numberOfStocks, youngPerson):
    likedTrends = []
    dislikedTrends = []
    trendsCopy = trends.copy()
    for _ in range(0, min(numberOfStocks, len(trendsCopy))):
        likedTrends.append(trendsCopy.pop(np.random.randint(0, len(trendsCopy)))['name'])

    for _ in range(0, min(numberOfStocks, len(trendsCopy))):
        dislikedTrends.append(trendsCopy.pop(np.random.randint(0, len(trendsCopy)))['name'])

    trendArr = []
    for trend in trends:
        if trend['name'] in likedTrends:
            trendArr.append(1)
        elif trend['name'] in dislikedTrends:
            trendArr.append(-1)
        else:
            trendArr.append(0)

    # totally legit thing to hard-code here
    if youngPerson:
        if (np.random.randint(0, 10) > 3):
            # mobility
            trendArr[0] = 1
            # technology
            trendArr[3] = 1

    return trendArr

users = []
for i in range(0, len(userDf.index)):
    trendsArr = []
    numberOfStocks = NUMBER_OF_STOCKS
    isYoungUser = userDf['age'][i] < 30
    users.append(findTrends(trends, numberOfStocks, isYoungUser))

usersTransposed = []
for index in range(0, len(labels)):
    newLine = []
    for userId in range(0, len(users)):
        newLine.append(users[userId][index])
    usersTransposed.append(newLine)


trendColumns = {}
for index in range(0, len(labels)):
    trendColumns[labels[index]] = usersTransposed[index]

trendDf = pd.DataFrame(data=trendColumns)
completeDf = userDf.merge(trendDf, left_index=True, right_index=True)
# print(completeDf)

#write to CSV
completeDf.to_csv('./output.csv', sep=',', index=False)

'''
Assigning stocks to users based on preferences
'''

tags = [stock['tag'] for stock in stocks]

portfolio = pd.DataFrame(index=range(0, NUM_OF_PEOPLE))

for stock in stocks:
    portfolio[stock['tag']] = [0 for _ in range(0, len(portfolio.index))]

# print(portfolio)

def isZero(x):
    return x == 0

def makeStockList(userId):
    rest = 100
    amountOfStocks = np.random.randint(1, 10)
    listOfPercentages = []

    while rest > 0 and len(listOfPercentages) < amountOfStocks:
        portfolioPercentage = np.random.randint(1, 1 + rest)
        rest = rest - portfolioPercentage
        minimum = min(portfolioPercentage, rest)
        if minimum == 0:
            minimum = 5
        listOfPercentages.append(minimum)

    if sum(listOfPercentages) < 100:
        listOfPercentages.append(100 - sum(listOfPercentages))

    listOfSortedPercentages = sorted(listOfPercentages, reverse=True)
    # print()
    # print(listOfSortedPercentages)

    # traverse labels
    liked = []
    disliked = []
    neutral = []
    for label in labels:
        if completeDf[label][userId] == 1:
            liked.append(label)
        elif completeDf[label][userId] == -1:
            disliked.append(label)
        else:
            neutral.append(label)

    # print(liked)

    trendsBought = []
    for i in range(0, min(len(liked), round(len(listOfSortedPercentages)/2))):
        trendsBought.append(liked[np.random.randint(0, len(liked))])

    difference = len(listOfSortedPercentages) - len(trendsBought)

    for i in range(0, difference):
        trendsBought.append(neutral[np.random.randint(0, len(neutral))])


    stocksBought = []
    for i in range(0, len(trendsBought)):
        for trend in trends:
            if (trend['name'] == trendsBought[i]):
                matchingTrend = trend
                break
        stocksBought.append(matchingTrend['stocks'][np.random.randint(0, len(matchingTrend['stocks']))])

    # print(trendsBought)
    # print(stocksBought)

    stocksBoughtSorted = [0] * len(stocks)
    print(tags)
    for index in range(0, len(stocksBought)):
        # stocksBoughtSorted[tags.index(stocksBought[index])] += listOfSortedPercentages[index]
        stocksBoughtSorted[tags.index(stocksBought[index])] = 1

    # print(stocksBoughtSorted)
    return stocksBoughtSorted

stocksDf = pd.DataFrame(columns=tags, index=range(0, NUM_OF_PEOPLE))
for index in range(0, NUM_OF_PEOPLE):
    stocksDf.loc[index] = makeStockList(index)

stocksDf.to_csv('./portfolios.csv', sep=',', index=False)
