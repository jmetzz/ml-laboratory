import numpy as np
import pandas as pd
import csv

NUM_OF_PEOPLE = 100
NUMBER_OF_STOCKS = 3

np.random.seed(444)
# age, gender, number of children
user_Df = pd.DataFrame(index=range(0, NUM_OF_PEOPLE))

number_of_children = np.random.normal(1, 1, 2 * NUM_OF_PEOPLE)
realistic_n_of_children = [round(child) for child in number_of_children if (child > -0.5)]

random_double_ages = np.random.normal(30, 10, 2 * NUM_OF_PEOPLE)
limited_int_ages = [round(age) for age in random_double_ages if (age > 18) and (age < 65)]

gender = ['M', 'F']
user_Df['gender'] = [gender[np.random.randint(0, 2)] for _ in range(len(user_Df.index))]
user_Df['age'] = [limited_int_ages[i] for i in range(len(user_Df.index))]
user_Df['numberOfChildren'] = [realistic_n_of_children[i] for i in range(len(user_Df.index))]

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
            specific_trends = []
            for tr in range(2, len(row) - 3):
                if row[tr] == 'yes':
                    specific_trends.append(labels[tr - 2])

            stocks.append({'name': row[0], 'tag': row[1], 'trends': specific_trends})

trends = []
for label in labels:
    stocksArr = []
    for stock in stocks:
        if label in stock['trends']:
            stocksArr.append(stock['tag'])
    trends.append({'name': label, 'stocks': stocksArr})


# Assign top 3 of liked and disliked trends
def find_trends(trends, number_of_stocks, young_person):
    liked_trends = []
    disliked_trends = []
    trends_copy = trends.copy()
    for _ in range(0, min(number_of_stocks, len(trends_copy))):
        liked_trends.append(trends_copy.pop(np.random.randint(0, len(trends_copy)))['name'])

    for _ in range(0, min(number_of_stocks, len(trends_copy))):
        disliked_trends.append(trends_copy.pop(np.random.randint(0, len(trends_copy)))['name'])

    trend_arr = []
    for trend in trends:
        if trend['name'] in liked_trends:
            trend_arr.append(1)
        elif trend['name'] in disliked_trends:
            trend_arr.append(-1)
        else:
            trend_arr.append(0)

    # totally legit thing to hard-code here
    if young_person and (np.random.randint(0, 10) > 3):
        trend_arr[0] = 1
        trend_arr[3] = 1

    return trend_arr


users = []
for i in range(0, len(user_Df.index)):
    trends_arr = []
    number_of_stocks = NUMBER_OF_STOCKS
    is_young_user = user_Df['age'][i] < 30
    users.append(find_trends(trends, number_of_stocks, is_young_user))

users_transposed = []
for index in range(0, len(labels)):
    new_line = []
    for userId in range(0, len(users)):
        new_line.append(users[userId][index])
    users_transposed.append(new_line)

trend_columns = {}
for index in range(0, len(labels)):
    trend_columns[labels[index]] = users_transposed[index]

trend_Df = pd.DataFrame(data=trend_columns)
complete_Df = user_Df.merge(trend_Df, left_index=True, right_index=True)
# print(completeDf)

# write to CSV
complete_Df.to_csv('./output.csv', sep=',', index=False)

'''
Assigning stocks to users based on preferences
'''

tags = [stock['tag'] for stock in stocks]

portfolio = pd.DataFrame(index=range(0, NUM_OF_PEOPLE))

for stock in stocks:
    portfolio[stock['tag']] = [0 for _ in range(0, len(portfolio.index))]


def make_stock_list(user_id):
    rest = 100
    amount_of_stocks = np.random.randint(1, 10)
    list_of_percentages = []

    while rest > 0 and len(list_of_percentages) < amount_of_stocks:
        portfolio_percentage = np.random.randint(1, 1 + rest)
        rest = rest - portfolio_percentage
        minimum = min(portfolio_percentage, rest)
        if minimum == 0:
            minimum = 5
        list_of_percentages.append(minimum)

    if sum(list_of_percentages) < 100:
        list_of_percentages.append(100 - sum(list_of_percentages))

    sorted_percentages = sorted(list_of_percentages, reverse=True)

    # traverse labels
    liked = []
    disliked = []
    neutral = []
    for label in labels:
        if complete_Df[label][user_id] == 1:
            liked.append(label)
        elif complete_Df[label][user_id] == -1:
            disliked.append(label)
        else:
            neutral.append(label)

    trends_bought = []
    for i in range(0, min(len(liked), round(len(sorted_percentages) / 2))):
        trends_bought.append(liked[np.random.randint(0, len(liked))])

    difference = len(sorted_percentages) - len(trends_bought)

    for i in range(0, difference):
        trends_bought.append(neutral[np.random.randint(0, len(neutral))])

    stocks_bought = []
    for i in range(0, len(trends_bought)):
        for trend in trends:
            if trend['name'] == trends_bought[i]:
                matching_trend = trend
                break
        stocks_bought.append(matching_trend['stocks'][np.random.randint(0, len(matching_trend['stocks']))])

    stocks_bought_sorted = [0] * len(stocks)
    for idx in range(0, len(stocks_bought)):
        stocks_bought_sorted[tags.index(stocks_bought[idx])] = 1

    return stocks_bought_sorted


stocks_Df = pd.DataFrame(columns=tags, index=range(0, NUM_OF_PEOPLE))
for index in range(0, NUM_OF_PEOPLE):
    stocks_Df.loc[index] = make_stock_list(index)

stocks_Df.to_csv('./portfolios.csv', sep=',', index=False)
