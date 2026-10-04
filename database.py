from pymongo import MongoClient

client = MongoClient("mongodb+srv://'''''")

db = client['mydatabase']
collection = db['users']

data = {'name' : 'Zainul', 'age' : 18}