from pymongo import MongoClient

client = MongoClient("mongodb+srv://pashazainul0_db_user:9M4lj2bILXVyGAra@sensorlive1.msaz0g8.mongodb.net/?appName=SENSORLIVE1")

db = client['mydatabase']
collection = db['users']

data = {'name' : 'Zainul', 'age' : 18}