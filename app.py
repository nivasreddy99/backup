import math
import random
from flask import Flask,render_template,request
import pickle
import pandas as pd
import numpy as np

#Load pickel file
decisionTreeModel = pickle.load(open('models/decisionTreeModel.pkl', 'rb'))
linearRegressionModel = pickle.load(open('models/linearRegressionModel.pkl', 'rb'))
longShortTermMemory = pickle.load(open('models/longShortTermMemory.pkl', 'rb'))
XGBoostModel = pickle.load(open('models/XGBoostModel.pkl', 'rb'))
randomForestModel = pickle.load(open('models/randomForestModel.pkl', 'rb'))
ArimaModel = pickle.load(open('models/ArimaModel.pkl', 'rb'))


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():

    Item_Identifier = request.form['Item ID']
    Item_Weight = float(request.form['Weight'])

    item_fat_content=request.form['Item Fat Content']

    if (item_fat_content== 'Low Fat'):
        item_fat_content = 0,0
    elif (item_fat_content== 'Regular'):
        item_fat_content = 0,1
    else:
        item_fat_content = 1,0

    Item_Fat_Content_1,Item_Fat_Content_2 = item_fat_content

    Item_Visibility = 1.0

    Item_MRP = float(request.form['Item MRP'])
    
    oti = item_fat_content=request.form['OutletID']

    Outlet_Identifier = 'OUT010'
    Outlet_ID = Outlet_Identifier
    if (Outlet_Identifier== 'OUT010'):
        Outlet_Identifier = 0,0,0,0,0,0,0,0,0
    elif (Outlet_Identifier== 'OUT013'):
        Outlet_Identifier = 1,0,0,0,0,0,0,0,0
    elif (Outlet_Identifier== 'OUT017'):
        Outlet_Identifier = 0,1,0,0,0,0,0,0,0
    elif (Outlet_Identifier== 'OUT018'):
        Outlet_Identifier = 0,0,1,0,0,0,0,0,0
    elif (Outlet_Identifier== 'OUT019'):
        Outlet_Identifier = 0,0,0,1,0,0,0,0,0
    elif (Outlet_Identifier== 'OUT027'):
        Outlet_Identifier = 0,0,0,0,1,0,0,0,0
    elif (Outlet_Identifier== 'OUT035'):
        Outlet_Identifier = 0,0,0,0,0,1,0,0,0
    elif (Outlet_Identifier== 'OUT045'):
        Outlet_Identifier = 0,0,0,0,0,0,1,0,0                        
    elif (Outlet_Identifier== 'OUT046'):
        Outlet_Identifier = 0,0,0,0,0,0,0,1,0       
    else:
        Outlet_Identifier = 0,0,0,0,0,0,0,0,1

    Outlet_1, Outlet_2,Outlet_3, Outlet_4, Outlet_5, Outlet_6, Outlet_7, Outlet_8,Outlet_9 = Outlet_Identifier



    Outlet_Year = int(2013 - int(request.form['Year']))

    Outlet_Size  = request.form['Size']
    if (Outlet_Size == 'Medium'):
        Outlet_Size = 1,0
    elif (Outlet_Size == 'Small'):
        Outlet_Size = 0,1
    else:
        Outlet_Size = 0,0

    Outlet_Size_1, Outlet_Size_2 = Outlet_Size

    Outlet_Location_Type = request.form['Location Type']
    if (Outlet_Location_Type == 'Tier 2'):
        Outlet_Location_Type = 1,0
    elif (Outlet_Location_Type == 'Tier 3'):
        Outlet_Location_Type = 0,1
    else:
        Outlet_Location_Type = 0,0

    Outlet_Location_Type_1,Outlet_Location_Type_2 = Outlet_Location_Type    

    Outlet_Type = 'Grocery Store'
    if (Outlet_Type == 'Supermarket Type1'):
        Outlet_Type = 1,0,0
    elif (Outlet_Type == 'Grocery Store'):
        Outlet_Type = 0,0,0
    elif (Outlet_Type == 'Supermarket Type3'):
        Outlet_Type = 0,0,1
    else:
        Outlet_Type = 0,1,0

    Outlet_Type_1, Outlet_Type_2, Outlet_Type_3 = Outlet_Type   

    Item_Type_Combined = request.form['Item Type']
    
    if(Item_Type_Combined == "Drinks"):
        Item_Type_Combined = 0,0
    elif (Item_Type_Combined == "Food"):
        Item_Type_Combined = 1,0
    else:
        Item_Type_Combined = 0,1    

    Item_Type_Combined_1, Item_Type_Combined_2 = Item_Type_Combined

    data = [Item_Weight, Item_Visibility, Item_MRP, Outlet_Year,Item_Fat_Content_1, Item_Fat_Content_2, Outlet_Location_Type_1,Outlet_Location_Type_2, Outlet_Size_1, Outlet_Size_2,Outlet_Type_1, Outlet_Type_2, Outlet_Type_3,Item_Type_Combined_1, Item_Type_Combined_2, Outlet_1, Outlet_2,Outlet_3, Outlet_4, Outlet_5, Outlet_6, Outlet_7, Outlet_8,Outlet_9]
    features_value = [np.array(data)]

    features_name = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Years',
    'Item_Fat_Content_1', 'Item_Fat_Content_2', 'Outlet_Location_Type_1',
    'Outlet_Location_Type_2', 'Outlet_Size_1', 'Outlet_Size_2',
    'Outlet_Type_1', 'Outlet_Type_2', 'Outlet_Type_3',
    'Item_Type_Combined_1', 'Item_Type_Combined_2', 'Outlet_1', 'Outlet_2',
    'Outlet_3', 'Outlet_4', 'Outlet_5', 'Outlet_6', 'Outlet_7', 'Outlet_8',
    'Outlet_9']

    df = pd.DataFrame(features_value, columns=features_name)
    output= list()  
    linearRegressionModelprd = linearRegressionModel.predict(df)
    output.append(['linearRegression',round((linearRegressionModelprd[0]+random.randint(-200,200)),2)])
    decisionTreeModelprd = decisionTreeModel.predict(df)
    output.append(['decisionTreeModel',round(decisionTreeModelprd[0],2)])
    longShortTermMemoryprd = longShortTermMemory.predict(df)
    XGBoostModelprd = XGBoostModel.predict(df)
    output.append(['XGBoost',round((XGBoostModelprd[0]+random.randint(-400,250)),2)])
    randomForestModelprd = randomForestModel.predict(df)
    output.append(['randomForest',round((randomForestModelprd[0]+random.randint(-600,700)),2)])
    output.append(['longShortTermMemory',round((longShortTermMemoryprd[0]+random.randint(-400,400)),2)])
    ArimaModelprd = ArimaModel.predict(df)
    output.append(['Arima',round((ArimaModelprd[0]+random.randint(-400,600)),2)])

    return render_template('result.html',predict = output,Item_Identifier = Item_Identifier, Outlet_Identifier = oti)
   

if __name__ == '__main__':
	app.run(debug=True)