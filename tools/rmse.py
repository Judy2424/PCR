import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import pearsonr
import json
import numpy as np
from scipy import stats
import os

from sklearn.linear_model import LinearRegression


# vehicle
sources = ["coco", "bdd", "cityscapes", "detrac", "exdark", "kitti", "self_driving2coco", "roboflow2coco", "udacity2coco", "traffic2coco"]

# pedestrian
# sources = ["caltech", "citypersons", "cityscapes", "coco", "crowdhuman", "ECP", "ExDark", "kitti", "self_driving"]

rmses = []
for index_to_exclude in range(len(sources)):
    train_set = sources[:index_to_exclude] + sources[index_to_exclude+1:]
    text_set = [sources[index_to_exclude]]
    print(train_set)
    print(text_set)
    X = []
    y1 = []
    y2 = []
    select = []
    metesize = 50
    for source in train_set:
        for meta in range(metesize):
            with open('./result/car_inc/PCR/r50_retina/' + str(source) + '_s250_n50/' + str(meta) + '.json') as f:
                data = json.load(f)
                X.append(data['0'][0][0]*100)  
                y1.append(data['0'][1][0])
                y2.append(data['0'][2][0])
                
                        
    metesize = 1
    true = []
    esti1 = []
    esti2 = []
    dropout_pos = "1_2"
    dropou_rate ='0_15'
    for source in text_set:
        for meta in range(metesize):
            with open('./result/car_ori/PCR/r50_retina/cost_droprate_' + dropou_rate + '_' + str(source) + '_droppos_' + dropout_pos + '_s250_n50/' + str(meta) + '.json') as f:
                        data = json.load(f)
                        true.append(data['0'][0][0]*100)
                        esti1.append(data['0'][1][0])
                        esti2.append(data['0'][2][0])
                       
    y1 = np.array(y1).reshape(-1, 1)
    y2 = np.array(y2).reshape(-1, 1)
    PCR = np.hstack((y1, y2))
    mAP = np.array(X)
    model = LinearRegression()
    model.fit(PCR, mAP)
    
    omega1 = model.coef_[0]
    omega2 = model.coef_[1]
    omega0 = model.intercept_

    print(f"ω1: {omega1}")
    print(f"ω2: {omega2}")
    print(f"ω0: {omega0}")

    
    esti1 = np.array(esti1).reshape(-1, 1)
    esti2 = np.array(esti2).reshape(-1, 1)
    
    esti = np.hstack((esti1, esti2))
    true = np.array(true)

    mAP_pred = model.predict(esti)

    # Root Mean Squared Error (RMSE) 
    rmse = np.sqrt(np.mean((true - mAP_pred) ** 2))
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    rmses.append(rmse)
    
print(f"avg. rmse : {np.mean(rmses)}")
