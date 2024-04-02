# kaggle-spaceship-titanic
Predict which passengers are transported to an alternate dimension.

參與 Kaggle [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic/overview) 競賽，配適模型預測哪些乘客會被傳送到異次元。本團隊最佳成績 0.80827，排名 136/2461（截至2023.12.26）。


**一、開發環境**

Python (Google Colab)   

**二、執行方式**

1. Data preprocessing

使用 pd.read_csv() 讀取 train.csv 和 test.csv，並合併檔案進行EDA分析與補缺失值處理。

2. Model fitting
 
(1) 輸入: x_train shape: (6954, 10)
          x_valid shape:  (1739, 10)

根據 permutation importance 計算出前十項較為重要的特徵:

    [Spa, VRDeck, RoomService, Expenses, CryoSleep_True, Deck_E, Side_S, Destination_TRAPPIST-1e, HomePlanet_Mars, Deck_C]

(2) 輸出: y_train shape: (6954,)
          y_valid shape:  (1739,)

    Label: [Transported]

(3) 超參數優化 (Optuna)

    model = XGBClassifier(**params, random_state=0)     #XGBoost模型
  
    score = cross_val_score(model, x, y, cv=5).mean()   #交叉驗證評估模型

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=500)             #迭代500次，找出最佳超參數組合
    best_params = study.best_params
    
(4) 模型訓練
    
    model = XGBClassifier(**best_params, random_state=0)
    model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)

3. Model Evaluation

使用 sklearn.metrics 中的评估指標 (accuracy_score, recall_score, precision_score, f1_score, roc_auc_score) 和混淆矩陣。

4. Final result 

將 test.csv 代入模型，進行分類預測:

    predictions = model.predict(test)
    submit = pd.read_csv("sample_submission.csv")
    submit['Transported'] = predictions
    submit['Transported'] = submit['Transported']>0.5       #結果轉換為True/False 
    submit.to_csv('final_result.csv', index=False)          #寫入CSV檔


[按此](https://jennyliucl.github.io/JennyLiu/project/spaceship_titanic.pdf)查看完整專題報告

