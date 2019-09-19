# 證交所買賣交易日報驗證碼

### 免責條款

此專案是個人學習如何使用 Deep Learning 中的 CNN，使用 Python 的 Keras、Tensorflow 進行實作，請勿使用於不法用途。若因使用該專案而侵害他人權利，請自行負責。

請參考 [證交所買賣日報](https://bsr.twse.com.tw/bshtm/use.htm) 使用條款

### 參考資料

其實基本上都是依照下面參考資料在實作，所以有疑問請在參考一下這些資料。

[如何透過OpenCV 破解台灣證券交易所買賣日報表的驗證碼(Captcha) (Part 1)?](https://www.youtube.com/watch?v=KESG8I9C3oA)

不過這專案主要是使用 Deep Learning 中的 CNN 影像辨識，而不是使用 pytesser 進行解碼。

### Dependencies

請先安裝相關的 python 套件

```sh
pip3 install -r requirements.txt
```

### 步驟

大致上分為四個步驟，以下會分步驟說明
- 爬蟲
- 預處理
- 標記圖片
- CNN深度學習

### 爬蟲

爬蟲請參考 `crawler.ipynb` 和編譯出來的 `crawler.py`。此程式使用 python requests 去抓取驗證碼存入至 captcha 目錄下。

檔案列表：

| # | ipython notebook 檔 | python檔 |
|---|---|---|
| 1 | crawler.ipynb |  |

### 預處理

圖片預處理就參考[參考資料](#參考資料)的 youtube 教學影片。

檔案列表：

| # | ipython notebook 檔 | python檔 |
|---|---|---|
| 1 | preprocess.ipynb |  |
| 2 | preprocess-batch.ipynb | preprocess-batch.py |

### 標記圖片

標記圖片花了很多時間，一開始使用我另外一個專案 [label_captcha_tool](https://github.com/maxmilian/label_captcha_tool) 進行標記。等到數量累積到一定量(>3000)後，並可以訓練並且使用爬蟲自動標記。自動標註請參考 `demo.ipynb`

這邊就標記檔，存成為 csv (`label.csv`)，每一個圖片一行，之後要丟入 CNN 當作 label 的訓練資料。

### CNN深度學習

CNN model 直接使用經典的VGG來改，不過為減小訓練時間，filters從32開始，並且加入了BatchNorm。

| # | ipython notebook 檔 | python檔 |
|---|---|---|
| 1 | cnn.ipynb | |

這邊就標記檔，存成為 csv，每一個圖片一行，之後要丟入 CNN 當作 label 的訓練資料。

model.summary()

model.fit log

#### 訓練 log

# 其他

```
# 影像預處理
jupyter nbconvert --to script preprocessBatch.ipynb

# utilties
jupyter nbconvert --to script utilities.ipynb
```
