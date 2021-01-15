from loaddata import load_csv
from model import *

ROOT_CSV_TRAIN = r"D:\工作日志\大三上\DIP\temp.csv"
ROOT_CSV_VAL = r"D:\工作日志\大三上\DIP\temp.csv"
ROOT_CSV_TEST = "D:\工作日志\大三上\DIP\新建 Microsoft Excel 工作表.xlsx"

# Train
x, y = load_csv(ROOT_CSV_TRAIN)
print(x)
print(y)
model = svm_model()
model.fit(x, y)

# Val
x_val, y_val = load_csv(ROOT_CSV_VAL)
pred = model.predict(x_val)
percentage = sum(pred == y_val) / len(pred)
print(percentage)

