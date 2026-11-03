import re
import pandas as pd
from langdetect import detect, DetectorFactory
from openpyxl import Workbook
from openpyxl.styles import Alignment
from openpyxl.worksheet.datavalidation import DataValidation

DetectorFactory.seed = 0
N = 300   # 200 is an acceptable minimum if you're short on time

pool = pd.read_csv("data/processed/comments_vader.csv")
done = pd.read_csv("data/processed/gold_to_label.csv", encoding="utf-8-sig",
                   encoding_errors="replace")
pool = pool[~pool["id"].isin(done["id"])]

def is_english(text):
    try:
        return detect(str(text)) == "en"
    except Exception:
        return False

pool = pool[pool["clean"].apply(is_english)]
print("English, unlabeled comments available:", len(pool))

batch = pool.sample(N, random_state=7)[["id", "clean"]]

wb = Workbook()
ws = wb.active
ws.title = "batch2"
ws.append(["id", "clean", "human_label"])
for row in batch.itertuples(index=False):
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", row.clean)
    ws.append([row.id, text, ""])

ws.column_dimensions["A"].width = 14
ws.column_dimensions["B"].width = 70
ws.column_dimensions["C"].width = 16
for r in ws.iter_rows(min_row=2):
    for cell in r:
        cell.alignment = Alignment(wrap_text=(cell.column == 2), vertical="top")
ws.freeze_panes = "A2"

dv = DataValidation(type="list", formula1='"positive,negative,neutral"', allow_blank=True)
ws.add_data_validation(dv)
dv.add(f"C2:C{N + 1}")

wb.save("data/processed/gold_batch2_to_label.xlsx")
print("saved data/processed/gold_batch2_to_label.xlsx")