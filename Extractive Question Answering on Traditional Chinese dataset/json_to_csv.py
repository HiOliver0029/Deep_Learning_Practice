import json
test = json.load(open("test.json", encoding="utf-8"))
preds = json.load(open("qa_output/predict/predict_predictions.json", encoding="utf-8"))
with open("prediction.csv", "w", encoding="utf-8") as fw:
    fw.write("id,answer\n")
    for ex in test:
        _id = ex["id"]
        ans = preds.get(_id, "")
        # basic CSV quoting
        safe = '"' + str(ans).replace('"','""') + '"'
        fw.write(f"{_id},{safe}\n")
print("written prediction.csv")