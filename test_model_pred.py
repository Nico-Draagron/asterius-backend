from models import XGBoostModel

if __name__ == "__main__":
    model = XGBoostModel()
    preds = model.predict_multiple_days(7, 1)
    print("Data       | Valor Predito")
    print("--------------------------")
    for p in preds:
        print(f"{p['date']} | {p['prediction']:.2f}")
