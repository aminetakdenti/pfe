import pandas as pd

class DataLoader:
    def __init__(self, csv_path):
        csv_data = pd.read_csv(csv_path)

data_loader = DataLoader("./data/ids_data.csv")
print(data_loader)