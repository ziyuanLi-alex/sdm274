import os
import pandas as pd

def load_data(path="/root/sdm274/final_proj/seeds_dataset.txt"):
    data_path = path

    if os.path.exists(data_path):
        try:
            # 直接读取数据，使用空格分隔符，没有表头
            data = pd.read_csv(data_path, 
                             sep='\s+',  # 使用任意空白字符作为分隔符
                             header=None, # 没有表头
                             engine='python')  # 使用Python引擎来处理复杂分隔符
            
            # 分离特征和标签
            X = data.iloc[:, :7]  # 前7列是特征
            y = data.iloc[:, 7]   # 第8列是标签
            
            return X, y
            
        except Exception as e:
            print(f"Error reading file: {e}")
            raise
    else:
        raise FileNotFoundError(f"Data file not found at {data_path}")
    
def pandas_to_numpy(df):
    return df.values
    

if __name__ == "__main__":
    
    try:
        X, y = load_data("/root/sdm274/final_proj/seeds_dataset.txt")
        print("Data shape:", X.shape)
        print("\nFirst few rows of X:")
        print(X.head())
        print("\nFirst few labels:")
        print(y.head())

        print("\nFeature statistics:")
        print(X.describe())
        print("\nUnique labels:", y.unique())
    except Exception as e:
        print(f"Error: {e}")

    




    