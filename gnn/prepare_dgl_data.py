import pandas as pd


RAW_PATH = './data/raw/gnn_data_small.csv'












if __name__ == "__main__":
    data = pd.read_csv(RAW_PATH).reset_index()

    num_of_data = len(data)
    val_index = int(num_of_data * 0.8)
    test_index = int(num_of_data * 0.9)

    print('Processing data...')
    train_data = data[:val_index]
    val_data = data[val_index:test_index]
    test_data = data[test_index:]
    train_data.to_csv('./data/raw/gnn_data_dgl_train_small.csv')
    val_data.to_csv('./data/raw/gnn_data_dgl_val_small.csv')
    test_data.to_csv('./data/raw/gnn_data_dgl_test_small.csv')