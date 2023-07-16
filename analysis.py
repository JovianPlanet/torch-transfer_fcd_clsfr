import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def assess(config):

    print(f"\nAnalisis del modelo: {config['files']['train_Loss']}\n")

    df_loss  = pd.read_csv(config['files']['train_Loss'], index_col=[0]) 
    df_train = pd.read_csv(config['files']['train_mets'], index_col=[0])
    df_val   = pd.read_csv(config['files']['val_mets'], index_col=[0])
    df_test  = pd.read_csv(config['files']['test_mets'], index_col=[0])

    mean_losses = df_loss.groupby("Epoca")["Loss"].mean()
    print(f'Average loss last training epoch = {mean_losses[19]:.3f}\n')

    # plt.scatter(range(20), mean_losses, marker='o')
    # plt.title('Función de costo: Entropía cruzada binaria\nÉpoca Vs. Costo')
    # plt.show()
    #plt.imsave()

    # fixCSV(config)

    mean_mets = df_train.groupby("Epoca")["Accuracy"].mean()
    print(f'Average Accuracy (Train) = {mean_mets[19]:.3f}')

    plt.scatter(range(20), mean_mets, marker='o')
    plt.title('Accuracy promedio (train)')
    plt.xticks(np.arange(0, 20, step=1))
    plt.show()

    #fixCSV(config['files']['val_mets'])

    mean_mets = df_val["Accuracy"].mean()
    print(f'Average Accuracy (Eval) = {mean_mets:.3f}')

    acxep = df_train.groupby("Epoca")["Accuracy"].mean()
    plt.scatter(range(len(acxep)), acxep, marker='o')
    plt.title('Accuracy promedio (Eval)')
    #plt.xticks(np.arange(0, 20, step=1))
    plt.show()

    cols = df_test.columns.values.tolist()

    for c in cols[2:-1]:
        #mexep = df_test.groupby("Epoca")[c].mean()
        plt.scatter(range(len(df_test[c])), df_test[c], marker='o')
        plt.title(f'{c} promedio (Eval)')
        #plt.xticks(np.arange(0, 20, step=1))
        plt.show()


def fixCSV(path):
    df_train = pd.read_csv(path, index_col=[0])

    for i, item in enumerate(df_train['Accuracy']):
        new = float(item.replace('tensor(', '').replace(", device='cuda:0')", ''))
        #print(new)
        df_train.at[i, 'Accuracy'] = new

    print('\n')
    for item in df_train['Accuracy'][-20:]:
        print(item)

    df_train = df_train.assign(id=df_train.index.values)
    df_train.to_csv(path)

    # mean_mets = df_train.groupby("Epoca")["Accuracy"].mean()
    # print(f'Average Accuracy (eval) = {mean_mets[19]:.3f}')