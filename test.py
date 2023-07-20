import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinarySpecificity, BinaryF1Score, BinaryRecall
import pandas as pd

from get_data import Cnn2D_Ds
from cnn import Cnn2D
# from utils.plots import plot_batch_full, plot_overlays


def test(config):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing {device} device\n")

    test_ds = Cnn2D_Ds(config, 'test')

    test_mris = DataLoader(
        test_ds, 
        batch_size=4
    )

    cnn = Cnn2D(1, config['hyperparams']['nclasses']).to(device, dtype=torch.double)
    cnn.load_state_dict(torch.load(config['weights']))

    print(f'Test del modelo {config["weights"]}\n')

    acc = BinaryAccuracy().to(device, dtype=torch.double)    # Accuracy
    pre = BinaryPrecision().to(device, dtype=torch.double)   # Precision
    spe = BinarySpecificity().to(device, dtype=torch.double) # Specificity
    f1s = BinaryF1Score().to(device, dtype=torch.double)     # F1 Score
    rec = BinaryRecall().to(device, dtype=torch.double)      # Recall (Sensibilidad) 

    metrics = []

    # Freeze gradients
    with torch.no_grad():
        cnn.eval()
        for i, data in enumerate(test_mris):
            images, labels = data
            images = images.unsqueeze(1).to(device, dtype=torch.double)
            labels = labels.to(device, dtype=torch.double)

            # calculate outputs by running images through the network
            outputs = cnn(images)

            '''Metricas''' 
            metrics.append([i, 
                            acc.forward(outputs, labels.unsqueeze(1)).item(),
                            pre.forward(outputs, labels.unsqueeze(1)).item(),
                            spe.forward(outputs, labels.unsqueeze(1)).item(),
                            f1s.forward(outputs, labels.unsqueeze(1)).item(),
                            rec.forward(outputs, labels.unsqueeze(1)).item()]
            )
            '''Fin metricas'''
            
            if (i + 1) % 10 == 0:
                print(f'\nMetricas promedio hasta el batch No. {i+1}:')
                print(f'Accuracy      = {acc.compute():.3f}')
                print(f'Precision     = {pre.compute():.3f}')
                print(f'Especificidad = {spe.compute():.3f}')
                print(f'F1 Score      = {f1s.compute():.3f}')
                print(f'Sensibilidad  = {rec.compute():.3f}\n')

            #plot_batch_full(images.squeeze(1), labels, preds.squeeze(1))
            #if torch.any(labels):
            #plot_overlays(images.squeeze(1), labels, preds.squeeze(1))
            
    print(f'\nMetricas totales:')
    print(f'Accuracy      = {acc.compute():.3f}')
    print(f'Precision     = {pre.compute():.3f}')
    print(f'Especificidad = {spe.compute():.3f}')
    print(f'F1 Score      = {f1s.compute():.3f}')
    print(f'Sensibilidad  = {rec.compute():.3f}\n')
    df_metrics = pd.DataFrame(metrics, columns=['Batch', 'Accuracy', 'Precision', 'Specificity', 'F1Score', 'Recall'])
    df_metrics = df_metrics.assign(id=df_metrics.index.values)
    df_metrics.to_csv(config['test_fn'])
