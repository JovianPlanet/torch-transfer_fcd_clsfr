from datetime import datetime
import pandas as pd
import numpy as np
from torchinfo import summary
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinarySpecificity, BinaryF1Score, BinaryRecall
from tqdm import tqdm

from cnn import CNN2D
from get_data import Unet2D_DS
from utils.write_params import conf_txt, cnnsummary_txt


def train(config):

    torch.cuda.empty_cache()

    start_time = datetime.now()
    print(f'\nHora de inicio: {start_time}')

    print(f"\nNum epochs = {config['hyperparams']['epochs']}, batch size = {config['hyperparams']['batch_size']}, \
        Learning rate = {config['hyperparams']['lr']}\n")

    print(f"Model file name = {config['files']['model']}\n")

    # Crear datasets #
    ds_train = Unet2D_DS(config, 'train', cropds=config['hyperparams']['crop'])
    ds_val   = Unet2D_DS(config, 'val')

    train_dl = DataLoader(
        ds_train,  
        batch_size=config['hyperparams']['batch_size'],
        shuffle=True,
    )

    val_dl = DataLoader(
        ds_val, 
        batch_size=config['hyperparams']['batch_size'],
    )

    print(f'Tamano del dataset de entrenamiento: {len(ds_train)} slices')
    print(f'Tamano del dataset de validacion: {len(ds_val)} slices \n')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    cnn = CNN2D(1, config['hyperparams']['nclasses']).to(device, dtype=torch.double)
    #print(torch.cuda.memory_summary(device=device, abbreviated=False))

    # Se carga el modelo preentrenado
    cnn.load_state_dict(torch.load(config['pretrained']))
    print(f'Modelo preentrenado: {config["pretrained"]}\n')

    summary(cnn)#, input_size=(batch_size, 1, 28, 28))

    print(cnn)

    # Congelar las capas del modelo
    for param in cnn.parameters():
        param.requires_grad = False

    # Anadir nuevas capas entrenables
    cnn.fc2 = nn.Linear(in_features=64*6*6, out_features=config['hyperparams']['nclasses']).to(device, dtype=torch.double)
    cnn.fc1 = nn.Linear(in_features=64*6*6, out_features=64*6*6).to(device, dtype=torch.double)
    
    summary(cnn)

    pos_weight = torch.Tensor([config['hyperparams']['class_w']]).to(device, dtype=torch.double)

    criterion = {'CELoss' : nn.CrossEntropyLoss(),  # Cross entropy loss performs softmax by default
                 'BCELog' : nn.BCEWithLogitsLoss(), # BCEWithLogitsLoss performs sigmoid by default
                 'BCELogW': nn.BCEWithLogitsLoss(pos_weight=pos_weight), # BCEWithLogitsLoss with weighted classes
                 'BCE'    : nn.BCELoss(),
    }

    optimizer = Adam(cnn.parameters(), lr=config['hyperparams']['lr'])
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=4)

    t_acc = BinaryAccuracy().to(device, dtype=torch.double)
    t_pre = BinaryPrecision().to(device, dtype=torch.double)
    t_spe = BinarySpecificity().to(device, dtype=torch.double)
    t_f1s = BinaryF1Score().to(device, dtype=torch.double)
    t_rec = BinaryRecall().to(device, dtype=torch.double) # Sensibilidad

    v_acc = BinaryAccuracy().to(device, dtype=torch.double)
    v_pre = BinaryPrecision().to(device, dtype=torch.double)
    v_spe = BinarySpecificity().to(device, dtype=torch.double)
    v_f1s = BinaryF1Score().to(device, dtype=torch.double)
    v_rec = BinaryRecall().to(device, dtype=torch.double) # Sensibilidad

    conf_txt(config)
    cnnsummary_txt(config, str(summary(cnn)))

    best_loss = 1.0
    best_ep_loss = 0
    best_ep_acc = 0
    best_acc = None

    losses = []
    train_accs = []
    val_accs = []

    for epoch in tqdm(range(config['hyperparams']['epochs'])):  # loop over the dataset multiple times

        #torch.cuda.empty_cache()

        running_loss = 0.0
        epoch_loss   = 0.0
        
        print(f'\n\nEpoca No. {epoch + 1}\n')

        cnn.train()
        for i, data in enumerate(train_dl, 0):

            inputs, labels = data
            inputs = inputs.unsqueeze(1).to(device, dtype=torch.double)
            labels = labels.to(device, dtype=torch.double)

            outputs = cnn(inputs)

            loss = criterion[config['hyperparams']['crit']](outputs.double(), labels.unsqueeze(1)) # BCELoss o DiceLoss
            #loss = criterion(outputs, labels.long()) # Cross entropy loss (multiclase)
            running_loss += loss.item()
            optimizer.zero_grad() # zero the parameter gradients
            loss.backward(retain_graph=True)
            optimizer.step()
            losses.append([epoch, i, loss.item()])

            '''Metricas''' 
            train_accs.append([epoch, i, t_acc.forward(outputs, labels.unsqueeze(1))])
            t_pre.update(outputs, labels.unsqueeze(1))
            t_spe.update(outputs, labels.unsqueeze(1))
            t_f1s.update(outputs, labels.unsqueeze(1))
            t_rec.update(outputs, labels.unsqueeze(1)) # Sensibilidad
            '''Fin metricas'''

            if (i+1) % 10 == 0: 
                print(f'Batch No. {i+1}')
                print(f'Loss = {running_loss/(i+1):.3f}')
                print(f'Accuracy (prom) = {t_acc.compute():.3f}')
                print(f'Precision (prom) = {t_pre.compute():.3f}, Especificidad = {t_spe.compute():.3f}')
                print(f'F1 Score (prom) = {t_f1s.compute():.3f}, Sensibilidad = {t_rec.compute():.3f}')


        before_lr = optimizer.param_groups[0]["lr"]
        if (epoch + 1) % 3 == 0:
            scheduler.step()
        after_lr = optimizer.param_groups[0]["lr"]

        epoch_loss = running_loss/(i + 1)  
        ep_val_acc = 0  

        with torch.no_grad():
            cnn.eval()
            print(f'\nValidacion\n')
            for j, testdata in enumerate(val_dl):
                x, y = testdata
                x = x.unsqueeze(1).to(device, dtype=torch.double)
                y = y.to(device, dtype=torch.double)

                outs  = cnn(x)
                #probs = nn.Softmax(dim=1) # Softmax para multiclase
                #probs = nn.Sigmoid()       # Sigmoid para biclase
                #pval  = probs(outs) 
                #preds = torch.where(pval>config['hyperparams']['thres'], 1., 0.)
                #preds = torch.argmax(pval, dim=1)

                '''Metricas''' 
                val_accs.append([epoch, j, v_acc.forward(outs, y.unsqueeze(1))])
                v_pre.update(outs, y.unsqueeze(1))
                v_spe.update(outs, y.unsqueeze(1))
                v_f1s.update(outs, y.unsqueeze(1))
                v_rec.update(outs, y.unsqueeze(1)) # Sensibilidad
                '''Fin metricas'''

                if (j+1) % 10 == 0: 
                    print(f'Accuracy promedio hasta batch No. {j+1} = {v_acc.compute():.3f}')
                    print(f'Precision (prom) = {v_pre.compute():.3f}, Especificidad = {v_spe.compute():.3f}')
                    print(f'F1 Score (prom) = {v_f1s.compute():.3f}, Sensibilidad = {v_rec.compute():.3f}')

                # if (j+1) % 16 == 0:
                #     if torch.any(y):
                #         plot_overlays(x.squeeze(1), 
                #                       y, 
                #                       preds.squeeze(1), 
                #                       mode='save', 
                #                       fn=f"{config['files']['pics']}-epoca_{epoch + 1}-b{j}.pdf")


        ep_val_acc  = v_acc.compute()

        if epoch == 0:
            best_loss = epoch_loss
            best_ep_loss = epoch + 1
            best_acc = ep_val_acc
            best_ep_acc = epoch + 1

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_ep_loss = epoch + 1

        print(f'\nEpoch Metrics (Training):')
        print(f'Loss = {epoch_loss:.3f}, Best loss = {best_loss:.3f} (epoca {best_ep_loss})')
        print(f'Accuracy = {t_acc.compute():.3f}')
        print(f'Precision = {t_pre.compute():.3f}, Especificidad = {t_spe.compute():.3f}')
        print(f'F1 Score = {t_f1s.compute():.3f}, Sensibilidad = {t_rec.compute():.3f}\n')

        # if epoch_val_dice > best_dice:
        #     best_dice = epoch_val_dice
        #     best_epoch_dice = epoch + 1
        #     #print(f'\nUpdated weights file!')

        if ep_val_acc > best_acc:
            best_acc = ep_val_acc
            best_ep_acc = epoch + 1
        
        torch.save(cnn.state_dict(), config['files']['model']+f'-e{epoch+1}.pth')

        print(f'\nEpoch Metrics (Validation):')
        print(f'Accuracy = {ep_val_acc:.3f}, Best accuracy = {best_acc:.3f} (epoca {best_ep_acc})')
        print(f'Precision = {v_pre.compute():.3f}, Especificidad = {v_spe.compute():.3f}')
        print(f'F1 Score = {v_f1s.compute():.3f}, Sensibilidad = {v_rec.compute():.3f}\n')
        print(f'lr = {before_lr} -> {after_lr}\n')

        t_acc.reset()
        t_pre.reset()
        t_spe.reset()
        t_f1s.reset()
        t_rec.reset() # Sensibilidad

        v_acc.reset()
        v_pre.reset()
        v_spe.reset()
        v_f1s.reset()
        v_rec.reset() # Sensibilidad

    df_loss = pd.DataFrame(losses, columns=['Epoca', 'Batch', 'Loss'])
    df_loss = df_loss.assign(id=df_loss.index.values)
    df_loss.to_csv(config['files']['losses'])

    df_train_acc = pd.DataFrame(train_accs, columns=['Epoca', 'Batch', 'Accuracy'])
    df_train_acc = df_train_acc.assign(id=df_train_acc.index.values)
    df_train_acc.to_csv(config['files']['t_accus'])

    df_val_acc = pd.DataFrame(val_accs, columns=['Epoca', 'Batch', 'Accuracy'])
    df_val_acc = df_val_acc.assign(id=df_val_acc.index.values)
    df_val_acc.to_csv(config['files']['v_accus'])

    print(f'\nFinished training. Total training time: {datetime.now() - start_time}\n')


