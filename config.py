import os
from datetime import datetime
from pathlib import Path


def get_parameters(mode):

    # mode = 'reg' # available modes: 'reg', 'train', 'test'

    hyperparams = {'model_dims': (128, 128, 64), # Dimensiones de entrada al modelo
                   'new_z'     : [2, 2, 2],      # Nuevo tamano de zooms
                   'lr'        : 0.0005,         # Taza de aprendizaje
                   'epochs'    : 20,             # Numero de epocas
                   'batch_size': 4,              # Tama;o del batch
                   'crit'      : 'BCELog',       # Fn de costo. Opciones: 'BCELog', 'CELoss', 'BCE', 'BCELogW'
                   'n_train'   : 19,             # "" Entrenamiento
                   'n_val'     : 2,              # "" Validacion
                   'n_test'    : 2,              # "" Prueba
                   'batchnorm' : False,          # Normalizacion de batch
                   'nclasses'  : 1,              # Numero de clases
                   'thres'     : 0.5,            # Umbral
                   'class_w'   : 5.,             # Peso ponderado de la clase
                   'crop'      : True,           # Recortar o no recortar slices sin fcd del volumen
                   'capas'     : 2               # Numero de capas entrenables     
    }

    labels = {'bgnd': 0, # Image background
              'FCD' : 1, # Focal cortical dysplasia
    }

    iatm_train = os.path.join('./data',
                               'train',
    )

    iatm_val = os.path.join('./data',
                             'val',
    )

    iatm_test = os.path.join('./data',
                             'test',
    )

    mri_fn  = 'Ras_t1.nii.gz'
    mask_fn = 'Ras_msk.nii.gz'

    datasets = {'train': iatm_train, 'val': iatm_val, 'test': iatm_test, 'mri_fn': mri_fn, 'mask_fn': mask_fn}

    folder = './outs/Ex-' + datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    if mode == 'train':

        Path(folder).mkdir(parents=True, exist_ok=True)

        files = {'model' : os.path.join(folder, 'weights'),
                 'losses': os.path.join(folder, 'losses.csv'),
                 't_mets': os.path.join(folder, 'train_metrics.csv'),
                 'v_mets': os.path.join(folder, 'val_metrics.csv'),
                 'params': os.path.join(folder, 'params.txt'),
                 'summary': os.path.join(folder, 'cnn_summary.txt'),
                 'log'    : os.path.join(folder, 'train.log')}

        #PATH_PRETRAINED_MODEL = './pretrained/weights-BCELog-20_eps-100_heads-2023-07-11-_nobn-e19.pth'
        PATH_PRETRAINED_MODEL ='./pretrained/Ex-2023-07-16-01-29-47weights-e14.pth'

        return {'mode'        : mode,
                'data'        : datasets,
                'pretrained'  : PATH_PRETRAINED_MODEL,
                'hyperparams' : hyperparams,
                'files'       : files,
        }

    elif mode == 'test':

        ex = 'Ex-2023-07-17-15-09-21'
        mo = 'weights-e15.pth'

        PATH_TRAINED_MODEL = os.path.join('./outs', ex, mo) 
        PATH_TEST_METS = os.path.join('./outs', ex, mo+'-test_metrics.csv')

        return {'mode'       : mode,
                'data'       : datasets,
                'hyperparams': hyperparams,
                'labels'     : labels,
                'weights'    : PATH_TRAINED_MODEL,
                'test_fn'    : PATH_TEST_METS,
        }

    elif mode == 'assess':

        ex = 'Ex-2023-07-17-15-09-21' 
        mo = 'weights-e15.pth'

        plots_folder = os.path.join('./outs', ex, 'plots'+mo[:-4])

        Path(plots_folder).mkdir(parents=True, exist_ok=True)

        train_losses = os.path.join('./outs', ex, 'losses.csv')
        train_metrics  = os.path.join('./outs', ex, 'train_metrics.csv')
        val_metrics  = os.path.join('./outs', ex, 'val_metrics.csv')
        test_metrics   = os.path.join('./outs', ex, mo+'-test_metrics.csv')

        files = {'train_Loss': train_losses,
                 'train_mets': train_metrics,
                 'val_mets'  : val_metrics,
                 'test_mets' : test_metrics,
        }

        return {'mode'     : mode,
                'labels'   : labels,
                'files'    : files,
                'plots'    : plots_folder,
        }
