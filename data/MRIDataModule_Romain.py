import os

import h5py  # pour lire les fichiers h5
import lightning as L
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


# permet de gérer les données sous forme d’objets que PyTorch peut utiliser pour entraîner les modèle
class MRIDataset(Dataset):
    def __init__(self, dataframe, dataset_path):
        self.dataframe = dataframe
        self.dataset_path = dataset_path
    
    # return le nb d'images dans le dataset
    def __len__(self):
        return len(self.dataframe)
    
    # lit une image et son label à partir d’un fichier .h5. Elle normalise l’image et retourne un tenseur compatible avec PyTorch
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        h5_path = os.path.join(self.dataset_path, row['slice_path'])
        label = row['target']
        
        with h5py.File(h5_path, 'r') as f:
            # Accéder à l'image à partir de la clé 'image'
            if 'image' in f:
                image = np.array(f['image'][:])
            else:
                raise KeyError(f"Key 'image' not found in {h5_path}")
            
            # Optionnel : si tu veux aussi utiliser le masque
            if 'mask' in f:
                mask = np.array(f['mask'][:])
                # Utiliser ou afficher le masque si besoin
                #print(f"Mask shape: {mask.shape}")
        
        # Normalisation des images (optionnelle)
        image = image / np.max(image)
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Pour garder un canal (noir et blanc)
        
        return image, torch.tensor(label, dtype=torch.float32)

class MRIDataModule(L.LightningDataModule):
    def __init__(self, dataset_path, batch_size=4, train_val_test_split=(0.8, 0.1, 0.1), seed=23):
        super().__init__()
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.train_val_test_split = train_val_test_split
        self.seed = seed

    # lit le csv survival_info.csv et les fichiers h5 pour associer les informations de survie aux images
    def prepare_data(self):
        # chargement des informations de survie
        survival_info = pd.read_csv(os.path.join(self.dataset_path, 'survival_info.csv'))

        # calculer la moyenne des jours de survie
        mean_survival_days = survival_info['Survival_days'].mean()

        # créer la colonne 'target' (0 pour traitement efficace, 1 pour traitement inefficace)
        survival_info['target'] = (survival_info['Survival_days'] < mean_survival_days).astype(int)

        # recup les chemins des fichiers h5 et en extraire le volume
        h5_files = []
        for root, _, files in os.walk(self.dataset_path):
            for file in files:
                if file.endswith('.h5'):
                    h5_files.append(os.path.join(root, file))

        # associer les fichiers h5 aux volumes dans le fichier survival_info
        data_list = []
        for h5_file in h5_files:
            # extraction de l'identifiant du volume à partir du nom du fichier (par exemple, volume_108)
            # (extraction robuste basée sur les motifs de nommage des fichiers)
            filename = os.path.basename(h5_file)
            try:
                volume_id = int(filename.split('_')[1])  # je pars du principe que le second élément après 'volume_' est l'ID du volume
            except ValueError:
                print(f"Erreur en extrayant l'identifiant de volume du fichier: {filename}")
                continue

            # recherche de la ligne correspondante dans le fichier survival_info
            matching_row = survival_info[survival_info['Brats20ID'] == f'BraTS20_Training_{volume_id:03d}']
            if not matching_row.empty:
                data_list.append({'slice_path': h5_file, 'target': matching_row['target'].values[0]})

        # créer un dataframe à partir des données associées
        self.data = pd.DataFrame(data_list)

    def setup(self, stage=None):
        train_val_data, self.test_data = train_test_split(self.data, test_size=self.train_val_test_split[2], random_state=self.seed)
        self.train_data, self.val_data = train_test_split(train_val_data, test_size=self.train_val_test_split[1] / (1 - self.train_val_test_split[2]), random_state=self.seed)

        self.train_dataset = MRIDataset(self.train_data, self.dataset_path)
        self.val_dataset = MRIDataset(self.val_data, self.dataset_path)
        self.test_dataset = MRIDataset(self.test_data, self.dataset_path)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)