"""
contains functionality for creating Pytorch dataloader's for image classification'
"""
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

num_workers=os.cpu_count()

def create_dataloaders(train_dir:str, test_dir:str,transform:transforms.Compose,batch_size:int,num_workers:int=num_workers):
  """
  create training and testing dataloader
  """
  #use image folder
  train_data=datasets.ImageFolder(train_dir, transform=transform)
  test_data=datasets.ImageFolder(test_dir,transform=transform)

  #class names
  class_names=train_data.classes

  #turn image DataLoaders
  #pin_memroy keeping gpu where possible
  train_dataloader=DataLoader(train_data,batch_size=batch_size,shuffle=True,num_workers=num_workers,pin_memory=True)
  test_dataloader=DataLoader(test_data, batch_size=batch_size,shuffle=False,num_workers=num_workers,pin_memory=True)
  


  return train_dataloader, test_dataloader, class_names




num_workers
