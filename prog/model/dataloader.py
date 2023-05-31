import prog.model.move as dataset 

train_dataloader = dataset.DataLoader(
    dataset.train_dataset, batch_size=2, shuffle=True,
     drop_last=True
)

val_dataloader = dataset.DataLoader(
    dataset.val_dataset, batch_size=2, shuffle=False,
     drop_last=True
)

dataloaders_dict = {
    'train': train_dataloader, 
    'valid': val_dataloader
}
