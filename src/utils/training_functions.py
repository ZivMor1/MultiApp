from src.utils.evaluation_functions import evaluate_recommender
import torch
import numpy as np
from tqdm.notebook import tqdm
import torch.nn as nn


# def create_dataloader(x, y, batch_size=32, shuffle=True):
#     g = torch.Generator().manual_seed(42)
#     dataset = TensorDataset(x, y)
#     loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=g)
#     return loader
#
#
# def create_val_hit10rate_dataloader(x, y, flows_id, src_padding_masking, batch_size=32, shuffle=True):
#     g = torch.Generator().manual_seed(42)
#     dataset = TensorDataset(x.float(), y.long(), flows_id.long(), src_padding_masking.bool())
#     loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=g)
#     return loader


def train_step(model, data_loader, loss_fn, optimizer, device):
    model.train()  # Set the model to training mode
    for inputs, targets in tqdm(data_loader):
        try:
            inputs, targets = inputs.to(device), targets.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

        except Exception as e:
            torch.save(inputs, 'fail_inputs.pt')
            torch.save(targets, 'fail_targets.pt')
            print(inputs.shape)
            print(inputs.max())
            raise ValueError('error in train_step')


def train_model(model,
                train_loader, val_loader, epochs,
                checkpoints_dir_path, initial_epoch=0, learning_rate=1e-3,
                weight_decay=1e-2, save_checkpoints=True, optimizer=None,
                apply_scheduler=False, device=None):
    # Move model to the appropriate compute device
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    best_val_metric = 0

    # Define loss function and optimizer
    scheduler = None
    loss_fn = nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        if apply_scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60, eta_min=1e-5)

    train_losses, val_losses = [], []

    all_epochs = initial_epoch + epochs

    # Training loop
    for epoch in range(epochs):
        running_epoch = initial_epoch + epoch
        print(f'\n\n**Epoch #{running_epoch}')
        train_step(model, train_loader, loss_fn, optimizer, device)

        print('finished_train_step, checking loss on train')
        train_loss, train_top10_hit_rate, train_ndcg_10 = evaluate_recommender(model,
                                                                               train_loader,
                                                                               device, k=10, loss_fn=loss_fn,
                                                                               n_sampling_eval=100)
        train_losses.append(train_loss)

        if scheduler is not None:
            scheduler.step()
            print(f"lr now → {scheduler.get_last_lr()[0]:.6f}")

        print('checking loss on validation')
        val_loss, val_top10_hit_rate, val_ndcg_10 = evaluate_recommender(model, val_loader, device, k=10,
                                                                         loss_fn=loss_fn, n_sampling_eval=100)
        val_losses.append(val_loss)
        if best_val_metric < val_top10_hit_rate:
            best_val_metric = val_top10_hit_rate

        message_end_epoch = f"""Train Epoch {running_epoch + 1}/{all_epochs} Stats:'
        *Train Loss: {train_loss:.4f}, Train Top10Hit: {train_top10_hit_rate:.4f}, Val NDCG@10:{train_ndcg_10:.4f};'
        *Val Loss: {val_loss:.4f},     Val Top10Hit: {val_top10_hit_rate:.4f},     Val NDCG@10:{val_ndcg_10:.4f};'
        """

        print(message_end_epoch)

        # checkpoints
        if save_checkpoints:
            torch.save(model.state_dict(),
                       f'{checkpoints_dir_path}/epoch_{running_epoch}_{round(val_loss, 2)}_val_loss.pth')
            np.save(f'{checkpoints_dir_path}/train_losses.npy', np.array(train_losses))
            np.save(f'{checkpoints_dir_path}/val_losses.npy', np.array(val_losses))

    print("Training complete!")
    return best_val_metric
