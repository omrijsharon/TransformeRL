import torch


# Define the training loop
def train(model, optimizer, criterion, train_loader, num_epochs, device):

    # Define the scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=5,
        verbose=True,
        min_lr=1e-6
    )
    model.to(device)
    model.train()

    # Initialize loss accumulator
    epoch_loss = 0.0

    # Iterate over the training data
    for i, (states, actions, next_states) in enumerate(train_loader):
        # Move tensors to device
        states = states.to(device)
        actions = actions.to(device)
        next_states = next_states.to(device)

        # Zero the gradients
        optimizer.zero_grad(set_to_none=True)

        # Forward pass
        predicted_states, predicted_actions = model(states, actions)

        # Calculate the loss
        loss = criterion(predicted_states, next_states) + criterion(predicted_actions, actions)

        # Backward pass
        loss.backward()

        # Update the parameters
        optimizer.step()

        # Accumulate the loss for reporting
        epoch_loss += loss.item()

    # Calculate the average loss for the epoch
    epoch_loss /= len(train_loader)

    # Update the learning rate
    scheduler.step(epoch_loss)

    # Print the epoch loss
    print(f"Epoch {epoch + 1} loss: {epoch_loss:.5f}")


print("Training complete.")