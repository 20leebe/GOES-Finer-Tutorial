import torch
import torch.nn.functional as F

def train(
    model, coords, gt, size,
    zero_mean=True,
    loss_fn=torch.nn.MSELoss(),
    lr=1e-3,
    num_epochs=2000,
    steps_til_summary=10
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / num_epochs, 1))

    for epoch in range(1, num_epochs + 1):
        model.train()
        pred = model(coords)
        loss = loss_fn(pred, gt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % steps_til_summary == 0 or epoch == 1 or epoch == num_epochs:
            print(f"Epoch {epoch}/{num_epochs} - Loss: {loss.item():.6f}")

            with torch.no_grad():
                model.eval()
                pred_eval = model(coords)
                if zero_mean:
                    psnr = psnr_fn(pred_eval / 2 + 0.5, gt / 2 + 0.5).item()
                else:
                    psnr = psnr_fn(pred_eval, gt).item()
                print(f"  PSNR: {psnr:.4f}")

    with torch.no_grad():
        final_pred = model(coords).reshape(size)
        if zero_mean:
            final_pred = final_pred / 2 + 0.5

    return final_pred

def train_model_simple(model, optimizer, criterion, coords_norm, gt, n_epochs=500, log_interval=50):
    best_loss = float('inf')
    best_model_state = None
    
    for epoch in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad()
        pred = model(coords_norm)
        loss = criterion(pred, gt)
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_model_state = model.state_dict()

        if epoch % log_interval == 0:
            print(f"Epoch {epoch:03d} — Loss: {loss.item():.6f}")

    # Restore best model weights
    model.load_state_dict(best_model_state)
    return model

def train_model_with_psnr(model, optimizer, criterion, coords_norm, gt, psnr_fn, n_epochs=500, log_interval=50, zero_mean=True):
    best_loss = float('inf')
    best_model_state = None
    loss_history = []
    psnr_history = []

    for epoch in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad()
        pred = model(coords_norm)
        loss = criterion(pred, gt)
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_model_state = model.state_dict()

        if epoch % log_interval == 0:
            with torch.no_grad():
                model.eval()
                pred_eval = model(coords_norm)
                if zero_mean:
                    pred_eval = pred_eval / 2 + 0.5
                    gt_eval = gt / 2 + 0.5
                else:
                    gt_eval = gt
                psnr = psnr_fn(pred_eval, gt_eval).item()
                psnr_history.append((epoch, psnr))

            print(f"Epoch {epoch:03d} — Loss: {loss.item():.6f} — PSNR: {psnr:.4f}")

    # Restore best model weights
    model.load_state_dict(best_model_state)
    return model, loss_history, psnr_history


def train_batches_model_with_psnr(
    model,
    optimizer,
    criterion,
    data_loader,          
    psnr_fn,
    n_epochs=500,
    log_interval=50,
    zero_mean=True,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    best_loss = float('inf')
    best_model_state = None
    loss_history = []
    psnr_history = []

    model.to(device)

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        for coords_batch, gt_batch in data_loader:            
            coords_batch = coords_batch.float().to(device)
            gt_batch = gt_batch.float().to(device)

            optimizer.zero_grad()
            pred = model(coords_batch)
            loss = criterion(pred, gt_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * coords_batch.size(0)  # Weighted by batch size

        avg_loss = epoch_loss / len(data_loader.dataset)
        loss_history.append(avg_loss)

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_state = model.state_dict()

        # Logging
        if epoch % log_interval == 0:
            model.eval()
            with torch.no_grad():
                # Full dataset PSNR (optional: use DataLoader again to do this in chunks)
                all_preds = []
                all_targets = []

                for coords_batch, gt_batch in data_loader:
                    coords_batch = coords_batch.float().to(device)
                    gt_batch = gt_batch.float().to(device)
                    preds = model(coords_batch)

                    if zero_mean:
                        preds = preds / 2 + 0.5
                        gt_batch = gt_batch / 2 + 0.5

                    all_preds.append(preds.cpu())
                    all_targets.append(gt_batch.cpu())

                pred_eval = torch.cat(all_preds, dim=0)
                gt_eval = torch.cat(all_targets, dim=0)
                psnr = psnr_fn(pred_eval, gt_eval).item()
                psnr_history.append((epoch, psnr))

            print(f"Epoch {epoch:03d} — Loss: {avg_loss:.6f} — PSNR: {psnr:.4f}")

    model.load_state_dict(best_model_state)
    return model, loss_history, psnr_history
