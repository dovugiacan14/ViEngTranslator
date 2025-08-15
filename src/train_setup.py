import time
import random
import torch
from torch import nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_seq2seq_model(encoder, decoder, train_loader, config):
    loss_history = []
    encoder.to(device)
    decoder.to(device)
    encoder.train()
    decoder.train()

    encoder_optimizer = config.optimizer(encoder.parameters(), lr=config.learning_rate)
    decoder_optimizer = config.optimizer(decoder.parameters(), lr=config.learning_rate)
    criterion = nn.NLLLoss(ignore_index=config.ignore_index, reduction="sum")

    total_start = time.time()
    for epoch in range(config.n_epochs + 1):
        epoch_start = time.time()
        total_loss = 0.0
        total_tokens = 0
        batch_count = 0

        for src_padded, tgt_padded, src_mask, _ in train_loader:
            src_padded = src_padded.to(device)
            tgt_padded = tgt_padded.to(device)
            src_mask = src_mask.to(device)

            batch_size = src_padded.size(0)
            batch_count += 1

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            # encode
            encode_hidden = encoder.init_hidden(batch_size)
            encode_outputs, encode_hidden = encoder(src_padded, encode_hidden)

            # decode
            decode_input = torch.full(
                (batch_size,), config.SOS_token, dtype=torch.long, device=device
            )
            decode_hidden = encode_hidden

            tgt_len = tgt_padded.size(1)
            use_tf = random.random() < config.teacher_forcing_ratio
            non_pad = (tgt_padded != config.ignore_index).sum().item()
            total_tokens += non_pad
            loss = 0.0

            for t in range(tgt_len):
                log_probs, decode_hidden, _ = decoder(
                    decode_input, decode_hidden, encode_outputs, src_mask
                )
                gold = tgt_padded[:, t]
                loss += criterion(log_probs, gold)
                decode_input = gold if use_tf else log_probs.argmax(dim=1)

            # backprop
            loss.backward()
            nn.utils.clip_grad_norm_(
                encoder.parameters(), max_norm=config.clip_grad_norm
            )
            nn.utils.clip_grad_norm_(
                decoder.parameters(), max_norm=config.clip_grad_norm
            )
            encoder_optimizer.step()
            decoder_optimizer.step()

            total_loss += loss.item()

        # end of epoch
        avg_loss = total_loss / max(total_tokens, 1)
        loss_history.append(avg_loss)
        epoch_time = time.time() - epoch_start
        tokens_per_sec = total_tokens / epoch_time

        print(
            f"[Epoch {epoch}/{config.n_epochs}] "
            f"Loss/token: {avg_loss:.4f} | "
            f"Tokens: {total_tokens} | "
            f"Batches: {batch_count} | "
            f"Time: {epoch_time:.2f}s | "
            f"{tokens_per_sec:.1f} tok/s"
        )

    # measure training time
    total_time = time.time() - total_start
    print(f"Training complete in {total_time:.2f}s")

    # save models
    torch.save(
        {"encoder": encoder.state_dict(), "decoder": decoder.state_dict()},
        config.model_save_path,
    )
    print(f"Saved model to {config.model_save_path}")

    # save logging
    with open(config.loss_log_path, "w") as f:
        for loss in loss_history:
            f.write(f"{loss}\n")


def train_model(model, train_loader, val_loader, config):
    model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=config.ignore_index)  # ignore padding
    optimizer = config.optimizer(
        model.parameters(), lr=config.learning_rate, betas=config.betas, eps=config.eps
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.8)
    best_val_loss = float("inf")
    loss_history = []

    for epoch in range(config.n_epochs):
        model.train()
        total_train_loss = 0
        for batch_idx, (src, tgt, _, _) in enumerate(train_loader):
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()

            output = model(src, tgt)  # forward pass
            output = output.reshape(-1, output.size(-1))  # reshape for loss
            target = tgt[:, 1:].reshape(-1)

            loss = criterion(output, target)
            loss.backward()

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=config.clip_grad_norm
            )
            optimizer.step()
            total_train_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        # validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for src, tgt, _, _ in val_loader:
                src, tgt = src.to(device), tgt.to(device)
                output = model(src, tgt)
                output = output.reshape(-1, output.size(-1))
                target = tgt[:, 1:].reshape(-1)

                val_loss = criterion(output, target)
                total_val_loss += val_loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        print(
            f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}"
        )

        loss_history.append((avg_train_loss, avg_val_loss))

        # save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), config.model_save_path)
            print(f"Saved best model to {config.model_save_path}")

        scheduler.step()

    # save history
    with open(config.loss_log_path, "w") as f:
        f.write("Epoch,Train_Loss,Val_Loss\n")
        for i, (train_loss, val_loss) in enumerate(loss_history):
            f.write(f"{i+1},{train_loss:.6f},{val_loss:.6f}\n")
