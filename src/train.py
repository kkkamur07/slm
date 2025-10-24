from __future__ import annotations
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import hydra
from omegaconf import DictConfig, OmegaConf


from src.components.byteDataset import ByteDataset
from src.components.bytetokenizer import ByteTokenizer
from src.components.model.gpt import GPT

def estimate_loss(
    model : GPT,
    train_loader : DataLoader,
    val_loader : DataLoader,
    eval_iters : int,
    device : torch.device
) -> dict[str, float] : 
    
    model.eval()
    out = {}
    
    with torch.no_grad():
        for split, loader in [("train", train_loader), ("val", val_loader)] : 
            losses = []
            loader_iter = iter(loader)
            
            for _ in range(min(eval_iters, len(loader))) : 
                inputs, target = next(loader_iter)
                inputs, target = inputs.to(device), target.to(device)
                _, loss = model(inputs, target)
                losses.append(loss.item())
            
            out[split] = sum(losses) / len(losses) if losses else 0.0
            
    model.train()
    return out
    
    
def train(
    path,
    seq_len,
    split,
    n_layer,
    heads,
    d_model,
    dropout,
    lr,
    weight_decay,
    amp,
    output_dir,
    train_steps,
    grad_clip,
    compile,
    eval_intervals,
    eval_iters,
    sample_every,
    sample_tokens,
    temperature,
    top_k, 
    top_p,
    batch_size
    ) : 
    
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    tokenizer = ByteTokenizer()
    
    train_dataset = ByteDataset(
        path=path,
        seq_len=seq_len,
        split=split,
        train=True
    )
    
    val_dataset = ByteDataset(
        path = path,
        seq_len=seq_len,
        split=split,
        train=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Initialize the model
    model = GPT(
        vocab_size=tokenizer.vocabSize,
        seq_len=seq_len, # This is essentially the batch size.
        n_layer=n_layer,
        heads = heads,
        d_model = d_model,
        dropout=dropout
    ).to(device)
    
    if compile :
        model = torch.compile(model)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = lr,
        betas = (0.9,0.95),
        weight_decay=weight_decay   
    )
    
    # Gradient Scaler
    scaler = torch.amp.GradScaler(
        enabled = (amp and device == "cuda")
    )
    
    # Training loop
    os.makedirs(output_dir, exist_ok=True)
    
    best_val_loss = float("inf")
    train_iter = iter(train_loader)
    t0 = time.time()
    model.train()
    
    for step in range(1, train_steps + 1) : 
        inputs, target = next(train_iter)
        inputs, target = inputs.to(device), target.to(device)
        
        # Forward pass
        with torch.amp.autocast(device_type=device, enabled=(amp and device == "cuda")) : 
            _, loss = model(inputs, target)
        
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        
        if grad_clip >0 : 
            scaler.unscale_(optimizer=optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        scaler.step(optimizer=optimizer)
        scaler.update()
        
        # Loggings
        
        if step % 50 == 0 : 
            t1 = time.time()
            print(f"Step {step} / {train_steps} : loss = {loss.item():.4f} | time per step = {(t1 - t0) / 50 :.4f} sec")
            
        if step % eval_intervals == 0 : 
            losses = estimate_loss(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                eval_iters=eval_iters,
                device=device
            )
            
            print(f"Step {step} / {train_steps} : Train Loss = {losses['train']:.4f} | Val Loss = {losses['val']:.4f}")
            
            # Save the model if the validation loss is the best we've seen so far.
            if losses["val"] < best_val_loss : 
                best_val_loss = losses["val"]
                torch.save({
                    "model" : model.state_dict(),
                    "optimizer" : optimizer.state_dict(),
                    "step" : step,
                    "config" : {
                        "seq_len" : seq_len,
                        "n_layer" : n_layer,
                        "heads" : heads,
                        "d_model" : d_model,
                        "dropout" : dropout
                    }
                }, f"{output_dir}/best_model.pth")
                print("Saved Best Model")
                
        if sample_every > 0 and step % sample_every == 0 : 
            start = torch.randint(
                low = 0, 
                high=len(train_dataset) - seq_len -1,
                size=(1,)
            ).to(device)
            seed, _ = train_dataset[start]
            seed = seed.unsqueeze(0).to(device)
            
            generated = model.generate(
                idx= seed,
                max_new_tokens = sample_tokens,
                temperature = temperature,
                top_k = top_k,
                top_p = top_p 
            )
            
            text = tokenizer.decode(generated[0].cpu())
            print(f"\n{'='*50}\n{text}\n{'='*50}\n")
            
        model.train()
        
    torch.save({
        "model" : model.state_dict(),
        "optimizer" : optimizer.state_dict(),
        "step" : step,
        "config" : {
            "seq_len" : seq_len,
            "n_layer" : n_layer,
            "heads" : heads,
            "d_model" : d_model,
            "dropout" : dropout
        }
    }, f"{output_dir}/final_model.pth")
    print("Training Complete")
    

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg : DictConfig) :
    print(OmegaConf.to_yaml(cfg))
    train(
        path=cfg.data.path,
        seq_len=cfg.data.seq_len,
        split=cfg.data.split,
        
        n_layer=cfg.model.n_layer,
        heads=cfg.model.heads,
        d_model=cfg.model.d_model,
        dropout=cfg.model.dropout,
        
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        amp=cfg.training.amp,
        batch_size=cfg.training.batch_size,
        output_dir=cfg.training.output_dir,
        train_steps=cfg.training.steps,
        grad_clip=cfg.training.grad_clip,
        compile=cfg.training.compile,
        eval_iters=cfg.training.eval_iters,
        eval_intervals=cfg.training.eval_intervals,
        
        sample_every=cfg.sampling.sample_every,
        sample_tokens=cfg.sampling.sample_tokens,
        temperature=cfg.sampling.temperature,
        top_k=cfg.sampling.top_k,
        top_p=cfg.sampling.top_p
    )
    
if __name__ == "__main__" :
    main()

            
            
            
        
        
        