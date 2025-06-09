import os
import torch
import torch.nn.functional as F
from transformers.optimization import get_cosine_with_min_lr_schedule_with_warmup
from utils import Evaluator

class Trainer():
    def __init__(self, model, train_dataloader, valid_dataloader, config):
        self.model = model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.device = config.device
        self.epoches = config.epoch
        self.model = self.model.to(self.device)
        self.optim = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
        self.scheduler = get_cosine_with_min_lr_schedule_with_warmup(self.optim, 100, self.epoches * len(self.train_dataloader), min_lr_rate=0.05)
        # checkpointing
        self.model_checkpoint = config.model_checkpoint
        self.save_path = os.path.join(config.model_checkpoint, config.run_name)
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)


    def train(self, ):
        self.train_loss = []
        best_acc = 0  
        for i in range(self.epoches):
            self.model.train()
            for idx, batch in enumerate(self.train_dataloader):
                self.optim.zero_grad()
                batch = [x.to(self.device) for x in batch]
                loss = self.model(batch)
                loss.backward()
                self.optim.step()
                self.scheduler.step()
                print(f"\r[Epoch {i}] batch: {idx}/{len(self.train_dataloader)}  loss: {loss.detach().cpu().item():.4f} lr:{self.scheduler.get_last_lr()[-1]:.4e}", end="")
                if idx % 100 == 0:
                    print()
                    self.train_loss.append(loss.detach().cpu().item())
                
            result_dict = self.eval(self.valid_dataloader)
            print("\n", result_dict)

            if i % 5 == 0:
                print(f"Save on epoch {i}")
                torch.save(self.model.state_dict(), os.path.join(self.save_path, "latest.pt"))
                if result_dict["acc"] > best_acc:
                    best_acc = result_dict["acc"]
                    torch.save(self.model.state_dict(), os.path.join(self.save_path, "best.pt"))

    def eval(self, valid_dataloader):
        self.model.eval()
        evaluator = Evaluator()
        with torch.no_grad():
            for batch in valid_dataloader:
                batch = [x.to(self.device) for x in batch]
                pred = self.model.predict(batch)
                evaluator.update(batch, pred)
        return evaluator.aggregate()

