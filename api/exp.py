import torch.optim
import torch.nn as nn
import wandb
from tqdm import tqdm
from typing import List,Tuple
from dataset.dataloader import build_dataset
from model.timeTransformer import E2Epredictor
class Trainer:
    def __init__(
        self,
        args
    ):
        if args.criterion == "mse":
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"no criterion{args.criterion}")
        self.dataset = build_dataset(args)
        self.model = E2Epredictor()
        self.optimiser = torch.optim.AdamW(params=self.model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimiser, T_max=50)

    def fit(self, dataloader: torch.utils.data.DataLoader, model: nn.Module, optimiser: torch.optim.Optimizer,scheduler,test_data):
        losses = []
        model = model.to(self.device)
        model.train()
        model.double()
        for epoch in range(self.n_epochs):
            losses = self.train_one_epoch(dataloader=dataloader, epoch_no=epoch, losses=losses, optimiser=optimiser, model=model,scheduler=scheduler)
            loss_avg = (sum(losses)/len(losses)).cpu().numpy()
            accuracy_,accuracy,errors = self.evaluate(dataloader=test_data,model=model)
            wandb.log({"acc": accuracy_})
            wandb.log({"losses": loss_avg})

    def train_one_epoch(self, epoch_no, losses, disable_tqdm=False):
        epoch_loss = 0
        i = 0
        with tqdm(self.dataset, unit="batch", disable=disable_tqdm) as tepoch:
            for idx, datas in enumerate(tepoch):
                i += 1
                loss, losses = self._train_one_loop(datas=datas, losses=losses)
                epoch_loss += loss.detach()
                tepoch.set_description(f"Epoch {epoch_no}")
                tepoch.set_postfix(loss=epoch_loss.item() / i)
        return losses


    def _train_one_loop(
        self, datas: torch.utils.data.DataLoader, losses: List[float]) -> Tuple[float, List[float]]:

        self.optimiser.zero_grad()
        output = self.model(datas)
        # data[1] from [B,N] to [B]
        target = torch.argmax(datas[1], dim=1).to(self.device)
        loss = self.criterion(output, target)
        loss.backward()
        self.optimiser.step()
        self.scheduler.step()
        losses.append(loss.detach())
        return loss.detach(), losses

    def evaluate(self, dataloader: torch.utils.data.DataLoader, model: nn.Module):
        """Run the model on the test set and return the accuracy."""
        model.eval()
        # pre = []
        # act = []
        # n_correct = 0
        # n_incorrect = 0
        # errors = []
        # for idx, data in enumerate(dataloader):
        #     padding_mask = torch.ones((data[0].shape[0], data[0].shape[1])) > 0
        #     output = model(data[0].to(self.device), padding_mask.to(self.device))
        #     predictions = torch.argmax(output, dim=1)
        #     target = torch.argmax(data[1], dim=1).to(self.device)
        #     pre.append(predictions.cpu().numpy())
        #     act.append(target.cpu().numpy())
        #     incorrect = torch.count_nonzero(predictions - target)
        #     n_incorrect += incorrect.detach()
        #     n_correct += (len(target) - incorrect).detach()
        # accuracy_ = n_correct / (n_correct + n_incorrect)
        # accuracy = confusion_matrix(act,pre)
        # return accuracy_,accuracy,errors