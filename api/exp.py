import torch.optim
import torch.nn as nn
import wandb
from tqdm import tqdm
from typing import List,Tuple
class Trainer:
    """Trainer class for the transformer model.

    Args:
        model: The model to train.
        dh: The data handler object.
        batch_size: The batch size.
        lr: The learning rate.
        betas: The betas for the Adam optimiser.
        eps: The epsilon for the Adam optimiser.
        epochs: The number of epochs to train for.
    """

    def __init__(
        self,
        epochs: int = 10,
    ):
        self.criterion = nn.MSELoss()
        self.dataset = None
        self.train_data = None
        self.test_data = None
        self.n_epochs = epochs
        cuda_dev = "0"
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:" + cuda_dev if use_cuda else "cpu")

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

    def train_one_epoch(self, dataloader, epoch_no, losses, optimiser, model, scheduler,disable_tqdm=False):
        epoch_loss = 0
        i = 0
        with tqdm(dataloader, unit="batch", disable=disable_tqdm) as tepoch:
            for idx, data in enumerate(tepoch):
                i += 1
                loss, losses = self._train_one_loop(data=data, losses=losses, model=model, optimiser=optimiser,scheduler=scheduler)
                epoch_loss += loss.detach()
                tepoch.set_description(f"Epoch {epoch_no}")
                tepoch.set_postfix(loss=epoch_loss.item() / i)
        return losses


    def _train_one_loop(
        self, data: torch.utils.data.DataLoader, losses: List[float], model: nn.Module, optimiser: torch.optim.Optimizer,scheduler
    ) -> Tuple[float, List[float]]:

        optimiser.zero_grad()
        data[0] = data[0].double()
        padding_mask = torch.ones((data[0].shape[0], data[0].shape[1])) > 0
        output = model(data[0].to(self.device), padding_mask.to(self.device))
        # data[1] from [B,N] to [B]
        target = torch.argmax(data[1], dim=1).to(self.device)
        loss = self.criterion(output, target)
        loss.backward()
        optimiser.step()
        scheduler.step()
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