from api.exp import Trainer
from utils.parser import get_args
args = get_args()
trainer = Trainer(args=args)
trainer.train_one_epoch(epoch_no=1,losses=[])
