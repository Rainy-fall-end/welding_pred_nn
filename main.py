from api.exp import Trainer
from utils.parser import get_args

args = get_args()
trainer = Trainer(args=args)
# losses = trainer._train_one_epoch(epoch_no=1, losses=[])
trainer.fit()