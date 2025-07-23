from api.exp import Trainer
from utils.parser import get_args
import wandb
args = get_args()
if args.enable_wb:
    wandb.init(
    project=args.project_name,    # 项目名称
    name=args.run_name, # 自定义 Run 名字
    config=vars(args)
)
trainer = Trainer(args=args)
# losses = trainer._train_one_epoch(epoch_no=1, losses=[])
trainer.fit()