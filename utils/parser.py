import argparse
import torch
def get_args():
    parser = argparse.ArgumentParser(description="Welding prediction Training Script Arguments")

    # ====== 数据与路径设置 ======
    parser.add_argument('--data_dir', type=str, default='./abaqus_data/data',
                        help='Directory where the dataset is located')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='Directory to save TensorBoard logs')
    parser.add_argument('--step_nums', type=int, default=44)
    parser.add_argument('--max_samples', type=int, default=100)

    # ====== 模型参数 ======
    parser.add_argument('--model', type=str, default='regression',choices=['regression'])
    parser.add_argument('--H', type=int, default=7)
    parser.add_argument('--W', type=int, default=7)
    parser.add_argument('--C', type=int, default=10)
    parser.add_argument('--embed_dim', type=int, default=20)
    parser.add_argument('--block_type',type=str,default='self',choices=['self','wave'])
    # ====== 训练参数 ======
    parser.add_argument('--nepochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay (L2 regularization)')
    parser.add_argument('--weight_last', type=float, default=2)
    # ====== 采样 ======
    parser.add_argument('--sample',type=str,default="none",choices=["none","random","strided","gumbel"])
    parser.add_argument('--sample_num',type=int,default=10)
    # ====== 优化器与调度器 ======
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'sgd'],
                        help='Optimizer type')
    parser.add_argument('--scheduler', type=str, default='step',
                        choices=['step', 'cosine', 'none'],
                        help='Learning rate scheduler')
    parser.add_argument('--step_size', type=int, default=10,
                        help='StepLR step size')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='LR decay factor for StepLR')

    # ====== 训练设置 ======
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader worker threads')
    parser.add_argument('--enable_wb', type=bool, default=False)
    parser.add_argument("--project_name", type=str, default="Welding_pred")
    parser.add_argument("--run_name", type=str, default="test")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    
    # ====== 模型恢复与测试 ======
    parser.add_argument('--resume', type=str, default='',
                        help='Path to resume checkpoint')
    parser.add_argument('--eval_only', action='store_true',
                        help='Only run evaluation (no training)')

    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args
