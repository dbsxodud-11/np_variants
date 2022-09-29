import os
import argparse

from utils.load_save import load_model, load_results_path

def train(args, model):
    print("SUCCESS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=["train"], 
                        default="train", help="choose the mode")
    parser.add_argument('--exp_id', type=str, default='trial')
    parser.add_argument('--gpu', type=str, default='0')

    parser.add_argument('--task', type=str, default="gp_regression")
    parser.add_argument('--model', type=str, default='cnp')
    parser.add_argument('--max_num_points', type=int, default=50)
    parser.add_argument('--train_batch_size', type=int, default=100)
    parser.add_argument('--train_num_samples', type=int, default=4)

    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--print_freq', type=int, default=200)
    parser.add_argument('--eval_freq', type=int, default=5000)
    parser.add_argument('--save_freq', type=int, default=1000)

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model = load_model(args.task, args.model)
    results_path = load_results_path(args.task, args.model, args.exp_id)

    if not os.path.isdir(results_path):
        os.makedirs(results_path)

    if args.mode == 'train':
        train(args, model)

    