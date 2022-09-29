import os
import argparse

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils.log import RunningAverage
from utils.load_save import load_task, load_model, load_trained_model, load_results_path, load_logger, save_info, save_model

def train(args, task, model):
    save_info(args)

    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_steps)
    ravg = RunningAverage()

    logger = load_logger(args.results_path, "train")
    logger.info(f'Total number of parameters: {sum(p.numel() for p in model.parameters())}\n')

    for step in range(1, args.num_steps + 1):
        model.train()
        optimizer.zero_grad()
        batch = task.sample(batch_size=args.train_batch_size,
                            max_num_points=args.max_num_points,
                            device="cuda")
        out = model(batch, num_samples=args.train_num_samples)
        out.loss.backward()
        optimizer.step()
        scheduler.step()

        for key, val in out.items():
            ravg.update(key, val)

        if step % args.print_freq == 0:
            log = f'{args.model}:{args.exp_id}\tstep: {step}\t{ravg.info()}'
            logger.info(log)
            ravg.reset()
        
        if step % args.save_freq == 0 or step == args.num_steps:
            save_model(args.results_path, model, optimizer, scheduler, step)

def evaluate(args, task, model):
    model = load_trained_model(args.task, args.model, args.exp_id)

    logger = load_logger(args.results_path, "eval")

    torch.manual_seed(args.eval_seed)
    torch.cuda.manual_seed(args.eval_seed)
    
    ravg = RunningAverage()
    
    model.eval()
    with torch.no_grad():
        for _ in range(args.eval_num_batches):
            batch = task.sample(batch_size=args.eval_batch_size,
                                max_num_points=args.max_num_points,
                                device="cuda")
            out = model(batch, num_samples=args.eval_num_samples)
            
            for key, val in out.items():
                ravg.update(key, val)
    
    log = f'{args.model}:{args.exp_id}\t{ravg.info()}'
    logger.info(log)    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=["train", "eval"], 
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

    parser.add_argument('--eval_seed', type=int, default=42)
    parser.add_argument('--eval_num_batches', type=int, default=3000)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--eval_num_samples', type=int, default=50)

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    task = load_task(args.task)
    model = load_model(args.task, args.model)
    args.results_path = load_results_path(args.task, args.model, args.exp_id)

    if args.mode == "train":
        train(args, task, model)
    if args.mode == "eval":
        evaluate(args, task, model)

    