from utils import now_time, load_data, evaluate_exp
from bperp import BPERp, Batchify
from transformers import AdamW
import numpy as np
import argparse
import torch


parser = argparse.ArgumentParser(description='Bayesian Personalized Explanation Ranking enhanced by BERT (BPER+)')
parser.add_argument('--data_dir', type=str, default=None,
                    help='directory for loading data')
parser.add_argument('--index_dir', type=str, default=None,
                    help='load indexes')
parser.add_argument('--dimension', type=int, default=20,
                    help='number of latent factors')
parser.add_argument('--model_name', type=str, default='bert-base-uncased',
                    help='BERT type or folder to load pre-downloaded, see https://huggingface.co/transformers/pretrained_models.html')
parser.add_argument('--hidden_size', type=int, default=768,
                    help='hidden size of BERT (see the above webpage)')
parser.add_argument('--seq_max_len', type=int, default=10,
                    help='number of words to use for each explanation')
parser.add_argument('--lr', type=float, default=0.00001,
                    help='learning rate')
parser.add_argument('--epochs', type=int, default=5,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--top_k', type=int, default=10,
                    help='select top k to evaluate')
parser.add_argument('--mu_on_user', type=float, default=0.7,
                    help='ratio on user for score prediction (-1 means from 0, 0.1, ..., 1)')
args = parser.parse_args()

if args.data_dir is None:
    parser.error('--data_dir should be provided for loading data')
if args.index_dir is None:
    parser.error('--index_dir should be provided for loading data splits')

print('-' * 40 + 'ARGUMENTS' + '-' * 40)
for arg in vars(args):
    print('{:40} {}'.format(arg, getattr(args, arg)))
print('-' * 40 + 'ARGUMENTS' + '-' * 40)

if torch.cuda.is_available():
    if not args.cuda:
        print(now_time() + 'WARNING: You have a CUDA device, so you should probably run with --cuda')
device = torch.device('cuda' if args.cuda else 'cpu')

print(now_time() + 'Loading data')
train_tuple_list, test_tuple_list, user2items_test, text_list, user2index, item2index, exp2index = load_data(args.data_dir, args.index_dir)
data = Batchify(train_tuple_list, text_list, args.model_name, args.seq_max_len, args.batch_size, device)
user_test, item_test, exp_test, text_test, mask_test = data.prediction_batch(test_tuple_list)
model = BPERp(args.batch_size, len(user2index), len(item2index), len(exp2index), args.model_name, args.hidden_size, args.dimension).to(device)
optimizer = AdamW(model.parameters(), lr=args.lr)

print(now_time() + 'Training')
for epoch in range(1, args.epochs + 1):
    model.train()
    total_loss = 0.
    total_sample = 0
    while True:
        user, item, exp, text, mask = data.next_batch()
        batch_size = user.size(0)
        optimizer.zero_grad()
        loss = model(user, item, exp, text, mask)
        loss.backward()
        optimizer.step()

        total_loss += batch_size * s_loss.item()
        total_sample += batch_size
        if data.step == data.total_step:
            break
    print(now_time() + 'epoch {} loss: {:4.4f}'.format(epoch, total_loss / total_sample))


print(now_time() + 'Evaluating on test set')
if args.mu_on_user == -1:
    mus = np.arange(0, 1.1, 0.1)
else:
    mus = [args.mu_on_user]
model.eval()
with torch.no_grad():
    for mu in mus:
        print(now_time() + '{:1.1f} on users'.format(mu))
        test_tuple_predict = model(user_test, item_test, exp_test, text_test, mask_test, args.top_k, mu)
        ndcg, precision, recall, f1 = evaluate_exp(test_tuple_list, test_tuple_predict)
        print(now_time() + 'NDCG: {}'.format(ndcg))
        print(now_time() + 'Precision: {}'.format(precision))
        print(now_time() + 'Recall: {}'.format(recall))
        print(now_time() + 'F1: {}'.format(f1))
