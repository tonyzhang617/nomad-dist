import numpy as np
import faiss
from tqdm import tqdm
from argparse import ArgumentParser
from os.path import isfile
import os

parser = ArgumentParser('Index Trainer')
parser.add_argument('--paths', type=str, nargs='+', default=['assets/codellama-7b-wikitext2-valid-keys'])
parser.add_argument('--save_path', type=str, default='assets/codellama-7b-wikitext2-valid-codebooks')
parser.add_argument('--overwrite', action='store_true')
parser.add_argument('--max_runs', type=int, default=1000)
parser.add_argument('--d_sub', type=int, default=1)
parser.add_argument('--range', type=int, nargs=2, default=[0, 1600])
parser.add_argument('--niter', type=int, default=32)
parser.add_argument('--dim', type=int, default=128)
parser.add_argument('--first_n', type=int, default=None)
parser.add_argument('--factory', type=str, default=None)
args = parser.parse_args()
print(args)

if os.path.exists(args.save_path) and not args.overwrite:
    print(f'{args.save_path} already exists, exiting')
    exit()
else:
    os.makedirs(args.save_path, exist_ok=True)

pbar = tqdm(range(args.range[0], args.range[1]))
for i in pbar:
    vecs = []
    for p in args.paths:
        for run in range(args.max_runs):
            index_file_name = f'{p}/r{run}_i{i}.index'
            if not isfile(index_file_name):
                break
            index_flat = faiss.read_index(index_file_name)
            if isinstance(args.first_n, int):
                vec = index_flat.reconstruct_n(0, min(index_flat.ntotal, args.first_n))
            else:
                vec = index_flat.reconstruct_n(0, index_flat.ntotal)
            vecs.append(vec)

    if len(vecs) == 0:
        print(f'No vectors found for index {i}, exiting')
        exit()

    x_train = np.concatenate(vecs, axis=0)
    pbar.set_description(f'num_train_vectors={x_train.shape[0]}')
    if isinstance(args.factory, str):
        index_pq = faiss.index_factory(args.dim, args.factory, faiss.METRIC_INNER_PRODUCT)
    else:
        index_pq = faiss.IndexPQFastScan(args.dim, args.dim // args.d_sub, 4, faiss.METRIC_INNER_PRODUCT)
        index_pq.pq.cp.niter = args.niter
    index_pq.train(x_train)
    faiss.write_index(index_pq, f'{args.save_path}/{i}.index')