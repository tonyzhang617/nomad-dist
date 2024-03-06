import logging
import os
import sys
from argparse import ArgumentParser
from os.path import isfile

import faiss
import numpy as np
from tqdm import tqdm


def setup_logging(model_name):
    log_file_name = f"{model_name}-learn_codebooks.log"
    logging.basicConfig(
        filename=log_file_name,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def create_directories(save_path, overwrite):
    if os.path.exists(save_path) and not overwrite:
        logging.info(
            f"{save_path} already exists. Use --overwrite to replace it."
        )
        sys.exit(1)
    os.makedirs(save_path, exist_ok=True)


def parse_arguments():
    parser = ArgumentParser("Index Trainer")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="The name of the model for which to generate codebooks",
    )
    parser.add_argument(
        "--paths",
        type=str,
        nargs="+",
        help="Paths to the saved attention keys.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        help="Directory to save the learned codebooks.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Flag to overwrite the existing save path.",
    )
    parser.add_argument(
        "--max_runs",
        type=int,
        default=1000,
        help="Maximum runs for loading vectors.",
    )
    parser.add_argument(
        "--d_sub", type=int, default=1, help="Dimension in each sub-quantizer."
    )
    parser.add_argument(
        "--range",
        type=int,
        nargs=2,
        help="Range of attention layers and heads.",
    )
    parser.add_argument(
        "--niter",
        type=int,
        default=32,
        help="Number of iterations for k-means.",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=128,
        help="Dimensionality of the attention key embeddings.",
    )
    parser.add_argument(
        "--first_n", type=int, help="Number of vectors to reconstruct."
    )
    parser.add_argument(
        "--factory", type=str, help="Factory string for faiss index creation."
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    setup_logging(args.model_name)
    create_directories(args.save_path, args.overwrite)

    pbar = tqdm(range(args.range[0], args.range[1]))
    for i in pbar:
        vecs = []
        for path in args.paths:
            for run in range(args.max_runs):
                index_file_name = f"{path}/r{run}_i{i}.index"
                if not isfile(index_file_name):
                    continue
                try:
                    index_flat = faiss.read_index(index_file_name)
                    if (
                        args.first_n is not None
                        and args.first_n < index_flat.ntotal
                    ):
                        vec = index_flat.reconstruct_n(0, args.first_n)
                    else:
                        vec = index_flat.reconstruct_n(0, index_flat.ntotal)
                    vecs.append(vec)
                except RuntimeError as e:
                    logging.error(
                        f"Error reading {index_file_name}: {e}. Skipping."
                    )

        if vecs:
            x_train = np.concatenate(vecs, axis=0)
            if args.factory:
                index_pq = faiss.index_factory(
                    args.dim, args.factory, faiss.METRIC_INNER_PRODUCT
                )
            else:
                index_pq = faiss.IndexPQFastScan(
                    args.dim,
                    args.dim // args.d_sub,
                    4,
                    faiss.METRIC_INNER_PRODUCT,
                )
            index_pq.pq.cp.niter = args.niter
            index_pq.train(x_train)
            faiss.write_index(index_pq, f"{args.save_path}/{i}.index")
        else:
            logging.error(f"No vectors found for index {i}, skipping.")


if __name__ == "__main__":
    main()
