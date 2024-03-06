import datetime
import logging
import os
import sys
from argparse import ArgumentParser
from os.path import isfile

import faiss
import numpy as np
from tqdm import tqdm


def setup_logging():
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    log_file_name = f"learn_codebooks-{timestamp}.log"
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
        "--paths",
        type=str,
        nargs="+",
        required=True,
        help="Paths to the saved attention keys.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
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
        required=True,
        help="Range of attention layers and heads.",
    )
    parser.add_argument(
        "--niter",
        type=int,
        default=32,
        help="Number of iterations for k-means.",
    )
    parser.add_argument(
        "--first_n",
        type=int,
        default=None,
        help="Number of vectors to reconstruct.",
    )
    parser.add_argument(
        "--factory",
        type=str,
        default=None,
        help="Factory string for FAISS index creation.",
    )
    # Dummy default value for dim to initiate the process
    parser.add_argument(
        "--dim", type=int, default=128, help="Dummy dimension of the vectors."
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    setup_logging()
    create_directories(args.save_path, args.overwrite)

    pbar = tqdm(range(args.range[0], args.range[1]))

    # Flag to track if dim printed
    vector_dimension_printed = False
    for i in pbar:
        vecs = []

        # init vector_dimension to None
        vector_dimension = None
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
                    # Dynamically get vector dimension
                    if vector_dimension is None:
                        vector_dimension = vec.shape[1]
                        # Check if the dimension has been printed
                        if not vector_dimension_printed:
                            print(
                                f"Determined vector dimension: {vector_dimension}"
                            )
                            # Set the flag to True after printing
                            vector_dimension_printed = True
                except RuntimeError as e:
                    logging.error(
                        f"Error reading {index_file_name}: {e}. Skipping."
                    )

        if vecs:
            x_train = np.concatenate(vecs, axis=0)
            if args.factory:
                index_pq = faiss.index_factory(
                    vector_dimension, args.factory, faiss.METRIC_INNER_PRODUCT
                )
            else:
                index_pq = faiss.IndexPQFastScan(
                    vector_dimension,
                    vector_dimension // args.d_sub,
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
