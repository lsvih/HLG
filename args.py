import argparse


def get_args():
    parser = argparse.ArgumentParser(description='BERT Baseline')

    parser.add_argument("--exp_name",
                        default="ChnSenti",
                        type=str,
                        help="The name of benchmark dataset")
    parser.add_argument("--model_type",
                        default="BERT",  # BERT | BERTwwm | ERNIE
                        type=str,
                        help="The name of bert model")

    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="The random seed for initialization")

    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")

    parser.add_argument("--train_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for training.")

    parser.add_argument("--dev_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for dev.")
    parser.add_argument("--test_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for test.")

    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")

    parser.add_argument("--num_train_epochs",
                        default=4,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")

    parser.add_argument("--bert_learning_rate",
                        default=3e-5,
                        type=float,
                        help="Fine-tuning learning rate of BERT parameters")
    parser.add_argument("--model_learning_rate",
                        default=1e-5,
                        type=float,
                        help="Learning rate of parameters of graph model")

    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    parser.add_argument("--early_stop", type=int, default=50,
                        help="early stop training")

    parser.add_argument("--note", type=str, default="", help="Comments of the experiment.")

    parser.add_argument("--gpu_ids", type=str, default="0", help="The id of GPU device, e.g: 0,1,2")

    parser.add_argument("--num_workers", type=int, default=0, help="Number of dataloader workers")

    config = parser.parse_args()

    return config
