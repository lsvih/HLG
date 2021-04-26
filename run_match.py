import args
from dataloader.matcher import load_data
from dataprocessor.matcher import get_processor
from main import main

if __name__ == "__main__":
    args = args.get_args()
    base_name = args.exp_name + args.model_type

    args.output_dir = 'output/' + base_name + '/'
    args.cache_dir = 'cache/' + base_name + '/'

    if args.model_type == 'BERT':
        args.bert_vocab_file = "bert-base-chinese"
        args.bert_model_dir = "/home/liyanzeng/embeddings/bert-base-chinese"
    elif args.model_type == 'ERNIE':
        args.bert_vocab_file = "/home/liyanzeng/embeddings/ERNIE_1.0_max-len-512-pytorch"
        args.bert_model_dir = "/home/liyanzeng/embeddings/ERNIE_1.0_max-len-512-pytorch"
    elif args.model_type == 'BERTwwm':
        args.bert_vocab_file = "/home/liyanzeng/embeddings/bert-wwm-chinese"
        args.bert_model_dir = "/home/liyanzeng/embeddings/bert-wwm-chinese"

    args.data_dir = 'dataset/MultiSeg' + base_name + '/'

    model_id = '1'

    processor = None
    if args.exp_name == 'LCQMC':
        processor = get_processor(['0', '1'])
    elif args.exp_name == 'BQ':
        processor = get_processor(['0', '1'])
    elif args.exp_name == 'XNLI':
        processor = get_processor(['contradiction', 'entailment', 'neutral'])
    else:
        print('incorrect exp_name')
        exit(0)

    main(args, model_id, processor, load_data)
