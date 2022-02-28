import args
from dataloader.matcher import load_data
from dataprocessor.matcher import get_processor
from main import main

if __name__ == "__main__":
    args = args.get_args()
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
