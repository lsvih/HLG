import args
from dataloader.classifier import load_data
from dataprocessor.classifier import get_processor
from main import main

if __name__ == "__main__":
    args = args.get_args()
    model_id = '1'

    processor = None
    if args.exp_name == 'ChnSenti':
        processor = get_processor(['0', '1'])
    elif args.exp_name == 'weibo':
        processor = get_processor(['0', '1'])
    elif args.exp_name == 'THUCNews':
        processor = get_processor([u'房产', u'科技', u'财经', u'游戏', u'娱乐', u'时尚', u'时政', u'家居', u'教育', u'体育'])
    else:
        print('incorrect exp_name')
        exit(0)

    main(args, model_id, processor, load_data)
