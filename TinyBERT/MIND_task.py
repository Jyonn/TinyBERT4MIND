import argparse
import os

import torch
from torch.utils.data import DataLoader, SequentialSampler

from task_distill import MindProcessor, MrpcProcessor, convert_examples_to_features, \
    get_tensor_data, do_eval, logger
from transformer import TinyBertForSequenceClassification, BertTokenizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. "
                             "Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--tiny_bert_model",
                        default=None,
                        type=str,
                        required=True,
                        help="The tiny bert model dir.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization."
                             "Sequences longer than this will be truncated, and sequences shorter"
                             "than this will be padded.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")

    task_name = 'mind'
    args = parser.parse_args()
    processor = MindProcessor()
    # processor = MrpcProcessor()
    label_list = processor.get_labels()
    num_labels = len(label_list)
    output_mode = "classification"

    tokenizer = BertTokenizer.from_pretrained(args.tiny_bert_model, do_lower_case=args.do_lower_case)

    eval_examples = processor.get_dev_examples(args.data_dir)
    eval_features = convert_examples_to_features(eval_examples, label_list, args.max_seq_length,
                                                 tokenizer, output_mode)
    eval_data, eval_labels = get_tensor_data(output_mode, eval_features)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    tiny_bert_model = TinyBertForSequenceClassification.from_pretrained(
        args.tiny_bert_model, num_labels=num_labels)
    tiny_bert_model.to(device)

    tiny_bert_model.eval()
    result = do_eval(tiny_bert_model, task_name, eval_dataloader,
                     device, output_mode, eval_labels, num_labels)

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))
