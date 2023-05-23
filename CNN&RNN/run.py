import torch

import numpy as np
import random
import os
import argparse
from data_utils import TC_Dataset, DataProcessor, collate_fn
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from model import TextCNN, TextRNN, TextRCNN, DPCNN
from sklearn.metrics import classification_report, accuracy_score
import logging
from tqdm import tqdm, trange

logger = logging.getLogger(__name__)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def train(args, model, train_dataset, dev_dataset):
    """ Train the model """

    args.train_batch_size = args.per_gpu_train_batch_size 
    train_sampler = RandomSampler(train_dataset) 
    train_dataloader = DataLoader(train_dataset,
                                  sampler=train_sampler,
                                  batch_size=args.train_batch_size,
                                  collate_fn=collate_fn)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    args.logging_steps = eval(args.logging_steps)
    if isinstance(args.logging_steps, float):
        args.logging_steps = int(args.logging_steps * len(train_dataloader)) 

    # Train
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Random seed = %d", args.seed)

    global_step = 0
    best_acc = 0.0
    tr_loss = 0.0

    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    set_seed(args)  # Added here for reproductibility
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for batch in epoch_iterator:
            model.train()

            batch = tuple(t.to(args.device) for t in batch.values())
            inputs = {
                "input_ids":batch[0],
                "label":batch[1],
                "mode": "train"
            }

            loss = model(**inputs)
            loss.backward()
            tr_loss += loss.item()
            epoch_iterator.set_description('Loss: {}'.format(round(loss.item(), 6)))
            optimizer.step()
            model.zero_grad()
            global_step += 1
            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                result = evaluate(args, model, dev_dataset)
                if best_acc < result['acc']:
                    best_acc = result['acc']
                    output_dir = os.path.join(args.output_dir, "best_checkpoint")
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    torch.save(model.state_dict(), os.path.join(output_dir, "model"))
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

    return global_step, tr_loss / global_step

def evaluate(args, model, eval_dataset):

    args.eval_batch_size = args.per_gpu_eval_batch_size 
    eval_sampler = SequentialSampler(eval_dataset) 
    eval_dataloader = DataLoader(eval_dataset,
                                 sampler=eval_sampler,
                                 batch_size=args.eval_batch_size,
                                 collate_fn=collate_fn)

    # Eval!
    logger.info("***** Running evaluation  *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = []
    trues = []
    model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(args.device) for t in batch.values())
        with torch.no_grad():
            inputs = {
                "input_ids":batch[0],
                "label":batch[1],
                "mode": "test"
            }
            tmp_eval_loss, logits = model(**inputs)
            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        pred = logits.argmax(-1)
        preds.extend(pred.detach().cpu().tolist())
        trues.extend(inputs['label'].detach().cpu().tolist())

    eval_loss = eval_loss / nb_eval_steps
    print('eval_loss={}'.format(eval_loss))

    result = {}
    eval_loss = eval_loss / nb_eval_steps
    report = classification_report(trues, preds, target_names=args.labels)
    acc = accuracy_score(trues, preds)
    print('acc:', acc)
    result['eval_loss'] = eval_loss
    result['acc'] = acc
    result['report'] = report
    return result

def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument('--dataset', default='THUCNews', type=str, help='THUCNews, toutiao')
    parser.add_argument('--model_type', default='DPCNN', type=str, help='TextCNN, TextRNN, TextRCNN, DPCNN')
    parser.add_argument("--embedding_file", default='../pretrained/token_vec_300.bin', type=str)
    parser.add_argument("--output_dir", default='output', type=str, 
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--per_gpu_train_batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=3e-4, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--dropout_prob", default=0.1, type=float,
                        help="dropout_prob.")
    parser.add_argument("--num_train_epochs", default=10.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--logging_steps", type=str, default='1.0',
                        help="Log every X updates steps.")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--emb_size", type=int, default=300, help="hidden size for word2vec")
   
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    datafiles = {
        'THUCNews': '../data/THUCNews',
        'toutiao': '../data/toutiao_news'
    }
    model_list = {
        'TextCNN':TextCNN,
        'TextRNN':TextRNN,
        'TextRCNN':TextRCNN,
        'DPCNN': DPCNN
    }

    args.data_dir = datafiles[args.dataset]
    # Set seed
    set_seed(args)
    
    processor = DataProcessor(args)
    args.labels = processor.labels
    args.label2id = processor.label2id
    args.id2label = processor.id2label
    args.num_labels = processor.num_labels
    args.embedding_dict, args.word2id, args.vec_mat = processor.embedding_dict, processor.word2id, processor.vec_mat
    
    train_dataset = TC_Dataset(args, 'train')
    dev_dataset = TC_Dataset(args, 'dev')
    test_dataset = TC_Dataset(args, 'test')

    model = model_list[args.model_type](args)
    model.to(args.device)

    # Training
    if args.do_train:
        global_step, tr_loss = train(args, model, train_dataset, dev_dataset)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
        # Create output directory if needed
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        output_dir = os.path.join(args.output_dir, "last_checkpoint")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        torch.save(model.state_dict(), os.path.join(output_dir, "model"))
        torch.save(args, os.path.join(output_dir, "training_args.bin"))
        logger.info("Saving model checkpoint to %s", output_dir)

    # Evaluation
    if args.do_eval:
        checkpoint = os.path.join(args.output_dir, 'best_checkpoint')
        state_dict = torch.load(os.path.join(checkpoint, "model"))
        model.load_state_dict(state_dict)
        model.to(args.device)
        result = evaluate(args, model, test_dataset)
        output_eval_file = os.path.join(args.output_dir, "test_results.txt")
        with open(output_eval_file, "a") as writer:
            writer.write('***** Model: {} Predict in {} test dataset *****'.format(args.model_type, args.dataset))
            writer.write("{} \n".format(result['report']))

if __name__ == '__main__':
    main()