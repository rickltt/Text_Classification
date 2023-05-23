import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

from model import Bert, BertCNN, BertRNN, BertRCNN, BertDPCNN

from arguments import ModelArguments, DataTrainingArguments
from datasets import load_dataset
import datasets
import evaluate
import numpy as np
import os
import sys
import logging
from sklearn.metrics import accuracy_score, precision_score,recall_score, f1_score, classification_report


logger = logging.getLogger(__name__)

MODEL_LIST = {
    'bert': Bert,
    'cnn': BertCNN,
    'rnn': BertRNN,
    'rcnn': BertRCNN,
    'dpcnn': BertDPCNN
}

def main():

    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    # if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    #     # If we pass only one argument to the script and it's the path to a json file,
    #     # let's parse it to get our arguments.
    #     model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    # else:
    #     model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    classes = []
    with open(os.path.join(data_args.dataset_dir,"class.txt"),'r') as f:
        lines = f.readlines()
        for line in lines:
            classes.append(line.strip())
    print(classes)
    num_labels = len(classes)
    id2label = { i:j for i,j in enumerate(classes)}
    label2id = { j:i for i,j in enumerate(classes)}

    config = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir, num_labels=num_labels)
    # model = AutoModelForSequenceClassification.from_pretrained(model_args.model_name_or_path, num_labels= num_labels)
    config.label2id = label2id
    config.id2label = id2label
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    model = MODEL_LIST[model_args.model_name].from_pretrained(model_args.model_name_or_path, config=config, cache_dir=model_args.cache_dir)
    
    # Load dataset
    data_files = {}
    data_files["train"] = os.path.join(data_args.dataset_dir,"train.txt")
    data_files["validation"] = os.path.join(data_args.dataset_dir,"dev.txt")
    data_files["test"] = os.path.join(data_args.dataset_dir,"test.txt")
    extension = "text"

    text_dataset = load_dataset(
        extension,
        data_files=data_files,
        cache_dir=model_args.cache_dir
    )

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    def preprocess_function(examples):
        model_inputs = {
            "input_ids": [],
            "token_type_ids": [],
            "attention_mask": [],
            "labels": [],
        }
        for example in examples['text']:
            tokenize_inputs = tokenizer(example.split('\t')[0], padding=padding, max_length=data_args.max_seq_length, truncation=True) 
            labels = int(example.split('\t')[1])
            model_inputs["input_ids"].append(tokenize_inputs['input_ids'])
            model_inputs["token_type_ids"].append(tokenize_inputs['token_type_ids'])
            model_inputs["attention_mask"].append(tokenize_inputs['attention_mask'])
            model_inputs["labels"].append(labels)

        return model_inputs

    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = text_dataset.map(preprocess_function, batched=True, remove_columns='text', load_from_cache_file= not data_args.overwrite_cache, desc="Running tokenizer")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, max_length=data_args.max_seq_length)

    set_seed(training_args.seed)

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    if training_args.do_predict:
        if "test" not in raw_datasets and "test_matched" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))


    # metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        pred = np.argmax(logits, axis=-1)
        # return metric.compute(predictions=predictions, references=labels)
        accuracy = accuracy_score(y_true=labels, y_pred=pred)
        recall = recall_score(y_true=labels, y_pred=pred,average="micro")
        precision = precision_score(y_true=labels, y_pred=pred,average="micro")
        f1 = f1_score(y_true=labels, y_pred=pred,average="micro")
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        max_eval_samples = (
            data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        # Removing the `label` columns because it contains -1 and Trainer won't like that.
        # predict_dataset = predict_dataset.remove_columns("labels")
        predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
        predictions = np.argmax(predictions, axis=1)
        labels = predict_dataset["labels"]
        report = classification_report(labels,predictions,target_names=classes)
        output_predict_file = os.path.join(training_args.output_dir, f"predict_results.txt")
        if trainer.is_world_process_zero():
            with open(output_predict_file, "w") as writer:
                logger.info(f"***** Predict results  *****")
                writer.write(report)
                # writer.write("index\tprediction\n")
                # for index, item in enumerate(predictions):
                #     item = classes[item]
                #     writer.write(f"{index}\t{item}\n")

if __name__ == '__main__':
    main()