import os
import pickle
import argparse
import configparser

import luigi
from transformers import T5Tokenizer, AutoModelForCausalLM
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

from trainer.preprocess import Preprocess
from trainer.shared_config import SharedConfig

class Train(luigi.Task):
    block_size = luigi.IntParameter()
    train_epochs = luigi.IntParameter()
    per_device_train_batch_size = luigi.IntParameter()
    logging_steps = luigi.IntParameter()
    save_steps = luigi.IntParameter()

    def requires(self):
        return {
            "preprocessed_text_file": Preprocess()
        }

    def output(self):
        return luigi.LocalTarget(
            os.path.join(SharedConfig().tmp_dir_path, 'fin.pkl'),
            format=luigi.format.Nop
        )

    def run(self):
        with self.input()["preprocessed_text_file"].open("r") as f:
            text_path: str = pickle.load(f)

        tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt2-medium")
        tokenizer.do_lower_case = True
        model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt2-medium")

        train_dataset = TextDataset(
            tokenizer=tokenizer,
            file_path=text_path,
            block_size=self.block_size
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )

        args = TrainingArguments(
            output_dir=SharedConfig().tmp_dir_path,  
            overwrite_output_dir=True,
            num_train_epochs=self.train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size, 
            logging_steps=self.logging_steps,  
            save_steps=self.save_steps
        )

        # トレーナーを設定
        trainer = Trainer(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
        )

        # トレーニング
        trainer.train()

        with self.output().open("w") as f:
            f.write(pickle.dumps("", protocol=pickle.HIGHEST_PROTOCOL))


def get_args() -> argparse.Namespace:
    """引数取得
    Returns:
        argparse.Namespace: 引数情報
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--new", action="store_true")
    return parser.parse_args()


def remove_tmp_dir(config_path: str) -> None:
    import shutil
    config = configparser.ConfigParser()
    config.read(config_path)
    tmp_path = config.get("SharedConfig", "tmp_dir_path")
    shutil.rmtree(tmp_path)


def main() -> None:
    args = get_args()
    if args.new:
        remove_tmp_dir(args.config)
    luigi.configuration.LuigiConfigParser.add_config_path(args.config)
    task = Train()
    luigi.build([task], local_scheduler=True)

if __name__ == "__main__":
    main()