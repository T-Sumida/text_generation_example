import os
import re
import pickle
from typing import List

import luigi
import mojimoji

from trainer.shared_config import SharedConfig
from trainer.scraping import Scraping


class Preprocess(luigi.Task):
    def requires(self):
        return {
            "download_files": Scraping()
        }

    def output(self):
        return luigi.LocalTarget(
            os.path.join(SharedConfig().tmp_dir_path, "preprocessed_file"),
            format=luigi.format.Nop
        )

    def run(self):
        with self.input()["download_files"].open("r") as f:
            download_files: List = pickle.load(f)

        output_path = os.path.join(
            os.path.join(SharedConfig().tmp_dir_path, "preprocessed_text.txt")
        )
        
        with open(output_path, "w", encoding="utf-8") as fw:
            for file in download_files:
                with open(file, "r", encoding="utf-8") as f:
                    data = f.read()
                    preprocessed_data = self._preprocess(data)
                    fw.write(preprocessed_data)
        
        with self.output().open("w") as f:
            f.write(
                pickle.dumps(output_path, protocol=pickle.HIGHEST_PROTOCOL)
            )

    def _preprocess(self, text_data: str) -> str:
        text_data = text_data.replace("\n", "").replace("\r", "").replace("\t", "")
        text_data = re.sub(r"[!”#$%&\'\\\\()*+,-./:;?@[\\]^_`{|}~「」〔〕“”〈〉『』【】＆＊・（）＄＃＠。,？！｀＊＋￥％]", '', text_data)
        text_data = mojimoji.zen_to_han(text_data)
        # text_data = re.sub(r"[\uFF01-\uFF0F\uFF1A-\uFF20\uFF3B-\uFF40\uFF5B-\uFF65\u3000-\u303F]", '', text_data)
        return text_data.lower().strip()
