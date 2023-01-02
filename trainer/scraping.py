import os
import re
import glob
import urllib
import pickle
from typing import List

import luigi
import requests
from bs4 import BeautifulSoup

from trainer.shared_config import SharedConfig


API_URL = "https://api.syosetu.com/novelapi/api/"
TEXT_URL = "https://ncode.syosetu.com/{}/{:d}/"


class Scraping(luigi.Task):
    limit_num: int = luigi.IntParameter()
    genre = luigi.Parameter()
    max_part_num: int = luigi.IntParameter()

    def requires(self):
        pass
    
    def output(self):
        return luigi.LocalTarget(
            os.path.join(SharedConfig().tmp_dir_path, 'file_list.pkl'),
            format=luigi.format.Nop
        )
    
    def run(self):
        output_dir = os.path.join(
            os.path.join(SharedConfig().tmp_dir_path, "text_data")
        )
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        ncode_list = self._get_ncode()

        for ncode in ncode_list:
            for part_num in range(1, self.max_part_num+1):
                url = TEXT_URL.format(ncode, part_num)
                self._scraping(url, output_dir)

        files = glob.glob(
            os.path.join(output_dir, "*.txt")
        )
        with self.output().open('w') as f:
            f.write(pickle.dumps(files, protocol=pickle.HIGHEST_PROTOCOL))

    def _get_ncode(self) -> None:
        ncode_list = []
        pattern = r"ncode:(.+)\n"
        if self.genre:
            payload = {'out': 'yaml', "lim": self.limit_num, "genre": self.genre}
        else:
            payload = {'out': 'yaml', "lim": self.limit_num}

        text = requests.get(API_URL, params=payload).text
        r = re.findall(pattern, text) 
        for n_code in r:
            ncode_list.append(n_code.strip())
        return ncode_list

    def _scraping(self, url: str, output_dir: str) -> List:
        try:
            res = urllib.request.urlopen(url)
            soup = BeautifulSoup(res, "html.parser")
            title = soup.select_one("title").text
            text = soup.select_one("#novel_honbun").text + "\n"

            file_path = os.path.join(output_dir, title + ".txt")
            mode = "w"
            if os.path.exists(file_path):
                mode = "a"
            with open(os.path.join(output_dir, title + ".txt"), mode, encoding="utf-8") as f:
                f.write(text)
        except Exception as e:
            print(url, e)