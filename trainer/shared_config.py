import luigi


class SharedConfig(luigi.Config):
    tmp_dir_path = luigi.Parameter()
