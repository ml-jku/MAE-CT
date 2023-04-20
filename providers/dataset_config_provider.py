from pathlib import Path


class DatasetConfigProvider:
    def __init__(
            self,
            global_dataset_paths,
            local_dataset_path=None,
            data_source_modes=None,
            data_caching_modes=None,
    ):
        self.global_dataset_paths = global_dataset_paths
        self.local_dataset_path = local_dataset_path
        self.data_source_modes = data_source_modes
        self.data_caching_modes = data_caching_modes

    def get_global_dataset_path(self, dataset_identifier):
        path = self.global_dataset_paths[dataset_identifier]
        path = Path(path).expanduser()
        # enforce path exists (e.g. mnist/cifar are downloaded automatically)
        assert path.exists(), f"path to '{dataset_identifier}' doesn't exist ({path})"
        return path

    def get_local_dataset_path(self):
        if self.local_dataset_path is None:
            return None
        path = Path(self.local_dataset_path).expanduser()
        assert path.exists(), f"local_dataset_path '{path}' doesn't exist"
        return path

    def get_data_source_mode(self, dataset_identifier):
        if self.data_source_modes is None or dataset_identifier not in self.data_source_modes:
            return None
        data_source_mode = self.data_source_modes[dataset_identifier]
        assert data_source_mode in ["global", "local"], \
            f'data_source_mode {data_source_mode} not in ["global", "local"]'
        return data_source_mode

    def get_data_caching_mode(self, dataset_identifier):
        if self.data_caching_modes is None or dataset_identifier not in self.data_caching_modes:
            return None
        data_caching_mode = self.data_caching_modes[dataset_identifier]
        assert data_caching_mode in [None, "shared_dict"], \
            f'data_caching_mode {data_caching_mode} not in [None, "shared_dict"]'
        return data_caching_mode
