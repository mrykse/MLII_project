import os
from PIL import Image
from torch.utils.data import Dataset


class ReadFromDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_list = self._get_file_list()

    def _get_file_list(self):
        file_list = []
        for subdir in os.listdir(self.root_dir):
            subdir_path = os.path.join(self.root_dir, subdir)
            if os.path.isdir(subdir_path):
                file_list.extend([(os.path.join(subdir, file), subdir) for file in os.listdir(subdir_path)])
        return file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path, label = self.file_list[idx]
        img = Image.open(os.path.join(self.root_dir, img_path)).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return {'image': img, 'label': 1 if label == 'modified' else 0}
