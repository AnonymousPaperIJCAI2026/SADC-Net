"""dataset"""
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import glob


class DAGMDataset(Dataset):
    def __init__(self, root, train=True, category=None, transform=None, gt_target_transform=None):
        super(DAGMDataset, self).__init__()
        self.categories = [
            'Class1', 'Class2', 'Class3', 'Class4', 'Class5',
            'Class6', 'Class7', 'Class8', 'Class9', 'Class10'
        ]
        self.train = train
        self.category = category
        self.root = os.path.join(root, 'DAGM_KaggleUpload')
        self.transform = transform
        self.gt_target_transform = gt_target_transform

        # Build global index first, then initialize the current "view" (cur_*) so
        # that the dataset is immediately indexable.
        # Without this, __getitem__/__len__ will fail because cur_* are undefined.
        self.preprocess()
        self.update(self.category)

        self.dataset_name = 'DAGM'

    def preprocess(self):
        self.img_paths = {
            'Train': {category: [] for category in self.categories},
            'Test': {category: [] for category in self.categories},
        }
        self.gt_paths = {
            'Train': {category: [] for category in self.categories},
            'Test': {category: [] for category in self.categories},
        }
        self.labels = {
            'Train': {category: [] for category in self.categories},
            'Test': {category: [] for category in self.categories},
        }

        for phase in ['Train', 'Test']:
            for category in self.categories:
                img_dir = os.path.join(self.root, category)
                img_paths = glob.glob(os.path.join(img_dir, phase) + "/*.PNG")
                img_paths.sort()
                gt_paths = []
                labels = []
                for img_path in img_paths:
                    idx_path = img_path.split('/')[-1]
                    idx = idx_path.split('.')[-2]
                    gt_path = os.path.join(img_dir, phase, "Label", idx + "_label.PNG")
                    label = 1
                    if not os.path.exists(gt_path):
                        gt_path = None
                        label = 0
                    gt_paths.append(gt_path)
                    labels.append(label)
                self.img_paths[phase][category] = img_paths
                self.gt_paths[phase][category] = gt_paths
                self.labels[phase][category] = labels

    def update(self, category=None):
        """Select a subset view (single class) or an all-class view (category=None)."""
        self.category = category
        self.cur_img_paths, self.cur_gt_paths = [], []
        self.cur_img_labels, self.cur_img_categories = [], []

        phase = 'Train' if self.train else 'Test'

        if self.category is not None:
            self.cur_img_paths = self.img_paths[phase][self.category]
            self.cur_gt_paths = self.gt_paths[phase][self.category]
            self.cur_img_labels = self.labels[phase][self.category]
            self.cur_img_categories = [self.category] * len(self.cur_img_paths)
        else:
            for cat in self.categories:
                self.cur_img_paths.extend(self.img_paths[phase][cat])
                self.cur_gt_paths.extend(self.gt_paths[phase][cat])
                self.cur_img_labels.extend(self.labels[phase][cat])
                self.cur_img_categories.extend([cat] * len(self.img_paths[phase][cat]))

        assert len(self.cur_img_paths) == len(self.cur_gt_paths)
        assert len(self.cur_img_paths) == len(self.cur_img_labels)
        assert len(self.cur_img_paths) == len(self.cur_img_categories)

    def __len__(self):
        # Be robust to external code that constructs the dataset but does not call update().
        if not hasattr(self, 'cur_img_paths'):
            self.update(self.category)
        return len(self.cur_img_paths)

    def __getitem__(self, idx):
        if not hasattr(self, 'cur_img_categories'):
            self.update(self.category)

        category = self.cur_img_categories[idx]
        img_path = self.cur_img_paths[idx]
        label = self.cur_img_labels[idx]

        img = Image.open(img_path).convert('RGB')
        if self.cur_gt_paths[idx] is not None:
            gt = np.array(Image.open(self.cur_gt_paths[idx]))
        else:
            gt = np.zeros((img.size[1], img.size[0]), dtype=np.uint8)
        gt = Image.fromarray(gt)

        if self.transform is not None:
            img = self.transform(img)
        if self.gt_target_transform is not None:
            gt = self.gt_target_transform(gt)

        return img, label, gt, category, img_path
