import glob
import re
import os.path as osp

from .bases import BaseImageDataset


class RealData(BaseImageDataset):
    """
    RealData Dataset
    Custom vehicle re-identification dataset with 4 camera views
    
    Dataset statistics will be printed when loading
    Structure:
    - image_train/: Training images
    - image_query/: Query images  
    - image_test/: Gallery images
    - keypoint_train.txt: Viewpoint annotations for training
    - keypoint_test.txt: Viewpoint annotations for testing
    
    Filename format: {vid}_c{camid}_{index}.jpg
    """

    dataset_dir = 'RealData_organized'

    def __init__(self, root='', verbose=True, **kwargs):
        super(RealData, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'image_train')
        self.query_dir = osp.join(self.dataset_dir, 'image_query')
        self.gallery_dir = osp.join(self.dataset_dir, 'image_test')

        self._check_before_run()

        # Load training viewpoint mapping
        path_train = osp.join(self.dataset_dir, 'keypoint_train.txt')
        self.image_map_view_train = {}
        if osp.exists(path_train):
            with open(path_train, 'r') as txt:
                lines = txt.readlines()
            for img_idx, img_info in enumerate(lines):
                content = img_info.strip().split(' ')
                if len(content) >= 2:
                    viewid = int(content[-1])
                    self.image_map_view_train[osp.basename(content[0])] = viewid
        else:
            print(f"Warning: {path_train} not found. Creating dummy viewpoint mapping.")
            # Create dummy mapping based on camera IDs
            self.image_map_view_train = {}

        # Load test viewpoint mapping
        path_test = osp.join(self.dataset_dir, 'keypoint_test.txt')
        self.image_map_view_test = {}
        if osp.exists(path_test):
            with open(path_test, 'r') as txt:
                lines = txt.readlines()
            for img_idx, img_info in enumerate(lines):
                content = img_info.strip().split(' ')
                if len(content) >= 2:
                    viewid = int(content[-1])
                    self.image_map_view_test[osp.basename(content[0])] = viewid
        else:
            print(f"Warning: {path_test} not found. Creating dummy viewpoint mapping.")
            self.image_map_view_test = {}

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> RealData loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(
            self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(
            self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(
            self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        """
        Process directory and extract image information
        
        Args:
            dir_path: Path to image directory
            relabel: Whether to relabel VIDs (True for train, False for test)
        
        Returns:
            List of tuples: (img_path, vid, camid, viewid)
        """
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        
        # Pattern to match: {vid}_c{camid}_{index}.jpg
        pattern = re.compile(r'(\d+)_c(\d+)_(\d+)')

        # Collect unique VIDs
        vid_container = set()
        for img_path in img_paths:
            match = pattern.search(osp.basename(img_path))
            if match:
                vid = int(match.group(1))
                vid_container.add(vid)
        
        # Create VID to label mapping for relabeling
        vid2label = {vid: label for label, vid in enumerate(sorted(vid_container))}

        dataset = []
        count_no_viewpoint = 0
        
        # Get max camera ID for viewpoint fallback
        max_camid = 0
        for img_path in img_paths:
            match = pattern.search(osp.basename(img_path))
            if match:
                camid = int(match.group(2))
                max_camid = max(max_camid, camid)
        
        for img_path in img_paths:
            match = pattern.search(osp.basename(img_path))
            if not match:
                print(f"Warning: Cannot parse filename: {osp.basename(img_path)}")
                continue
            
            vid = int(match.group(1))
            camid = int(match.group(2))
            
            # Adjust camera ID to 0-indexed
            camid -= 1
            
            # Relabel VID if needed
            if relabel:
                vid = vid2label[vid]
            
            # Get viewpoint ID
            img_name = osp.basename(img_path)
            if img_name in self.image_map_view_train:
                viewid = self.image_map_view_train[img_name]
            elif img_name in self.image_map_view_test:
                viewid = self.image_map_view_test[img_name]
            else:
                # Fallback: use camera ID as viewpoint
                viewid = camid
                count_no_viewpoint += 1
            
            dataset.append((img_path, vid, camid, viewid))
        
        if count_no_viewpoint > 0:
            print(f"  {count_no_viewpoint} samples without viewpoint annotations (using camid as viewid)")
        
        return dataset
