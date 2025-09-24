# datasets/realdata.py
import os
import os.path as osp
import glob
from .bases import BaseImageDataset

class RealData(BaseImageDataset):
    """
    Cấu trúc dữ liệu:

    RealData/
    └── dataID/
        ├── goc_1/
        │   ├── <pid>/
        │   │   ├── v1/
        │   │   └── v2/
        ├── goc_2/
        │   ├── <pid>/
        │   │   ├── v1/
        │   │   └── v2/
        ├── goc_3/ ...
        └── goc_4/ ...

    - Chọn phiên bản làm QUERY qua biến môi trường:
        REALDATA_QUERY_VER in {'v1','v2'} (mặc định: 'v1')
      Phiên bản còn lại làm GALLERY.

    - (Tùy chọn) Chỉ lấy query từ một gốc cụ thể:
        REALDATA_QUERY_GOC = 'goc_1' (mặc định: None -> lấy tất cả goc_* cho query)

    - camid: gán theo thứ tự goc_* đã sort (goc_1->0, goc_2->1, goc_3->2, ...)
    - viewid: v1 -> 0 ; v2 -> 1
    - train set để rỗng (eval-only) — phù hợp chạy test/mAP/CMC.
    """
    dataset_dir = 'RealData'

    def __init__(self, root='', verbose=True, **kwargs):
        super().__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.dataid_dir = osp.join(self.dataset_dir, 'dataID')

        self._check_before_run()

        # chọn phiên bản query
        query_ver = os.environ.get('REALDATA_QUERY_VER', 'v1').strip().lower()
        assert query_ver in {'v1', 'v2'}, "REALDATA_QUERY_VER must be 'v1' or 'v2'"
        gallery_ver = 'v1' if query_ver == 'v2' else 'v2'

        # (tuỳ chọn) chỉ lấy query từ 1 gốc
        query_goc_only = os.environ.get('REALDATA_QUERY_GOC', '').strip() or None

        # duyệt toàn bộ thư mục goc_*
        all_goc_dirs = []
        for name in sorted(os.listdir(self.dataid_dir)):
            p = osp.join(self.dataid_dir, name)
            if osp.isdir(p) and name.startswith('goc_'):
                all_goc_dirs.append(name)
        if not all_goc_dirs:
            raise RuntimeError("No 'goc_*' folders found under RealData/dataID")

        # ánh xạ goc_* -> camid (ổn định theo thứ tự sort)
        goc_to_cam = {name: idx for idx, name in enumerate(all_goc_dirs)}
        ver_to_view = {'v1': 0, 'v2': 1}

        train = []    # eval-only
        query = []
        gallery = []

        # duyệt từng gốc
        for goc_name in all_goc_dirs:
            goc_path = osp.join(self.dataid_dir, goc_name)
            camid = goc_to_cam[goc_name]

            # mỗi pid là 1 thư mục con
            for pid_name in sorted(os.listdir(goc_path)):
                id_dir = osp.join(goc_path, pid_name)
                if not osp.isdir(id_dir):
                    continue
                # chỉ nhận pid là số
                try:
                    pid_raw = int(pid_name)
                except:
                    continue

                for ver in ['v1', 'v2']:
                    ver_dir = osp.join(id_dir, ver)
                    if not osp.isdir(ver_dir):
                        continue

                    img_paths = []
                    img_paths += glob.glob(osp.join(ver_dir, '*.jpg'))
                    img_paths += glob.glob(osp.join(ver_dir, '*.jpeg'))
                    img_paths += glob.glob(osp.join(ver_dir, '*.png'))
                    img_paths = sorted(img_paths)

                    viewid = ver_to_view[ver]
                    is_query_ver = (ver == query_ver)
                    is_in_query_goc = (query_goc_only is None or goc_name == query_goc_only)

                    for p in img_paths:
                        tup = (p, pid_raw, camid, viewid)
                        if is_query_ver and is_in_query_goc:
                            query.append(tup)
                        else:
                            # phiên bản còn lại, hoặc khác gốc, cho vào gallery
                            if ver == gallery_ver:
                                gallery.append(tup)

        if verbose:
            print("=> RealData loaded (eval-only)")
            print(f"Query ver = {query_ver} | Gallery ver = {gallery_ver} | Query goc = {query_goc_only or 'ALL'}")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        # thống kê cho upstream
        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        if not osp.exists(self.dataset_dir):
            raise RuntimeError(f"'{self.dataset_dir}' is not available")
        if not osp.exists(self.dataid_dir):
            raise RuntimeError(f"'{self.dataid_dir}' is not available")
