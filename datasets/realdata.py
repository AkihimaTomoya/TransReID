import os
import os.path as osp
import glob
from collections import defaultdict
from .bases import BaseImageDataset

class RealData(BaseImageDataset):
    """
    RealData/
    └── dataID/
        ├── goc_1/<pid>/{v1,v2}/
        ├── goc_2/<pid>/{v1,v2}/
        ├── goc_3/<pid>/{v1,v2}/
        └── goc_4/<pid>/{v1,v2}/

    ENV:
      - REALDATA_QUERY_VER ∈ {'v1','v2'} (default 'v1')
      - REALDATA_QUERY_GOC = 'goc_x' (optional). Nếu set, query & gallery chỉ lấy trong gốc này.

    Labels:
      - viewid: v1 -> 0 ; v2 -> 1
      - camid_goc: index theo thứ tự sort goc_* (goc_1->0, goc_2->1, ...)
      - camid_final = camid_goc * 10 + viewid   # đảm bảo v1 & v2 khác camera
      - pid_norm = camid_goc * 1_000_000 + pid_raw (namespace theo gốc)

    Chỉ đưa PID vào split nếu PID đó có đủ 2 phiên bản (query_ver & gallery_ver) trong phạm vi gốc xét.
    """
    dataset_dir = 'RealData'

    def __init__(self, root: str = '', verbose: bool = True, **kwargs):
        super().__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.dataid_dir = osp.join(self.dataset_dir, 'dataID')
        self._check_before_run()

        # 1) ENV
        query_ver = os.environ.get('REALDATA_QUERY_VER', 'v1').strip().lower()
        assert query_ver in {'v1', 'v2'}, "REALDATA_QUERY_VER must be 'v1' or 'v2'"
        gallery_ver = 'v1' if query_ver == 'v2' else 'v2'

        query_goc_only = os.environ.get('REALDATA_QUERY_GOC', '').strip() or None

        # 2) list goc_*
        all_goc_dirs = []
        for name in sorted(os.listdir(self.dataid_dir)):
            p = osp.join(self.dataid_dir, name)
            if osp.isdir(p) and name.startswith('goc_'):
                all_goc_dirs.append(name)
        if not all_goc_dirs:
            raise RuntimeError("No 'goc_*' folders found under RealData/dataID")

        if query_goc_only and query_goc_only not in all_goc_dirs:
            if verbose:
                print(f"[RealData] WARNING: REALDATA_QUERY_GOC='{query_goc_only}' not found. "
                      f"Available: {all_goc_dirs}. Using ALL.")
            query_goc_only = None

        # 3) maps
        goc_to_cam = {name: idx for idx, name in enumerate(all_goc_dirs)}
        ver_to_view = {'v1': 0, 'v2': 1}
        IMG_EXTS = ('*.jpg', '*.jpeg', '*.png', '*.bmp')

        # 4) collect paths: data[goc_name][pid_raw]['v1'/'v2'] = [paths...]
        data = defaultdict(lambda: defaultdict(lambda: {'v1': [], 'v2': []}))
        goc_scope = [query_goc_only] if query_goc_only else list(all_goc_dirs)

        for goc_name in goc_scope:
            goc_path = osp.join(self.dataid_dir, goc_name)
            for pid_name in sorted(os.listdir(goc_path)):
                id_dir = osp.join(goc_path, pid_name)
                if not osp.isdir(id_dir):
                    continue
                try:
                    pid_raw = int(pid_name)
                except Exception:
                    continue

                for ver in ('v1', 'v2'):
                    ver_dir = osp.join(id_dir, ver)
                    if not osp.isdir(ver_dir):
                        continue
                    paths = []
                    for ext in IMG_EXTS:
                        paths.extend(glob.glob(osp.join(ver_dir, ext)))
                    if paths:
                        data[goc_name][pid_raw][ver].extend(sorted(paths))

        # 5) build splits with overlap enforced & per-(goc,ver) camera ids
        train, query, gallery = [], [], []

        for goc_name in goc_scope:
            camid_goc = goc_to_cam[goc_name]
            pid_dict = data[goc_name]

            for pid_raw, ver_dict in pid_dict.items():
                q_list = ver_dict.get(query_ver, [])
                g_list = ver_dict.get(gallery_ver, [])
                if not q_list or not g_list:
                    continue  # require both versions to ensure overlap

                pid_norm = camid_goc * 1_000_000 + pid_raw
                q_view = ver_to_view[query_ver]
                g_view = ver_to_view[gallery_ver]

                # camid_final: make v1 & v2 different cameras within the same goc
                camid_q = camid_goc * 10 + q_view
                camid_g = camid_goc * 10 + g_view

                for p in q_list:
                    query.append((p, pid_norm, camid_q, q_view))
                for p in g_list:
                    gallery.append((p, pid_norm, camid_g, g_view))

        # 6) sanity checks
        if not query or not gallery:
            print("[RealData] WARNING: Empty query or gallery after filtering. "
                  "Check folder structure or change REALDATA_QUERY_VER / REALDATA_QUERY_GOC.")
        else:
            q_ids = {pid for _, pid, _, _ in query}
            g_ids = {pid for _, pid, _, _ in gallery}
            overlap = len(q_ids & g_ids)
            if overlap == 0:
                print("[RealData] WARNING: No PID overlap between query and gallery in current split.")
                print("           Try switching REALDATA_QUERY_VER or REALDATA_QUERY_GOC.")

        # 7) assign & stats
        if verbose:
            print("=> RealData loaded (eval-only)")
            print(f"Query ver = {query_ver} | Gallery ver = {gallery_ver} | Query goc = {query_goc_only or 'ALL'}")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        # 8) info for upstream
        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = \
            self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = \
            self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = \
            self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        if not osp.exists(self.dataset_dir):
            raise RuntimeError(f"'{self.dataset_dir}' is not available")
        if not osp.exists(self.dataid_dir):
            raise RuntimeError(f"'{self.dataid_dir}' is not available")
