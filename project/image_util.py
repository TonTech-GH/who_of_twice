import glob
import hashlib
import os

import pandas as pd
import tqdm


class ImageInfo:
    def __init__(self, img_path):
        self.img_path = img_path

    @property
    def hash_sha256(self):
        with open(self.img_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()

    @classmethod
    def hash_list(cls, path_list):
        return [ImageInfo(p).hash_sha256 for p in tqdm.tqdm(path_list, desc='calc hash')]


class ImageUtil:
    def __init__(self, root_dir):
        self.root_dir = root_dir

    def images_df(self, with_hash=False):
        images = glob.glob(f'{self.root_dir}/**/*', recursive=True)
        images = [img for img in images if os.path.isfile(img)]
        names = [os.path.basename(os.path.dirname(img)) for img in images]
        fnames = [os.path.basename(img) for img in images]
        df = pd.DataFrame({'name': names, 'image': fnames, 'path': images})

        if with_hash:
            path_list = df['path']
            df['hash'] = ImageInfo.hash_list(path_list)

        return df

    def count_images(self):
        df = self.images_df().drop('path', axis=1)
        return df.groupby('name').count()

    def delete_duplicate_images(self):
        """重複画像の削除"""
        df = self.images_df(with_hash=True)

        # ハッシュ値が重複している画像のパスを抽出
        hash_checked = []
        to_del_list = []
        for idx, row in df.iterrows():
            img_path = row['path']
            hash_val = row['hash']
            if hash_val in hash_checked:
                to_del_list.append(img_path)
            hash_checked.append(hash_val)

        # 削除実行
        for img_path in to_del_list:
            os.remove(img_path)

        return to_del_list
