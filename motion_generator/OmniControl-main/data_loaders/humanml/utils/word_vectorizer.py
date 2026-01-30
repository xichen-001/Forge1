# Minimal, robust WordVectorizer for OmniControl reproduction
# - Provides POS_enumerator (used by get_opt)
# - Loads meta files from meta_root/prefix_* and builds word2idx & word2vec
# - Offers tokens2idx and get_vector helpers

import os
import pickle
from os.path import join as pjoin

import numpy as np

# Minimal POS enumerator (length matters only for get_opt)
POS_enumerator = ["NOUN", "VERB", "ADJ", "ADV", "PRON", "DET", "ADP", "NUM", "CONJ", "PRT", "X", "INTJ"]


class WordVectorizer:
    def __init__(self, meta_root, prefix):
        """
        meta_root: path to meta dir (string)
        prefix: dataset prefix, e.g. 't2m' (string)
        Expects files like:
          - {prefix}_data.npy   (metadata dict)
          - {prefix}_word2id.npy (np.save of dict) or .pkl
          - {prefix}_wordvec.npy (2D array)
          - {prefix}_words.pkl (pickle list of words)  (optional)
        This implementation is robust to object-arrays and .pkl variants.
        """
        self.meta_root = meta_root
        self.prefix = prefix
        self.word2idx = {"<pad>": 0, "<unk>": 1}
        self.word2vec = {}
        self.words = ["<pad>", "<unk>"]  # index 0,1
        self.dim_word = 300

    def __getitem__(self, token):
        """
        返回: (word_emb, pos_onehot)
        - word_emb: np.ndarray [dim_word]
        - pos_onehot: np.ndarray [len(POS_enumerator)]（如果没有就返回长度为0的向量）
        """
        import numpy as np

        # 取 '<unk>' 的向量作为兜底
        dim = getattr(self, "dim_word", 300)
        unk_vec = self.word2vec.get("<unk>", np.zeros((dim,), dtype=np.float32))
        vec = self.word2vec.get(token, unk_vec)

        # POS one-hot（本仓库未必提供POS；没有就返回长度为0的零向量）
        try:
            pos_len = len(POS_enumerator)
        except NameError:
            pos_len = 0
        pos_oh = np.zeros((pos_len,), dtype=np.float32) if pos_len > 0 else np.zeros((0,), dtype=np.float32)

        return vec, pos_oh

        # helper paths
        def _path(name):
            return os.path.join(meta_root, name)

        # 1) load meta data (if any)
        data_path = _path(f"{prefix}_data.npy")
        if os.path.exists(data_path):
            try:
                meta = np.load(data_path, allow_pickle=True)
                # could be 0-d ndarray wrapping a dict
                if isinstance(meta, np.ndarray) and meta.shape == ():
                    meta = meta.item()
                if isinstance(meta, dict):
                    self.dim_word = int(meta.get("dim_word", self.dim_word))
            except Exception:
                pass

        # 2) try load word2id mapping (np.npy or pickle)
        w2id = None
        w2id_npy = _path(f"{prefix}_word2id.npy")
        if os.path.exists(w2id_npy):
            try:
                tmp = np.load(w2id_npy, allow_pickle=True)
                # if it's 0-d array holding dict
                if isinstance(tmp, np.ndarray) and tmp.shape == ():
                    try:
                        w2id = tmp.item()
                    except Exception:
                        # fallback if item is not dict
                        if isinstance(tmp.tolist(), dict):
                            w2id = tmp.tolist()
                        else:
                            w2id = None
                elif isinstance(tmp, dict):
                    w2id = tmp
                else:
                    # maybe saved as array-of-objects; try to get item
                    try:
                        w2id = tmp.item()
                    except Exception:
                        w2id = None
            except Exception:
                w2id = None

        # 3) try load pkl words/idx variants if needed
        words_pkl = _path(f"{prefix}_words.pkl")
        idx_pkl = _path(f"{prefix}_idx.pkl")
        # Also handle legacy names 'our_vab_*' — user repo may have these
        our_words = _path("our_vab_words.pkl")
        our_idx = _path("our_vab_idx.pkl")
        our_data = _path("our_vab_data.npy")

        if w2id is None:
            # try idx pkl first
            if os.path.exists(idx_pkl):
                try:
                    with open(idx_pkl, "rb") as f:
                        idx_obj = pickle.load(f)
                except TypeError:
                    idx_obj = pickle.load(open(idx_pkl, "rb"), encoding="latin1")
                if isinstance(idx_obj, dict):
                    # assume word->id
                    w2id = {
                        str(k): int(v)
                        for k, v in idx_obj.items()
                        if (isinstance(v, (int, np.integer)) or (isinstance(v, str) and v.isdigit()))
                    }
                    # if values are strings non-digits, fallback handled later
            elif os.path.exists(our_idx):
                try:
                    with open(our_idx, "rb") as f:
                        idx_obj = pickle.load(f)
                except TypeError:
                    idx_obj = pickle.load(open(our_idx, "rb"), encoding="latin1")
                if isinstance(idx_obj, dict):
                    # If this idx is word->id (likely), use it
                    # Otherwise we will align with words
                    # Check value types:
                    vals = list(idx_obj.values())
                    if vals and all(isinstance(v, (int, np.integer)) for v in vals):
                        w2id = {str(k): int(v) for k, v in idx_obj.items()}

            # if still none, try words list + our_idx list
            if w2id is None and os.path.exists(our_words):
                try:
                    with open(our_words, "rb") as f:
                        words_list = pickle.load(f)
                except TypeError:
                    words_list = pickle.load(open(our_words, "rb"), encoding="latin1")
                # if we have our_vab_idx as list/array, align
                if os.path.exists(our_idx):
                    try:
                        with open(our_idx, "rb") as f:
                            idx_obj = pickle.load(f)
                    except TypeError:
                        idx_obj = pickle.load(open(our_idx, "rb"), encoding="latin1")
                    if isinstance(idx_obj, (list, tuple, np.ndarray)) and len(idx_obj) == len(words_list):
                        w2id = {}
                        for w, i in zip(words_list, idx_obj):
                            try:
                                w2id[str(w)] = int(i)
                            except Exception:
                                pass
                # otherwise enumerate words_list
                if w2id is None and isinstance(words_list, (list, tuple)):
                    w2id = {}
                    cur = 2
                    for w in words_list:
                        if str(w) not in w2id:
                            w2id[str(w)] = cur
                            cur += 1

        # If we still don't have mapping, try load words pkl then enumerate
        if w2id is None and os.path.exists(words_pkl):
            try:
                with open(words_pkl, "rb") as f:
                    words_list = pickle.load(f)
            except TypeError:
                words_list = pickle.load(open(words_pkl, "rb"), encoding="latin1")
            w2id = {}
            cur = 2
            for w in words_list:
                if str(w) not in w2id:
                    w2id[str(w)] = cur
                    cur += 1

        # Final fallback: minimal vocab
        if w2id is None:
            w2id = {"<pad>": 0, "<unk>": 1}
        # Normalize mapping to contiguous ids starting from 0: ensure pad 0 unk 1
        # Build sorted words excluding pad and unk
        keys = [k for k in w2id.keys() if k not in ("<pad", "<unk>")]
        keys_sorted = sorted(set(keys))
        mapping = {"<pad>": 0, "<unk>": 1}
        cur = 2
        for w in keys_sorted:
            mapping[w] = cur
            cur += 1
        self.word2idx = mapping
        # words list by id
        max_id = max(self.word2idx.values())
        words_by_id = [""] * (max_id + 1)
        for w, i in self.word2idx.items():
            if i < len(words_by_id):
                words_by_id[i] = w
        self.words = words_by_id

        # 4) load word vectors
        # prefer explicit prefix_wordvec.npy; else try our_vab_data.npy
        vec_arr = None
        vec_path = _path(f"{prefix}_wordvec.npy")
        if os.path.exists(vec_path):
            try:
                vec_arr = np.load(vec_path, allow_pickle=True)
                # If 0-d object, unwrap
                if isinstance(vec_arr, np.ndarray) and vec_arr.shape == ():
                    try:
                        vec_arr = vec_arr.item()
                    except Exception:
                        pass
            except Exception:
                vec_arr = None
        if vec_arr is None and os.path.exists(our_data):
            try:
                vec_arr = np.load(our_data, allow_pickle=True)
            except Exception:
                vec_arr = None

        # if vec_arr is a dict (id->vec or word->vec), try to convert to ndarray if numeric keys
        if isinstance(vec_arr, dict):
            vals = list(vec_arr.values())
            if vals and all(isinstance(k, (int, np.integer)) for k in vec_arr.keys()):
                max_k = max(int(k) for k in vec_arr.keys())
                dim = len(np.asarray(vals[0]))
                arr = np.zeros((max_k + 1, dim), dtype=np.float32)
                for k, v in vec_arr.items():
                    try:
                        arr[int(k)] = np.asarray(v, dtype=np.float32)
                    except Exception:
                        pass
                vec_arr = arr
            else:
                # leave as dict word->vec
                pass

        if isinstance(vec_arr, np.ndarray):
            # ensure shape[0] >= vocab size: pad or trim
            try:
                vocab_size = len(self.word2idx)
                if vec_arr.ndim == 1:
                    # single vector -> replicate or treat as scalar (not expected)
                    vec_arr = np.expand_dims(vec_arr, 0)
                if vec_arr.shape[0] < vocab_size:
                    extra = np.zeros((vocab_size - vec_arr.shape[0], vec_arr.shape[1]), dtype=np.float32)
                    vec_arr = np.vstack([vec_arr.astype(np.float32), extra])
                elif vec_arr.shape[0] > vocab_size:
                    vec_arr = vec_arr[:vocab_size].astype(np.float32)
                self.dim_word = int(vec_arr.shape[1])
            except Exception:
                pass
        else:
            # fallback create random vectors
            rng = np.random.RandomState(12345)
            vocab_size = len(self.word2idx)
            vec_arr = rng.normal(scale=0.1, size=(vocab_size, self.dim_word)).astype(np.float32)
            vec_arr[0] = np.zeros(self.dim_word, dtype=np.float32)

        # Build word2vec dict mapping word->vector
        for w, idx in self.word2idx.items():
            try:
                self.word2vec[w] = np.asarray(vec_arr[int(idx)], dtype=np.float32)
            except Exception:
                self.word2vec[w] = np.zeros(self.dim_word, dtype=np.float32)

    def tokens2idx(self, tokens, max_len=None):
        """Convert a list of tokens (strings) to list of indices (ints)."""
        res = []
        for t in tokens:
            if t in self.word2idx:
                res.append(self.word2idx[t])
            else:
                res.append(self.word2idx.get("<unk>", 1))
        if max_len is not None:
            if len(res) < max_len:
                res += [self.word2idx.get("<pad>", 0)] * (max_len - len(res))
            else:
                res = res[:max_len]
        return res

    def get_vector(self, word):
        """Return vector for a single word (or zero vector)."""
        return self.word2vec.get(word, np.zeros(self.dim_word, dtype=np.float32))

    def encode(self, tokens, max_len=None):
        """Return ids and a matrix of vectors for tokens (ids padded/truncated to max_len if provided)."""
        ids = self.tokens2idx(tokens, max_len=max_len)
        mat = (
            np.stack([
                self.get_vector(w if i >= 2 else self.words[i]) if i < len(self.words) else self.get_vector("<unk>")
                for (w, i) in zip(tokens + [""] * (max(0, (max_len or len(ids)) - len(tokens))), ids)
            ])
            if ids
            else np.zeros((0, self.dim_word), dtype=np.float32)
        )
        return ids, mat
