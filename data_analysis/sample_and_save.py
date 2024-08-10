import numpy as np
import os
import time

from .utils import readCNF, sample_y


if __name__ == "__main__":
    sample_size = 100000
    file_mode = "MIN"

    # torch.manual_seed(42)
    np.random.seed(42)

    ds_root = "../benchmarks/altogether"
    ds_name = "easy"
    ds_path = os.path.join(ds_root, ds_name)
    print("Sampling from dataset class:", ds_path)
    smp_path = os.path.join(ds_root, ds_name + "_samples")
    print("Samples saving to:", smp_path)

    all_items = os.listdir(ds_path)
    files = [fn for fn in all_items if os.path.isfile(os.path.join(ds_path, fn))]

    all_items = os.listdir(smp_path)
    files_processed = [fn for fn in all_items if
                    os.path.isfile(os.path.join(smp_path, fn)) and
                    os.path.getsize(os.path.join(smp_path, fn)) > 0
                    ]

    for fn in files:
        if fn + ".npy" in files_processed:
            continue
        print(f"Processing {fn}:")
        with open(os.path.join(ds_path, fn)) as f:
            cnf, weights, _ = readCNF(f, mode=file_mode)
            
        probs = (weights / weights.sum(axis=1, keepdims=True))[:, 0]
        
        t0 = time.time()
        y = sample_y(probs, cnf, size=sample_size)
        t1 = time.time()
        print(f"Sampling time: {t1 - t0:.2f}")
        
        np.save(open(os.path.join(smp_path, fn + ".npy"), "wb"), y)
