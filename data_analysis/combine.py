import shutil
import json
import os
import numpy as np

# Creating easy dataset class and listing the exact results in easy_logans.json
ds_root = "../benchmarks/altogether"

psw_ans = json.load(open(os.path.join(ds_root, "pseudoweighted_logans.json")))
bay_ans = json.load(open(os.path.join(ds_root, "bayes_logans.json")))

psw_ans = {k: v for k, v in psw_ans.items() if not v is None and not np.isneginf(v)}
bay_ans = {k: v for k, v in bay_ans.items() if not v is None and not np.isneginf(v)}

for fn in psw_ans:
	src = os.path.join(ds_root, "pseudoweighted_MINIC2D", fn)
	dst = os.path.join(ds_root, "easy", "pseudoweighted_" + fn)
	shutil.copy(src, dst)

for fn in bay_ans:
	src = os.path.join(ds_root, "bayes_MINIC2D", fn)
	dst = os.path.join(ds_root, "easy", "bayes_" + fn)
	shutil.copy(src, dst)

psw_ans_new = {"pseudoweighted_" + fn: ans for fn, ans in psw_ans.items()}
bay_ans_new = {"bayes_" + fn: ans for fn, ans in bay_ans.items()}

json.dump(psw_ans_new | bay_ans_new, open(os.path.join(ds_root, "easy_logans.json"), "w"), indent=4)
