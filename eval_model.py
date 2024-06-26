import torch
from model import IndependentModel, HMM
import math
import json

model = HMM(dim=577, num_states=10)

modelpth = "models/easy/bayes_90-12-4-q.cnf.pth"
filename = "bayes_90-12-4-q.cnf"
# approx WMC
model.load_state_dict(torch.load(modelpth))
model.eval()
with torch.no_grad():
	log_prob = model.log_p(torch.ones(577).unsqueeze(0)).item()
print(f'Approx WMC: {math.exp(log_prob)}')

# exact WMC
with open("benchmarks/altogether/easy_logans.json") as ans:
	exact_ans = json.load(ans)
log_exact_prob = exact_ans[filename]
print(f'Exact WMC: {math.exp(log_exact_prob)}')

# log sacle error
loglogMAE = abs(log_prob - log_exact_prob)
print(f'log-log MAE: {loglogMAE}')