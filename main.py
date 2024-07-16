import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.distributions.bernoulli import Bernoulli
import math
import os
import json
import numpy as np
import logging
import cpuinfo
import wandb

from data_analysis.utils import readCNF
from argparser import parsearg
from model import IndependentModel, HMM, inhHMM

import random
import matplotlib.pyplot as plt

from cirkit.templates.region_graph import LinearRegionGraph
from cirkit.symbolic.circuit import Circuit
from cirkit.pipeline import PipelineContext
from cirkit_factories import categorical_layer_factory, hadamard_layer_factory, dense_layer_factory, mixing_layer_factory


if __name__ == "__main__":
    torch.cuda.empty_cache()
    
    # seed
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(42)
    ds_root = "benchmarks/altogether"
    
    # configurations
    args = parsearg()
    config = {
        'ds_class': args.dsclass,
        'file_name': args.filename,

	    'sample_size': args.sample_size,
	    'batch_size': args.batch_size,
	    'learning_rate': args.lr,
	    'model': args.model
    }
    model_cf = args.model
    if config['model'] == 'hmm' or config['model'] == 'inh':
        config['num_state'] = args.num_state
        model_cf += "(" + str(args.num_state) + ")"
    config_str = f"{config['file_name']}-{model_cf}-bs{config['batch_size']}-lr{config['learning_rate']}"

    # logger
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("logs/" + config_str + ".log", mode='w'),
            logging.StreamHandler()
        ])
    logger = logging.getLogger()

    # device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f'GPU: {torch.cuda.get_device_name(0)}')
    else:
        device = torch.device('cpu')
        logger.info(f"CPU: {cpuinfo.get_cpu_info()['brand_raw']}")
    logger.info(f'Using device: {device}')

    # model path
    modelpth = os.path.join(args.modelpth, config_str + ".pth")
    logger.info(f"Model path: {modelpth}")

    # wandb config
    wandb.init(project="approxWMC", config=config)
    wandb.config.update(config)

    # exact WMC
    exact_result_path = os.path.join(ds_root, "easy_logans.json")
    with open(exact_result_path) as ans:
        exact_ans = json.load(ans)
    log_exact_prob = exact_ans[config['file_name']]

    # Dataset
    # cnf_path = os.path.join(ds_root, config['ds_class'], config['file_name'])
    # with open(cnf_path) as f:
    #     cnf, weights, _ = readCNF(f, mode=args.format)
    # clscnt, varcnt = len(cnf), len(weights)
    sample_path = os.path.join(ds_root, config['ds_class'] + "_samples", config['file_name'] + ".npy")
    with open(sample_path, "rb") as f:
        y = torch.from_numpy(np.load(f))[:config['sample_size'], ]
    clscnt = y.shape[1]

    ds = TensorDataset(y)
    train_ds, val_ds = random_split(ds, [0.8, 0.2])

    # Dataloader
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'])

    # model & optimiser
    if config['model'] == 'ind':
        model = IndependentModel(dim=clscnt, device=device)
    elif config['model'] == 'hmm':
        model = HMM(dim=clscnt, device=device, num_states=config['num_state'])
    elif config['model'] == 'inh':
        model = inhHMM(dim=clscnt, device=device, num_states=config['num_state'])
    else:
        logger.info("Start constructing circuits:")
        region_graph = LinearRegionGraph(num_variables=clscnt)
        
        symbolic_circuit = Circuit.from_region_graph(
            region_graph,
            num_input_units=100,
            num_sum_units=100,
            input_factory=categorical_layer_factory,
            sum_factory=dense_layer_factory,
            prod_factory=hadamard_layer_factory,
            mixing_factory=mixing_layer_factory
        )
        logger.info(f'Smooth: {symbolic_circuit.is_smooth}')
        logger.info(f'Decomposable: {symbolic_circuit.is_decomposable}')
        logger.info(f'Number of variables: {symbolic_circuit.num_variables}')

        ctx = PipelineContext(
            backend='torch',   # Choose the torch compilation backend
            fold=True,         # Fold the circuit, this is a backend-specific compilation flag
            semiring='lse-sum' # Use the (R, +, *) semiring, where + is the log-sum-exp and * is the sum
        )
        model = ctx.compile(symbolic_circuit)
        
        logger.info(f'Layer counts: {len(list(symbolic_circuit.layers))}')
        logger.debug(f'Circuit: {model}')
        
        pf_model = ctx.integrate(model)
        logger.info(f'Circuit type: {type(pf_model)}')

        model = model.to(device)
        pf_model = pf_model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    epoch = 0
    # early stopping param
    patience = 5
    best_val_loss = float('inf')
    counter = 0

    # training loop
    while True:
        model.train()
        train_loss = 0
        for i, batch in enumerate(train_loader):
            batch = batch[0].to(device).unsqueeze(dim=1) if config['model'] == 'pcs' else batch[0].to(device)
            log_output = model(batch)
            lls = log_output - pf_model() if config['model'] == 'pcs' else log_output
            nll = - torch.mean(lls)
            nll.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += nll.item()
            
            if i % 100 == 0:
                logger.info(f"Batch {i} loss: {nll}")
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            alltrue = torch.ones((1, clscnt), device=device)
            if config['model'] == 'pcs':
                log_pf = pf_model()
                alltrue = alltrue.unsqueeze(dim=1)
            log_output = model(alltrue)
            lls = log_output - log_pf if config['model'] == 'pcs' else log_output
            loglogMAE = abs(lls.item() - log_exact_prob)

            for batch in val_loader:
                batch = batch[0].to(device)
                if config['model'] == 'pcs': batch = batch.unsqueeze(dim=1)
                log_output = model(batch)
                lls = log_output - log_pf if config['model'] == 'pcs' else log_output
                nll = - torch.mean(lls)
                val_loss += nll.item()
        val_loss /= len(val_loader)

        logger.info(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, log-log MAE: {loglogMAE:.4f}')

        wandb.log({'Epoch': epoch, 'Train Loss': train_loss, 'Validation Loss': val_loss, 'log-log MAE': loglogMAE})

        # early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), modelpth)
        else:
            counter += 1
            if counter >= patience:
                logger.info("Early stopping")
                break

        epoch += 1

    # approx WMC
    model.load_state_dict(torch.load(modelpth))
    model.eval()
    with torch.no_grad():
        alltrue = torch.ones((1, clscnt), device=device)
        if config['model'] == 'pcs':
            log_pf = pf_model()
            alltrue = alltrue.unsqueeze(dim=1)
        log_output = model(alltrue)
        lls = log_output - log_pf if config['model'] == 'pcs' else log_output

    logger.info(f'Approx WMC: {math.exp(lls.item())}')
    logger.info(f'Exact WMC: {math.exp(log_exact_prob)}')

    # log sacle error
    loglogMAE = abs(lls.item() - log_exact_prob)
    logger.info(f'log-log MAE: {loglogMAE}')

    wandb.finish()
