import torch
from torch import nn
from peft.tuners.tuners_utils import check_target_module_exists
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ProcessPoolExecutor, as_completed

from cotan.CoTAn import CoTAnLayer


def get_layer(model, name):
    layer = model
    for attr in name.split("."):
        layer = getattr(layer, attr)
    return layer


def set_layer(model, name, layer):
    attrs, name = name.rsplit(".", 1)
    model = get_layer(model, attrs)
    setattr(model, name, layer)


def _get_total_parameters(model):
    return sum(p.numel() for p in model.parameters())


def _get_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_num_trainable(model):
    total_params = _get_total_parameters(model)
    trainable_params = _get_trainable_parameters(model)
    print(trainable_params, total_params)
    frac = (float(trainable_params) / total_params) * 100
    
    print(f"trainable: {trainable_params}  |  total: {total_params}  |  trainable(%): {frac:.6f}")


def process_layer(name, module, cotan_config, logs=None):
    if not check_target_module_exists(cotan_config, name):
        return (name, None)

    if not isinstance(module, nn.Linear):
        return (name, None)

    out_f, in_f = module.weight.shape
    kwargs = {
        'r': cotan_config.r,
        'alpha': cotan_config.alpha,
        'init_strategy': cotan_config.init_strategy,
        'X': cotan_config.dic[name] if hasattr(cotan_config, 'dic') else None,
    }
    
    # if logs != None:
    #     make_logging_prev(cotan_config, name, logs)

    cotan_layer = CoTAnLayer(
        module,
        in_features=in_f,
        out_features=out_f,
        **kwargs
    )

    if logs != None:
        make_logging(cotan_layer, name, logs)
        
    return (name, cotan_layer)


def inject_cotan(cotan_config, model, logs=None):
    model_adapter = model

    for param_name, param in model_adapter.named_parameters():
        param.requires_grad = False
        

    futures = {}
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    with ThreadPoolExecutor(max_workers=1) as executor:
        for name, module in model_adapter.named_modules():
            future = executor.submit(process_layer, name, module, cotan_config, logs)
            futures[future] = name

        for future in as_completed(futures):
            name, cotan_layer = future.result()
            if cotan_layer is not None:
                set_layer(model_adapter, name, cotan_layer)
                print(f"Setting adapter at {name:20} layer")

    print_num_trainable(model_adapter)
    return model_adapter


def prepare_get_samples(model, cotan_config):
    cotan_config.dic = {}
    hooks = []
    
    for name, module in model.named_modules():
        if check_target_module_exists(cotan_config, name):
            hooks.append(module.register_forward_pre_hook(_calculate(name, cotan_config.dic)))

    return model, hooks
            
            
def _calculate(name, dic):
        def hook(model, input):
            X = input[0].cpu()
            X = X.permute(2, 1, 0).reshape(X.shape[2], X.shape[0] * X.shape[1])
            prev = dic.get(name)
            if prev != None:
                dic[name] =  torch.cat((prev, X), dim=1)
            else:
                dic[name] = X
            del X

        return hook
    
    
def after_get_samples(model, cotan_config, hooks):
    for h in hooks:
        h.remove()
    
    
def make_logging(cotan_layer, name, logs):
    A = cotan_layer.adapter.adapter_A
    B = cotan_layer.adapter.adapter_B
    
    # logs.setdefault("W_s", {})
    # _, S, _ = torch.linalg.svd(cotan_layer.pre_layer.weight, full_matrices=False)
    # logs["W_s"][name] = S.tolist()
    
    logs.setdefault("norms", {})
    logs.setdefault("to_orthogonal", {})
    logs.setdefault("shapes", {})

    logs["norms"][name] = {"A": torch.norm(A).item(), "B": torch.norm(B).item()}
    logs["shapes"][name] = {"A": A.shape, "B": B.shape}
    I = torch.eye(A.shape[1]).to('cuda')
    logs["to_orthogonal"][name] = {"A": torch.dist(A.T @ A, I).item(), "B": torch.dist(B @ B.T, I).item()}
    

def make_logging_prev(cotan_config, name, logs):
    
    logs.setdefault("X", {})
    print('MEOW')
    logs["X"][name] = cotan_config.dic[name]
    