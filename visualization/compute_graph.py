from torchviz import make_dot, make_dot_from_trace
import torch

def log_grad_graph(module, input, log_name, log_dir, view=False):
    """
        input:
            module : target module,
            input  : for module input
            log_dir: output place
        return Digraph
    """
    out = make_dot(module(input), params=dict(list(module.named_parameters()) + [('input', input)]))
    out.render(filename=log_name ,directory=log_dir, view=view, cleanup=True)
    return out

def log_grad_graph_from_trace(module, input, log_name, log_dir, view=False):
    """
        (not always work!)
        input:
            module : target module,
            input  : for module input
            log_dir: output place
        return Digraph
    """
    with torch.onnx.set_training(module, False):
        trace, _ = torch.jit.get_trace_graph(module, args=(input,))
    out = make_dot_from_trace(trace)
    out.render(filename=log_name, directory=log_dir, view=view, cleanup=True)
    return out