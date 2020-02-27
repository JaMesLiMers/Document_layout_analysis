# try:
#     from torch.utils.tensorboard import SummaryWriter
# except Exception as e:
# from tensorboardX import SummaryWriter

# log_dir = "./board"
# tb_writer = SummaryWriter(log_dir)

def log_module(module, input, tb_writer):
    """
        input:
            module : target module,
            input  : for module input
        return None
    """
    tb_writer.add_graph(module, input)
    return None
    
# from tensorboardX import SummaryWriter

def log_grads(module, tb_writer, tb_index, tb_name):
    """
    log a particular module grad
    input:
        module   : Class(nn.module),
        tb_writer: tb_writer,
        tb_index : epoch/step,
        tb_name  : module_name
    """
    def weights_grads(module):
        """
        get moudle's grad and weights
        imput: 
            module: Class(nn.module)
        return: 
            grad: {name:grad}, weights: {name:weight}
        """
        grad = {}
        weights = {}
        for name, param in module.named_parameters():
            if param.grad is not None:
                grad[name] = param.grad
                weights[name] = param.data
        return grad, weights
    # get module's grad and weights
    grad, weights = weights_grads(module)   
    tot_norm = 0
    for k, g in grad.items():
        _norm = g.data.norm(2)
        weight = weights[k]
        w_norm = weight.norm(2)
        tot_norm += _norm ** 2

        tb_writer.add_scalar('grad_weight/grad_{}/'.format(tb_name)+k.replace('.', '/'),
                             _norm, tb_index)
        tb_writer.add_scalar('grad_weight/weight_{}/'.format(tb_name)+k.replace('.', '/'),
                             w_norm, tb_index)
        tb_writer.add_scalar('grad_weight/w-g_{}/'.format(tb_name)+k.replace('.', '/'),
                             w_norm/(1e-20 + _norm), tb_index)
    tot_norm = tot_norm ** 0.5

    tb_writer.add_scalar('grad_weight/grad_{}_total/'.format(tb_name), tot_norm, tb_index)