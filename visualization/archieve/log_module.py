

def log_module(module, input, tb_writer):
    """
        input:
            module : target module,
            input  : for module input
        return None
    """
    tb_writer.add_graph(module, input)
    return None
