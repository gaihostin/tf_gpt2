


def parse_arguments():
    pass 

def train(configs):
    params = {
        "num_layers": configs['num_layers'],
        "d_model": configs["embedding_size"],
        "num_heads": configs["num_heads"],
        "dff": configs['dff'],
        "max_seq_len": configs['max_seq_len'],
        "vocab_size": configs['vocab_size']
    }


    if config['mode'] == 'local': 
        pass 


if __name__ == "__main__":

    configs = parse_arguments()

    train(configs)