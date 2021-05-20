from networks  import *
import argparse 


def main(args):
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    experiment = Framework(args)
    experiment.show_model()
    experiment.train()
    result = experiment.test()
    output_file(args.train_data_path, result)
    
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed'           , type = int   , default = 1)
    parser.add_argument('--device'         , type = str   , default = 'cuda')
    parser.add_argument('--epoch'          , type = int   , default = 5)
    parser.add_argument('--lr'             , type = float , default = 1e-4)
    parser.add_argument('--batch_size'     , type = int   , default = 64)
    parser.add_argument('--train_data_path', type = str   ,default = './../../data')
    args = parser.parse_args()
    print(args)
    print("-"*100)
    main(args)