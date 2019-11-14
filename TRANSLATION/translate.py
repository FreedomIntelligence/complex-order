
import torch
import argparse
from tqdm import tqdm
from transformer.Translator import Translator
from DataLoader import DataLoader
from preprocess import read_instances_from_file, convert_instance_to_idx_seq

def main():

    parser = argparse.ArgumentParser(description='translate.py')

    parser.add_argument('-model', default='trained.chkpt',
                        help='Path to model .pt file')
    parser.add_argument('-src', default = 'data/multi30k/test.en.atok',
                        help='Source sequence to decode (one line per sequence)')
    parser.add_argument('-ctx', required=False, default="",
                        help='Context sequence to decode (one line per sequence)')
    parser.add_argument('-vocab', default='data/multi30k.atok.low.pt',
                        help='Data that contains the source vocabulary')
    parser.add_argument('-output', default='pred.txt',
                        help="""Path to output the predictions (each line will
                        be the decoded sequence""")
    parser.add_argument('-beam_size', type=int, default=5,
                        help='Beam size')
    parser.add_argument('-batch_size', type=int, default=30,
                        help='Batch size')
    parser.add_argument('-n_best', type=int, default=1,
                        help="""If verbose is set, will output the n_best
                        decoded sentences""")
    parser.add_argument('-no_cuda', action='store_false')
    parser.add_argument('-max_token_seq_len', type=int, default=100)

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda

    # Prepare DataLoader
    preprocess_data = torch.load(opt.vocab)
    preprocess_settings = preprocess_data['settings']

    test_src_word_insts = read_instances_from_file(
        opt.src,
        opt.max_token_seq_len,
        preprocess_settings.keep_case)
    test_src_insts = convert_instance_to_idx_seq(
        test_src_word_insts, preprocess_data['dict']['src'])

    if opt.ctx:
        from preprocess_ctx import read_instances_from_file as read_instances_from_file_ctx
        test_ctx_word_insts = read_instances_from_file_ctx(
            opt.ctx,
            opt.max_token_seq_len,
            preprocess_settings.keep_case,
            is_ctx=True)
        test_ctx_insts = convert_instance_to_idx_seq(
            test_ctx_word_insts, preprocess_data['dict']['src'])

    test_data = DataLoader(
        preprocess_data['dict']['src'],
        preprocess_data['dict']['tgt'],
        src_insts=test_src_insts,
        ctx_insts=(test_ctx_insts if opt.ctx else None),
        cuda=opt.cuda,
        shuffle=False,
        batch_size=opt.batch_size,
        is_train=False)

    translator = Translator(opt)
    translator.model.eval()

    with open(opt.output, 'w') as f:
        for batch in tqdm(test_data, mininterval=2, desc='  - (Test)', leave=False):
            print(---------1111111111)
            all_hyp, all_scores = translator.translate_batch(*batch)
            print(---------2222222222)
            for idx_seqs in all_hyp:
                for idx_seq in idx_seqs:
                    if idx_seq[-1] == 3: # if last word is EOS
                        idx_seq = idx_seq[:-1]
                    pred_line = ' '.join([test_data.tgt_idx2word[int(idx)] for idx in idx_seq])
                    f.write(pred_line + '\n')
            print("end")
            
    print('[Info] Finished.')

if __name__ == "__main__":
    main()
