''' Data Loader class for training iteration '''
import random
import numpy as np
import torch
from torch.autograd import Variable
import transformer.Constants as Constants

class DataLoader(object):

    def __init__(
            self, src_word2idx, tgt_word2idx,
            src_insts=None, tgt_insts=None, ctx_insts=None,
            cuda=True, batch_size=64, shuffle=True,
            is_train=True, sort_by_length=False,
            maxibatch_size=20):

        assert src_insts
        assert len(src_insts) >= batch_size

        if tgt_insts:
            assert len(src_insts) == len(tgt_insts)

        self.cuda = cuda
        self._n_batch = int(np.ceil(len(src_insts) / batch_size))

        self._batch_size = batch_size

        self._src_insts = src_insts
        self._tgt_insts = tgt_insts
        self._ctx_insts = ctx_insts

        src_idx2word = {idx:word for word, idx in src_word2idx.items()}
        tgt_idx2word = {idx:word for word, idx in tgt_word2idx.items()}

        self._src_word2idx = src_word2idx
        self._src_idx2word = src_idx2word

        self._tgt_word2idx = tgt_word2idx
        self._tgt_idx2word = tgt_idx2word

        self._iter_count = 0

        self._need_shuffle = shuffle

        self.is_train = is_train

        if self._need_shuffle:
            self.shuffle()

        self._sort_by_length = sort_by_length

        self._maxibatch_size = maxibatch_size


    @property
    def n_insts(self):
        ''' Property for dataset size '''
        return len(self._src_insts)

    @property
    def src_vocab_size(self):
        ''' Property for vocab size '''
        return len(self._src_word2idx)

    @property
    def tgt_vocab_size(self):
        ''' Property for vocab size '''
        return len(self._tgt_word2idx)

    @property
    def src_word2idx(self):
        ''' Property for word dictionary '''
        return self._src_word2idx

    @property
    def tgt_word2idx(self):
        ''' Property for word dictionary '''
        return self._tgt_word2idx

    @property
    def src_idx2word(self):
        ''' Property for index dictionary '''
        return self._src_idx2word

    @property
    def tgt_idx2word(self):
        ''' Property for index dictionary '''
        return self._tgt_idx2word

    def shuffle(self):
        ''' Shuffle data for a brand new start '''
        if self._tgt_insts:
            if self._ctx_insts:
                paired_insts = list(zip(self._src_insts, self._tgt_insts, self._ctx_insts))
                random.shuffle(paired_insts)
                self._src_insts, self._tgt_insts, self._ctx_insts = zip(*paired_insts)
            else:
                paired_insts = list(zip(self._src_insts, self._tgt_insts))
                random.shuffle(paired_insts)
                self._src_insts, self._tgt_insts = zip(*paired_insts)
        else:
            if self._ctx_insts:
                paired_insts = list(zip(self._src_insts, self._ctx_insts))
                random.shuffle(paired_insts)
                self._src_insts, self._ctx_insts = zip(*paired_insts)
            else:
                random.shuffle(self._src_insts)


    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self._n_batch

    def next(self):
        ''' Get the next batch '''

        def pad_to_longest(insts):
            ''' Pad the instance to the max seq length in batch '''

            max_len = max(len(inst) for inst in insts)

            inst_data = np.array([
                inst + [Constants.PAD] * (max_len - len(inst))
                for inst in insts])

            inst_position = np.array([
                [pos_i+1 if w_i != Constants.PAD else 0 for pos_i, w_i in enumerate(inst)]
                for inst in inst_data])
            with torch.no_grad():
                # inst_data_tensor = Variable(torch.LongTensor(inst_data), volatile=(not self.is_train))
                # inst_position_tensor = Variable(torch.LongTensor(inst_position), volatile=(not self.is_train))
                inst_data_tensor = Variable(torch.LongTensor(inst_data))
                inst_position_tensor = Variable(torch.LongTensor(inst_position))
            if self.cuda:
                inst_data_tensor = inst_data_tensor.cuda()
                inst_position_tensor = inst_position_tensor.cuda()
            return inst_data_tensor, inst_position_tensor

        if self._iter_count < self._n_batch:
            
            batch_idx = self._iter_count
            self._iter_count += 1


            if self._sort_by_length:

                assert self._tgt_insts, 'Target must be provided to do sort_by_length'

                if batch_idx % self._maxibatch_size == 0:

                    start_idx = batch_idx * self._batch_size
                    end_idx = (batch_idx + self._maxibatch_size) * self._batch_size

                    src_insts = self._src_insts[start_idx:end_idx]
                    tgt_insts = self._tgt_insts[start_idx:end_idx]

                    tlen = np.array([len(t) for t in tgt_insts])
                    tidx = tlen.argsort()

                    self._sbuf = [src_insts[i] for i in tidx]
                    self._tbuf = [tgt_insts[i] for i in tidx]

                    if self._ctx_insts:
                        ctx_insts = self._ctx_insts[start_idx:end_idx]
                        self._cbuf = [ctx_insts[i] for i in tidx]

                cur_start = (batch_idx % self._maxibatch_size) * self._batch_size
                cur_end = ((batch_idx % self._maxibatch_size) + 1) * self._batch_size

                cur_src_insts = self._sbuf[cur_start:cur_end]
                src_data, src_pos = pad_to_longest(cur_src_insts)

                cur_tgt_insts = self._tbuf[cur_start:cur_end]
                tgt_data, tgt_pos = pad_to_longest(cur_tgt_insts)

                if self._ctx_insts:
                    cur_ctx_insts = self._cbuf[cur_start:cur_end]
                    ctx_data, ctx_pos = pad_to_longest(cur_ctx_insts)

                    return (src_data, src_pos), (tgt_data, tgt_pos), (ctx_data, ctx_pos)

                else:
                    return (src_data, src_pos), (tgt_data, tgt_pos)

            else:
                start_idx = batch_idx * self._batch_size
                end_idx = (batch_idx + 1) * self._batch_size

                src_insts = self._src_insts[start_idx:end_idx]
                src_data, src_pos = pad_to_longest(src_insts)

                if self._ctx_insts:
                    ctx_insts = self._ctx_insts[start_idx:end_idx]
                    ctx_data, ctx_pos = pad_to_longest(ctx_insts)
                
                if not self._tgt_insts:
                    if self._ctx_insts:
                        return (src_data, src_pos), (ctx_data, ctx_pos)
                    else:
                        return src_data, src_pos
                else:
                    tgt_insts = self._tgt_insts[start_idx:end_idx]
                    tgt_data, tgt_pos = pad_to_longest(tgt_insts)
                    if self._ctx_insts:
                        return (src_data, src_pos), (tgt_data, tgt_pos), (ctx_data, ctx_pos)
                    else:
                        return (src_data, src_pos), (tgt_data, tgt_pos)

        else:

            if self._need_shuffle:
                self.shuffle()

            self._iter_count = 0
            raise StopIteration()
