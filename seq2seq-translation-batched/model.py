import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence#, masked_cross_entropy
from masked_cross_entropy import * 
# 其实使用 packedSequence 直接规避掉这个或许会更好， 先 pack 通过 RNN 然后再次 pad


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, dropout=0.1):
        super(EncoderRNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=True)
        # 注意， 默认是双向 RNN， 所以output的时候是形状是 (T,B,DH) 或者 (S,DH), hidden 是 (LD, B, H)
        
    def forward(self, input_seqs, input_lengths, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(input_seqs) # (T, B, H)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths) # (S, H)
        outputs, hidden = self.gru(packed, hidden) # (S, DH) , (LD, B, H)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs) # unpack (back to padded)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:] # Sum bidirectional outputs
        return outputs, hidden #(T,B,H), (LD,B,H)

        
# A different implementation of the global attention which avoids for-loop
class Attn2(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn2, self).__init__()
        
        self.method = method
        self.hidden_size = hidden_size
        
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn_e = nn.Linear(self.hidden_size, hidden_size)
            self.attn_h = nn.Linear(self.hidden_size, hidden_size)
            self.v = nn.Linear(hidden_size, 1)
    
    def forward(self, hidden, encoder_outputs):
        # hidden (1,B,DH), encoder_outputs (T, B, DH)
        if self.method == 'general':
            attn_energies = torch.bmm(hidden.transpose(0,1), 
                                      self.attn(encoder_outputs).transpose(0,1).transpose(1,2))
        if self.method == 'dot':
            attn_energies = torch.bmm(hidden.transpose(0,1), 
                                      encoder_outputs.transpose(0,1).transpose(1,2))
        if self.method == 'concat':
            attn_energies = self.v(self.attn_h(hidden) + self.attn_e(encoder_outputs)).transpose(0,1).transpose(1,2)
            
        return F.softmax(attn_energies, dim=-1)


class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        
        # Choose attention model
        if attn_model != 'none':
            self.attn = Attn2(attn_model, hidden_size)

    def forward(self, input_seq, last_hidden, encoder_outputs):
        # Note: we run this one step at a time

        # Get the embedding of the current input word (last output word)
        batch_size = input_seq.size(0) # 就是得到了 batch_size, 形状可能是 (B,1) 或者 (B)
        embedded = self.embedding(input_seq)
        embedded = self.embedding_dropout(embedded)
        embedded = embedded.view(1, batch_size, self.hidden_size) # S=1 x B x N #embedded(1,B,H)

        # Get current hidden state from input word and last hidden state
        rnn_output, hidden = self.gru(embedded, last_hidden) # （1，B，DH）, (LD,B,H) #(1,B,H), (L,B,H)

        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average
        attn_weights = self.attn(rnn_output, encoder_outputs) #(B,1,T)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1)) # B x S=1 x N # (B, 1, H)

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        rnn_output = rnn_output.squeeze(0) # S=1 x B x N -> B x N #(B,H)
        context = context.squeeze(1)       # B x S=1 x N -> B x N #(B,H)
        concat_input = torch.cat((rnn_output, context), 1) #(B,2H)
        concat_output = F.tanh(self.concat(concat_input)) #(B,H)

        # Finally predict next token (Luong eq. 6, without softmax)
        output = self.out(concat_output) #(B,O), not logsoftmaxed

        # Return final output, hidden state, and attention weights (for visualization)
        return output, hidden, attn_weights # (B,O), (L,B,H), (B,1,T)


if __name__ == "__main__":
    # 以下就是一個例子
    from data import *
    small_batch_size = 3
    input_batches, input_lengths, target_batches, target_lengths = random_batch(small_batch_size)

    print('input_batches', input_batches.size()) # (max_len x batch_size)
    print('target_batches', target_batches.size()) # (max_len x batch_size)

    small_hidden_size = 8
    small_n_layers = 2

    encoder_test = EncoderRNN(input_lang.n_words, small_hidden_size, small_n_layers)
    decoder_test = LuongAttnDecoderRNN('general', small_hidden_size, output_lang.n_words, small_n_layers)
    # decoder_test = BahdanauAttnDecoderRNN(small_hidden_size, output_lang.n_words, small_n_layers)

    if USE_CUDA:
        encoder_test.cuda()
        decoder_test.cuda()

    encoder_outputs, encoder_hidden = encoder_test(input_batches, input_lengths, None)

    print('encoder_outputs', encoder_outputs.size()) # max_len x batch_size x hidden_size  #(T,B,DH)
    print('encoder_hidden', encoder_hidden.size()) # n_layers * 2 x batch_size x hidden_size #(LD,B,H)

    max_target_length = max(target_lengths)

    # Prepare decoder input and outputs
    decoder_input = Variable(torch.LongTensor([SOS_token] * small_batch_size)) #(B)
    decoder_hidden = encoder_hidden[:decoder_test.n_layers] # Use last (forward) hidden state from encoder 
    # 所以从上面可以直到事实上 LD 维度的组合方式是先Layer后Direction， forward在前面， backward 在后面 (L,B,H)
    all_decoder_outputs = Variable(torch.zeros(max_target_length, small_batch_size, decoder_test.output_size))
    #(T_o, B,O)

    if USE_CUDA:
        all_decoder_outputs = all_decoder_outputs.cuda()
        decoder_input = decoder_input.cuda()

    # Run through decoder one time step at a time
    for t in range(max_target_length):
        decoder_output, decoder_hidden, decoder_attn = decoder_test(
            decoder_input, decoder_hidden, encoder_outputs
        ) #(B,O), (L,B,H), (B,1,H)
        all_decoder_outputs[t] = decoder_output # Store this step's outputs
        decoder_input = target_batches[t] # Next input is current target (B) teacher forcing

    # Test masked cross entropy loss
    loss = masked_cross_entropy(
        all_decoder_outputs.transpose(0, 1).contiguous(), #(B, T_o, O)
        target_batches.transpose(0, 1).contiguous(), #(B, T_o)
        target_lengths #(B)
    )
    print('loss', loss.data[0])
