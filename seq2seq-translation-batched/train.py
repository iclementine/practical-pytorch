import torch
from torch.autograd import Variable
from torch import nn, optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from masked_cross_entropy import masked_cross_entropy
# 其实使用 packedSequence 直接规避掉这个或许会更好， 先 pack 通过 RNN 然后再次 pad

from data import *
from model import *
from utils import *

from evaluate import *
import pdb

def train(input_batches, input_lengths, target_batches, target_lengths, encoder, decoder,encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    # 訓練一個 minibatch
    
    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0 # Added onto for each word

    # Run words through encoder
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)
    
    # Prepare input and output variables
    decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size))
    decoder_hidden = encoder_hidden[::2].contiguous()
    #decoder_hidden = encoder_hidden[:decoder.n_layers]
    #decoder_hidden = encoder_hidden[:decoder.n_layers] # Use last (forward) hidden state from encoder

    max_target_length = max(target_lengths)
    all_decoder_outputs = Variable(torch.zeros(max_target_length, batch_size, decoder.output_size))

    # Move new Variables to CUDA
    if USE_CUDA:
        decoder_input = decoder_input.cuda()
        all_decoder_outputs = all_decoder_outputs.cuda()

    # Run through decoder one time step at a time
    # actually, no teacher forcing is used.
    # the problem and #TODO is: valid lengths for cross_entropy, whether to use teacher forcing uniformly in a mini-batch, what todo when encounterd <EOS>, how to collect valid lengths of generation, to use target lengths or generation lengths
    for t in range(max_target_length):
        decoder_output, decoder_hidden, decoder_attn = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )

        all_decoder_outputs[t] = decoder_output
        decoder_input = target_batches[t] # Next input is current target

    # Loss calculation and backpropagation
    loss = masked_cross_entropy(
        all_decoder_outputs.transpose(0, 1).contiguous(), # -> batch x seq
        target_batches.transpose(0, 1).contiguous(), # -> batch x seq
        target_lengths
    )
    loss.backward()
    
    # Clip gradient norms
    ec = torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
    dc = torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)

    # Update parameters with optimizers
    encoder_optimizer.step() # 優化步驟已經在 train 函數裡面了
    decoder_optimizer.step()
    
    return loss.data[0], ec, dc # 注意， 返回的 ec, dc 已經是梯度了， 而不是 loss 然後在外面 backward, loss.data[0] 不是 Variable了， 所以只是用於顯示



# Configure models
attn_model = 'dot'
hidden_size = 500
n_layers = 2
dropout = 0.1
batch_size = 100
batch_size = 50

# Configure training/optimization
clip = 50.0
teacher_forcing_ratio = 0.5
learning_rate = 0.0001
decoder_learning_ratio = 5.0
n_epochs = 50000 #TODO: 到時候改回來
epoch = 0
plot_every = 20
print_every = 100
evaluate_every = 1000

# Initialize models
encoder = EncoderRNN(input_lang.n_words, hidden_size, n_layers, dropout=dropout)
decoder = LuongAttnDecoderRNN(attn_model, hidden_size, output_lang.n_words, n_layers, dropout=dropout)

# Initialize optimizers and criterion
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
criterion = nn.CrossEntropyLoss()

# Move models to GPU
if USE_CUDA:
    encoder.cuda()
    decoder.cuda()

#import sconce
# Sconce is a dashboard for monitoring and comparing data in real time. It was built to be an easy way to visualize the training progress of different machine learning models.

#job = sconce.Job('seq2seq-translate', {
    #'attn_model': attn_model,
    #'n_layers': n_layers,
    #'dropout': dropout,
    #'hidden_size': hidden_size,
    #'learning_rate': learning_rate,
    #'clip': clip,
    #'teacher_forcing_ratio': teacher_forcing_ratio,
    #'decoder_learning_ratio': decoder_learning_ratio,
#})

## Job 有兩個參數， 一個 name, 一個是 params 是一個字典
#job.plot_every = plot_every
#job.log_every = print_every

# Keep track of time elapsed and running averages
start = time.time()
plot_losses = []
print_loss_total = 0 # Reset every print_every
plot_loss_total = 0 # Reset every plot_every

# Begin!
print("Start training!\n")
ecs = []
dcs = []
eca = 0
dca = 0

while epoch < n_epochs:
    epoch += 1
    
    # Get training data for this cycle
    input_batches, input_lengths, target_batches, target_lengths = random_batch(batch_size)

    # Run the train function
    #pdb.set_trace()
    loss, ec, dc = train(
        input_batches, input_lengths, target_batches, target_lengths,
        encoder, decoder,
        encoder_optimizer, decoder_optimizer, criterion
    )
    
    # Keep track of loss
    print_loss_total += loss
    plot_loss_total += loss
    eca += ec # a stands for "accumulate"
    dca += dc
    
    #job.record(epoch, loss)

    if epoch % print_every == 0:
        print_loss_avg = print_loss_total / print_every
        print_loss_total = 0
        print_summary = '%s (%d %d%%) %.4f' % (time_since(start, epoch / n_epochs), epoch, epoch / n_epochs * 100, print_loss_avg)
        print(print_summary)
        
    if epoch % evaluate_every == 0:
        #evaluate_randomly(encoder, decoder)
        pass

    if epoch % plot_every == 0:
        plot_loss_avg = plot_loss_total / plot_every
        plot_losses.append(plot_loss_avg)
        plot_loss_total = 0
        
        # TODO: Running average helper
        ecs.append(eca / plot_every)
        dcs.append(dca / plot_every)
        #ecs_win = 'encoder grad (%s)' % hostname
        #dcs_win = 'decoder grad (%s)' % hostname
        #vis.line(np.array(ecs), win=ecs_win, opts={'title': ecs_win}) # disable plot temporally
        #vis.line(np.array(dcs), win=dcs_win, opts={'title': dcs_win})
        eca = 0
        dca = 0

print("Training finished! \n")

evaluate_randomly(encoder, decoder)
#evaluate_and_show_attention(pairs[30][0], encoder, decoder)
#evaluate_and_show_attention("elle est trop petit .", encoder, decoder)
#evaluate_and_show_attention("je ne crains pas de mourir .", encoder, decoder)
#evaluate_and_show_attention("c est un jeune directeur plein de talent .", encoder, decoder)
#evaluate_and_show_attention("est le chien vert aujourd hui ?", encoder, decoder)
#evaluate_and_show_attention("le chat me parle .", encoder, decoder)
#evaluate_and_show_attention("des centaines de personnes furent arretees ici .", encoder, decoder)
#evaluate_and_show_attention("des centaines de chiens furent arretees ici .", encoder, decoder)
#evaluate_and_show_attention("ce fromage est prepare a partir de lait de chevre .", encoder, decoder)
    


