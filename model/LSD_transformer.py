import torch
import numpy as np
import torch.nn as nn
import math
import torch.nn.functional as F

def init_method(tensor, **kwargs):
    nn.init.kaiming_uniform_(tensor, a=math.sqrt(5))

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=1, repeats=n_rep)"""
    bs, n_kv_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )
import math
import random

# 固定随机种子
random.seed(42)  # 设置一个固定种子值

# 自动生成满足条件的 x2，并固定 x1 = 0.8
def set_parameters():
    x1 = random.uniform(0, 1)  # 固定 x1 为 0.8
    while True:
        x2 = random.uniform(0, 1)  # 从 [0, 1] 中随机生成 x2
        if x1 > x2 and x1 - x2 > 0.2:  # 确保满足条件
            return x1, x2

# 初始化参数
x1, x2 = set_parameters()

# 修改后的 lambda 初始化函数
def lambda_init_fn(depth):
    return x1 - x2 * math.exp(-0.3 * depth)

class avg(nn.Module):


    def __init__(self, kernel_size, stride):
        super(avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=(kernel_size // 2))

    def forward(self, x):

        front = x[:, 0:1, :].repeat(1, self.kernel_size // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class attention(nn.Module):
    def __init__(self, mask_flag=True, factor=1, scale=None, attention_dropout=0.2, output_attention=False,depth=2048):
        super(attention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

        self.l_q1 = nn.Parameter(torch.zeros(factor).normal_(mean=0, std=0.1))
        self.l_k1 = nn.Parameter(torch.zeros(factor).normal_(mean=0, std=0.1))
        self.l_q2 = nn.Parameter(torch.zeros(factor).normal_(mean=0, std=0.1))
        self.l_k2 = nn.Parameter(torch.zeros(factor).normal_(mean=0, std=0.1))
        self.l_init = lambda_init_fn(depth=2048)


    def time_delay_agg_training(self, values, corr):
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]
        weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)
        tmp_corr = torch.softmax(weights, dim=-1)
        tmp_values = values
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            pattern = torch.roll(tmp_values, -int(index[i]), -1)
            delays_agg = delays_agg + pattern * (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        return delays_agg

    def time_delay_agg_inference(self, values, corr):
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0)\
            .repeat(batch, head, channel, 1).to(values.device)
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        weights, delay = torch.topk(mean_value, top_k, dim=-1)
        tmp_corr = torch.softmax(weights, dim=-1)
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        return delays_agg

    def forward(self, Q, K, V, attn_mask=None):
        B, H, L, E = Q.shape
        _, _, S, D = V.shape
        if L > S:
            zeros = torch.zeros_like(Q[:, :, :(L - S), :]).float()
            V = torch.cat([V, zeros], dim=2)
            K = torch.cat([K, zeros], dim=2)
        else:
            V = V[:, :, :L, :]
            K = K[:, :, :L, :]

        # 首先对于q，k进行FFT变换
        q_fft = torch.fft.rfft(Q.permute(0, 1, 3, 2).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(K.permute(0, 1, 3, 2).contiguous(), dim=-1)
        # 对k进行求共轭，并与q点乘
        res = q_fft * torch.conj(k_fft)
        # 傅里叶反变换计算得出的值就是每一个时移因子的相似性大小
        corr = torch.fft.irfft(res, n=L, dim=-1)
        corr = self.dropout(corr)
        # 应用 l调整机制
        l1 = torch.exp(torch.sum(self.l_q1 * self.l_k1, dim=-1).float()).type_as(corr)
        l2 = torch.exp(torch.sum(self.l_q2 * self.l_k2, dim=-1).float()).type_as(corr)
        l_full = l1 - l2 + self.l_init
        # 差分注意力计算
        attn_weights_1 = F.softmax(corr, dim=-1)  # softmax(Q1 K^T)
        attn_weights_2 = F.softmax(corr, dim=-1)  # softmax(Q2 K^T)
        # 差分计算
        diff_attn_weights = attn_weights_1 - l_full.unsqueeze(-1) * attn_weights_2


        if self.training:
            V = self.time_delay_agg_training(V.permute(0, 1, 3, 2).contiguous(), diff_attn_weights).permute(0, 1, 3, 2)
        else:
            V = self.time_delay_agg_inference(V.permute(0, 1, 3, 2).contiguous(), diff_attn_weights).permute(0, 1, 3, 2)

        if self.output_attention:
            return V.contiguous(), corr.permute(0, 3, 1, 2)
        else:
            return V.contiguous(), None

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads, len_q, len_k):
        super(MultiHeadAttention, self).__init__()

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)  # Linear only change the last dimension

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.ScaledDotProductAttention = attention(mask_flag=True, factor=1, scale=None, attention_dropout=0.1,
                                                         output_attention=False)
        self.len_q = len_q
        self.len_k = len_k

    def forward(self, input_Q, input_K, input_V):


        residual, batch_size = input_Q, input_Q.size(0)

        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # Q: [batch_size, n_heads, len_q, d_k]

        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # K: [batch_size, n_heads, len_k, d_k]

        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]


        context, attn = self.ScaledDotProductAttention(Q, K, V)
        context = context.transpose(1, 2).reshape(batch_size, -1,
                                                  self.n_heads * self.d_v)  # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)  # [batch_size, len_q, d_model]
        return nn.LayerNorm(self.d_model).cuda()(output + residual), attn  # All batch size dimensions are reserved.


class FFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FFN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
        self.d_model = d_model

    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(self.d_model).cuda()(output + residual)  # [batch_size, seq_len, d_model]


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, d_k, d_v, n_heads, len_q):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model, d_k, d_v, n_heads, 1, len_q)
        self.pos_ffn = FFN(d_model, d_ff)
        self.moving_avg =avg(kernel_size=6,stride=1)
        # self.conv = ConvModule( embed_dim = 2048,context_size = 5 )

    def forward(self, enc_inputs):


        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs)  # enc_inputs to same Q,K,V
        enc_outputs =self.moving_avg(enc_outputs)
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn


class Encoder(nn.Module):
    def __init__(self, d_model, d_ff, d_k, d_v, n_layers, n_heads, len_q):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_ff, d_k, d_v, n_heads, len_q) for _ in range(n_layers)])

    def forward(self, enc_inputs):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        '''
        enc_outputs = enc_inputs
        enc_self_attns = []
        for layer in self.layers:

            enc_outputs, enc_self_attn = layer(enc_outputs)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, d_k, d_v, n_heads, len_q):
        super(DecoderLayer, self).__init__()
        self.dec_enc_attn = MultiHeadAttention(d_model, d_k, d_v, n_heads, 1, len_q)
        self.pos_ffn = FFN(d_model, d_ff)
        self.moving_avg =avg(kernel_size=6,stride=1)

    def forward(self, dec_inputs, enc_outputs):

        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_inputs, enc_outputs, enc_outputs)
        dec_outputs =self.moving_avg(dec_outputs)
        dec_outputs = self.pos_ffn(dec_outputs)  # [batch_size, tgt_len, d_model]
        return dec_outputs, dec_enc_attn


class Decoder(nn.Module):
    def __init__(self, d_model, d_ff, d_k, d_v, n_layers, n_heads, len_q):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, d_ff, d_k, d_v, n_heads, len_q) for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_outputs):

        dec_outputs = dec_inputs  # self.tgt_emb(dec_inputs) # [batch_size, tgt_len, d_model]


        dec_enc_attns = []
        for layer in self.layers:

            dec_outputs, dec_enc_attn = layer(dec_outputs, enc_outputs)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs



class Transformer1(nn.Module):
    def __init__(self, d_model, d_ff, d_k, d_v, n_layers, n_heads, len_q):
        super(Transformer1, self).__init__()
        self.encoder = Encoder(d_model, d_ff, d_k, d_v, n_layers, n_heads, len_q).cuda()
        self.decoder = Decoder(d_model, d_ff, d_k, d_v, 1, n_heads, len_q).cuda()

    def forward(self, enc_inputs, dec_inputs):

        enc_outputs, enc_self_attns = self.encoder(enc_inputs)  # Self-attention for temporal features
        dec_outputs = self.decoder(dec_inputs, enc_outputs)
        return dec_outputs



class LSD_Transformer(nn.Module):
    def __init__(self, mstcn_f_maps, mstcn_f_dim, out_features, len_q, d_model=None):
        super(LSD_Transformer, self).__init__()
        self.num_f_maps = mstcn_f_maps  # 32
        self.dim = mstcn_f_dim  # 2048
        self.num_classes = out_features  # 7
        self.len_q = len_q
        self.d_model = 2048

        self.spatial_encoder = EncoderLayer(self.d_model, mstcn_f_maps, mstcn_f_maps, mstcn_f_maps, 8, 5)
        self.transformer = Transformer1(d_model=self.d_model, d_ff=mstcn_f_maps, d_k=mstcn_f_maps,
                                            d_v=mstcn_f_maps, n_layers=1, n_heads=8, len_q=len_q)
        self.fc = nn.Linear(mstcn_f_dim, 2048, bias=False)


        self.out = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(2048, out_features, bias=False)
        )


    def forward(self, x, long_feature):

        out_features = x.transpose(1, 2)
        bs = out_features.size(0)  # 1
        inputs = []
        for i in range(out_features.size(1)):
            if i<self.len_q-1:
                input0 = torch.zeros((bs, self.len_q-1-i, 2048)).cuda()
                input0 = torch.cat([input0, out_features[:, 0:i+1]], dim=1)
            else:
                input0 = out_features[:, i-self.len_q+1:i+1]  # Collect all previous features
            inputs.append(input0)
        inputs = torch.stack(inputs, dim=0).squeeze(1)

        feas = torch.tanh(self.fc(long_feature))  # .transpose(0, 1))  # Project the input to desired dimension
        out_feas = []
        spa_len = 10
        for i in range(feas.size(1)):
            if i < spa_len - 1:
                input0 = torch.zeros((bs, spa_len - 1 - i, 2048)).cuda()
                input0 = torch.cat([input0, feas[:, 0:i + 1]], dim=1)
            else:
                input0 = out_features[:, i - spa_len + 1:i + 1]  # Collect all previous features
            out_feas.append(input0)
        out_feas = torch.stack(out_feas, dim=0).squeeze(1)

        out_feas, _ = self.spatial_encoder(out_feas)
        output = self.transformer(inputs, out_feas)  # Feature fusion between  temporal and spatial features
        output = self.out(output)
        return output[:, -1]
