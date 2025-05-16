import math
import torch
import torch.nn as nn


import math
import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_heads,
        dropout=0.1
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.mm_am_qproj = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.mm_am_kproj = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.mm_am_vproj = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.out_proj = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        hidden_states,
        context_states=None,
        attention_mask=None
    ):
        if context_states is None:
            context_states = hidden_states
        
        
        # print("Hidden, Context ", hidden_states.shape, context_states.shape)
        Q = self.mm_am_qproj(hidden_states)
        K = self.mm_am_kproj(context_states)
        V = self.mm_am_vproj(context_states)
        
        
        _, q_len, _ = Q.size()
        # if K.dim() == 4:
        #     B, _, k_len, _ = K.size()
        # else:
        B, k_len, _ = K.size()
        # print("Before", Q.shape, K.shape, V.shape)
        Q = Q.view(1, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, k_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, k_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # print("After", Q.shape, K.shape, V.shape)
        
        attn_scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        
        attn_weights = torch.softmax(attn_scores, dim=-1)
        # print("attn_weights", attn_weights.shape)
        attn_output = torch.matmul(attn_weights, V)
        # print("attn_output", attn_output.shape)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(B, q_len, self.hidden_dim)
        # attn_weights = attn_weights.transpose(1, 2).contiguous()
        # attn_weights = attn_weights.view(B, q_len, self.hidden_dim)
        attn_output = self.out_proj(attn_output)
        
        return attn_output, attn_weights


class FeedForwardLayer(nn.Module):
    def __init__(
        self, 
        hidden_dim, 
        intermediate_dim, 
        dropout=0.1
    ):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, intermediate_dim)
        self.fc2 = nn.Linear(intermediate_dim, hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x        


class MMAllingmentEncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_heads,
        dropout=0.1
    ):
        super().__init__()
        self.q_head = MultiHeadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.v_head = MultiHeadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.a_head = MultiHeadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.v_attn_ln = nn.LayerNorm(normalized_shape=hidden_dim)
        self.a_attn_ln = nn.LayerNorm(normalized_shape=hidden_dim)
        self.q_attn_ln = nn.LayerNorm(normalized_shape=hidden_dim)
        self.v_ff = FeedForwardLayer(hidden_dim=hidden_dim, intermediate_dim=hidden_dim, dropout=dropout)
        self.a_ff = FeedForwardLayer(hidden_dim=hidden_dim, intermediate_dim=hidden_dim, dropout=dropout)
        self.q_ff = FeedForwardLayer(hidden_dim=hidden_dim, intermediate_dim=hidden_dim, dropout=dropout)
        self.activation = nn.GELU()
    
    def forward(
        self, query_tokens, video_feat, audio_feat
    ):
        query_tokens_start = query_tokens
        # video_feat, v_attn = self.v_head(hidden_states=query_tokens, context_states=video_feat)
        # audio_feat, a_attn = self.a_head(hidden_states=query_tokens, context_states=audio_feat)
        
        query_tokens_, _ = self.q_head(hidden_states=query_tokens)
        query_tokens = query_tokens_ + query_tokens
        query_tokens = self.q_attn_ln(query_tokens)
        query_tokens_ = self.q_ff(query_tokens)
        query_tokens = query_tokens + query_tokens_
        query_tokens = self.q_attn_ln(query_tokens)
        
        video_feat_, v_attn = self.v_head(hidden_states=query_tokens_start, context_states=video_feat)
        video_feat = video_feat_ + query_tokens_start
        video_feat_ = self.v_attn_ln(video_feat)
        video_feat = self.v_ff(video_feat_)
        video_feat = video_feat + video_feat_
        video_feat = self.v_attn_ln(video_feat)
        
        audio_feat_, a_attn = self.a_head(hidden_states=query_tokens_start, context_states=audio_feat)
        audio_feat = audio_feat_ + query_tokens_start
        audio_feat_ = self.a_attn_ln(audio_feat)
        audio_feat = self.a_ff(audio_feat_)
        audio_feat = audio_feat + audio_feat_
        audio_feat = self.a_attn_ln(audio_feat)
        
        return video_feat, audio_feat, v_attn, a_attn


class MMAllingmentModule(nn.Module):
    def __init__(
        self,
        hidden_dim=3584,
        num_heads=8,
        num_layers=2,
        num_query_tokens=2048,
        dropout=0.1
    ):
        super().__init__()
        
        self.query_tokens = nn.Parameter(
            torch.zeros(1, num_query_tokens, hidden_dim)
        )
        nn.init.trunc_normal_(self.query_tokens, std=0.02)
        # self.query_position_embeddings = nn.Parameter(torch.zeros(num_query_tokens, hidden_dim))
        # nn.init.trunc_normal_(self.query_position_embeddings, std=0.02)
        self.mm_alling_layers = nn.ModuleList(
            MMAllingmentEncoderLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout
                ) for _ in range(num_layers)
        )
        self.mm_am_vproj = FeedForwardLayer(hidden_dim=hidden_dim, intermediate_dim=hidden_dim)
        self.mm_am_aproj = FeedForwardLayer(hidden_dim=hidden_dim, intermediate_dim=hidden_dim)
    
    def forward(self, video_feat, audio_feat, v_attn=None, a_attn=None):
        B = video_feat.size(0)
        # query_embeds = self.query_tokens.unsqueeze(0).expand(B, -1, -1, -1)
        # pos_embeds = self.query_position_embeddings.unsqueeze(0).expand(B, -1, -1, -1)
        # query_embeds = query_embeds + pos_embeds
        # video_feat = video_feat + pos_embeds
        # audio_feat = audio_feat + pos_embeds
        for i, mm_alling_layer in enumerate(self.mm_alling_layers):
            video_feat, audio_feat, v_attn, a_attn = mm_alling_layer(
                self.query_tokens, video_feat, audio_feat
            )
        video_feat = self.mm_am_vproj(video_feat)
        audio_feat = self.mm_am_aproj(video_feat)
        return video_feat, audio_feat, v_attn, a_attn
    

class AllingmentModule(nn.Module):
    def __init__(
        self,
        hidden_dim=3584,
        num_heads=32,
        mm_feat_dim=2848,  # 2848 for AV 2968 for AVS
        num_layers=2,
        dropout=0.1
    ):
        super(AllingmentModule, self).__init__()
        self.mm_feat_dim = mm_feat_dim
        self.multi_head_attention = MultiHeadAttention(
            hidden_dim, num_heads
        )
        self.fc1 = nn.Linear(hidden_dim, mm_feat_dim)
        self.fc2 = nn.Linear(mm_feat_dim, mm_feat_dim)
        
        self.ln = nn.LayerNorm(normalized_shape=mm_feat_dim)
        
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def get_embed(self, new_input_embeds):
        text_embed = []
        mm_feat = []
        for feat in new_input_embeds:
            feat = feat.unsqueeze(0)
            total_len = feat.shape[1]
            text_embed_len = total_len - self.mm_feat_dim
            text_embed.append(feat[:, :text_embed_len,:])
            mm_feat.append(feat[:, text_embed_len:, :])
        max_len = max(x.shape[1] for x in text_embed)
        new_text_embed = []
        for t_embed in text_embed:
            text_new_embed = torch.cat(
                (
                    t_embed, 
                    torch.zeros((1, max_len - t_embed.shape[1], t_embed.shape[2]), dtype=t_embed.dtype, device=t_embed.device)),
                    dim=1
                )
            new_text_embed.append(text_new_embed)
        text_embed = torch.cat(new_text_embed, 0)
        mm_feat = torch.cat(mm_feat, 0)
        return text_embed, mm_feat
    
    def weight_token(self, token_weights, non_weighted_tokens, inference=False):
        weighted_new_input_embeds = []
        for weights, feat in zip(token_weights, non_weighted_tokens):
            total_len = feat.shape[0]
            text_embed_len = total_len - self.mm_feat_dim
            # if not inference:
            scaled_feat = torch.mul(weights, feat[text_embed_len:, :].transpose(1, 0)).transpose(1, 0)
            # else:
            #     mask = weights > 0.5
            #     scaled_feat = feat[text_embed_len:, :][mask]
            new_feat = torch.cat([feat[:text_embed_len, :], scaled_feat], dim=0)
            weighted_new_input_embeds.append(new_feat)
        weighted_new_input_embeds = torch.stack(weighted_new_input_embeds, dim=0)
        return weighted_new_input_embeds
    
    def forward(self, new_input_embeds, inference=False):
        text_embed, mm_feat = self.get_embed(new_input_embeds)
        x, _ = self.multi_head_attention(text_embed, mm_feat)
        x = torch.mean(x, 1)
        x = self.fc1(x)
        x = self.ln(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        new_input_embeds = self.weight_token(
            x, new_input_embeds, inference
        )
        return new_input_embeds



def build_mm_allingment_module():
    hidden_dim = 3584
    num_heads = 8
    num_layers = 2
    num_query_tokens = 32
    dropout = 0.1
    # mm_allingment_module = MMAllingmentModule(
    #     hidden_dim=hidden_dim,
    #     num_heads=num_heads,
    #     num_layers=num_layers,
    #     num_query_tokens=num_query_tokens,
    #     dropout=dropout
    # )
    mm_allingment_module = AllingmentModule()
    return mm_allingment_module