import torch
import sys
from torch import nn
import torch.nn.functional as F
from modules.transformer import TransformerEncoder


class MULTModel(nn.Module):
    def __init__(self, orig_d_l, orig_d_a, orig_d_v, orig_d_f, MULT_d):
        """
        Construct a MulT model.
        """
        super(MULTModel, self).__init__()
        self.orig_d_l, self.orig_d_a, self.orig_d_v, self.orig_d_f = orig_d_l, orig_d_a, orig_d_v, orig_d_f
        self.d_l, self.d_a, self.d_v, self.d_f = MULT_d, MULT_d, MULT_d, MULT_d
        self.num_heads = 8
        self.layers = 6
        self.attn_dropout = 0.1
        self.attn_dropout_a = 0.1
        self.attn_dropout_v = 0.1
        self.attn_dropout_f = 0.1
        self.relu_dropout = 0.1
        self.res_dropout = 0.1
        self.out_dropout = 0.1
        self.embed_dropout = 0.25
        self.attn_mask = True

        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)
        self.proj_f = nn.Conv1d(self.orig_d_f, self.d_f, kernel_size=1, padding=0, bias=False)

        # 2. Crossmodal Attentions
        self.trans_l_with_a = self.get_network(self_type='la')
        self.trans_l_with_v = self.get_network(self_type='lv')
        self.trans_l_with_f = self.get_network(self_type='lf')

        self.trans_a_with_l = self.get_network(self_type='al')
        self.trans_a_with_v = self.get_network(self_type='av')
        self.trans_a_with_f = self.get_network(self_type='af')

        self.trans_v_with_l = self.get_network(self_type='vl')
        self.trans_v_with_a = self.get_network(self_type='va')
        self.trans_v_with_f = self.get_network(self_type='vf')

        self.trans_f_with_l = self.get_network(self_type='fl')
        self.trans_f_with_a = self.get_network(self_type='fa')
        self.trans_f_with_v = self.get_network(self_type='fv')

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl', 'fl']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va', 'fa']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av', 'fv', 'rv']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type in ['f', 'lf', 'af', 'vf']:
            embed_dim, attn_dropout = self.d_f, self.attn_dropout_f
        
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask,
                                  position_emb = True)
            
    def forward(self, x_l, x_a, x_v, x_f):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        """
        #x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
        x_l = x_l.transpose(1, 2)  # [batch_size, n_features, seq_len]
        x_a = x_a.transpose(1, 2)
        x_v = x_v.transpose(1, 2)
        x_f = x_f.transpose(1, 2)
       
        # Project the textual/visual/audio features
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        proj_x_f = x_f if self.orig_d_f == self.d_f else self.proj_f(x_f)

        proj_x_a = proj_x_a.permute(2, 0, 1)   # [ seq_len, batch_size, n_features]
        proj_x_v = proj_x_v.permute(2, 0, 1)
        proj_x_l = proj_x_l.permute(2, 0, 1)
        proj_x_f = proj_x_f.permute(2, 0, 1)

        h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a) # [ seq_len, batch_size, n_features]
        h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)  # [ seq_len, batch_size, n_features]
        h_l_with_fs = self.trans_l_with_f(proj_x_l, proj_x_f, proj_x_f)  # [ seq_len, batch_size, n_features]

        h_ls = F.dropout(torch.cat([h_l_with_as, h_l_with_vs, h_l_with_fs], dim=2), p=self.out_dropout, training=self.training)
        #h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)

        # (L,V) --> A
        h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
        h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)
        h_a_with_fs = self.trans_a_with_f(proj_x_a, proj_x_f, proj_x_f)

        h_as = F.dropout(torch.cat([h_a_with_ls, h_a_with_vs, h_a_with_fs], dim=2), p=self.out_dropout, training=self.training)
        #h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)

        # (L,A) --> V
        h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
        h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)
        h_v_with_fs = self.trans_v_with_f(proj_x_v, proj_x_f, proj_x_f)

        h_vs = F.dropout(torch.cat([h_v_with_ls, h_v_with_as, h_v_with_fs], dim=2), p=self.out_dropout, training=self.training)
        #h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=2)

        h_f_with_ls = self.trans_f_with_l(proj_x_f, proj_x_l, proj_x_l)
        h_f_with_as = self.trans_f_with_a(proj_x_f, proj_x_a, proj_x_a)
        h_f_with_vs = self.trans_f_with_v(proj_x_f, proj_x_v, proj_x_v)

        h_fs = F.dropout(torch.cat([h_f_with_ls, h_f_with_as, h_f_with_vs], dim=2), p=self.out_dropout,training=self.training)

        return h_ls, h_as, h_vs, h_fs

