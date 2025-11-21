import torch.nn as nn
import torch.nn.functional as F
import torch


class CapsuleSequenceToGraph(nn.Module):
    def __init__(self, MULT_d, dim_capsule, vertex_num, routing,
                 T_t, T_a, T_v, T_f):
        super(CapsuleSequenceToGraph, self).__init__()
        self.d_c = dim_capsule
        self.n = vertex_num
        self.routing = routing
        # create primary capsule
        self.W_tpc = nn.Parameter(torch.Tensor(T_t, self.n, MULT_d, self.d_c))
        self.W_apc = nn.Parameter(torch.Tensor(T_a, self.n, MULT_d, self.d_c))
        self.W_vpc = nn.Parameter(torch.Tensor(T_v, self.n, MULT_d, self.d_c))
        self.W_fpc = nn.Parameter(torch.Tensor(T_f, self.n, MULT_d, self.d_c))
        nn.init.xavier_normal_(self.W_tpc)
        nn.init.xavier_normal_(self.W_apc)
        nn.init.xavier_normal_(self.W_vpc)
        nn.init.xavier_normal_(self.W_fpc)

    def forward(self, text, audio, video, frames):
        text = text.transpose(1, 0)
        audio = audio.transpose(1, 0)
        video = video.transpose(1, 0)
        frames = frames.transpose(1, 0)
        # get dimensionality
        T_t = text.shape[0]
        T_a = audio.shape[0]
        T_v = video.shape[0]
        T_f = frames.shape[0]

        # create primary capsule
        # 张量乘法，输入三维和四维张量，下标分别为tbj和tnjd，按t,j求和，再交换第一维和第二维
        text_pri_caps = (torch.einsum('tbj, tnjd->tbnd', text, self.W_tpc)).permute(1, 0, 2, 3)
        audio_pri_caps = (torch.einsum('tbj, tnjd->tbnd', audio, self.W_apc)).permute(1, 0, 2, 3)
        video_pri_caps = (torch.einsum('tbj, tnjd->tbnd', video, self.W_vpc)).permute(1, 0, 2, 3)
        frames_pri_caps = (torch.einsum('tbj, tnjd->tbnd', frames, self.W_fpc)).permute(1, 0, 2, 3)

        # routing mechanism does not participate in back propagation
        text_pri_caps_temp = text_pri_caps.detach()
        audio_pri_caps_temp = audio_pri_caps.detach()
        video_pri_caps_temp = video_pri_caps.detach()
        frames_pri_caps_temp = frames_pri_caps.detach()

        # begin routing
        for r in range(self.routing + 1):
            if r == 0:
                b_t = torch.zeros(text.shape[1], T_t, self.n).cuda()  # initialize routing coefficients
                b_a = torch.zeros(text.shape[1], T_a, self.n).cuda()
                b_v = torch.zeros(text.shape[1], T_v, self.n).cuda()
                b_f = torch.zeros(text.shape[1], T_f, self.n).cuda()
            rc_t = F.softmax(b_t, 2).cuda()
            rc_a = F.softmax(b_a, 2).cuda()
            rc_v = F.softmax(b_v, 2).cuda()
            rc_f = F.softmax(b_f, 2).cuda()

            text_vertex = torch.tanh(torch.sum(text_pri_caps_temp * rc_t.unsqueeze(-1), 1))
            audio_vertex = torch.tanh(torch.sum(audio_pri_caps_temp * rc_a.unsqueeze(-1), 1))
            video_vertex = torch.tanh(torch.sum(video_pri_caps_temp * rc_v.unsqueeze(-1), 1))
            frames_vertex = torch.tanh(torch.sum(frames_pri_caps_temp * rc_f.unsqueeze(-1), 1))

            # update routing coefficients
            if r < self.routing:
                last = b_t
                new = ((text_vertex.unsqueeze(1)) * text_pri_caps_temp).sum(3)
                b_t = last + new

                last = b_a
                new = (audio_vertex.unsqueeze(1) * audio_pri_caps_temp).sum(3)
                b_a = last + new

                last = b_v
                new = (video_vertex.unsqueeze(1) * video_pri_caps_temp).sum(3)
                b_v = last + new

                last = b_f
                new = (frames_vertex.unsqueeze(1) * frames_pri_caps_temp).sum(3)
                b_f = last + new

        # create vertex using the routing coefficients in final round
        text_vertex = torch.tanh(torch.sum(text_pri_caps * rc_t.unsqueeze(-1), 1))
        audio_vertex = torch.tanh(torch.sum(audio_pri_caps * rc_a.unsqueeze(-1), 1))
        video_vertex = torch.tanh(torch.sum(video_pri_caps * rc_v.unsqueeze(-1), 1))
        frames_vertex = torch.tanh(torch.sum(frames_pri_caps * rc_f.unsqueeze(-1), 1))
        return text_vertex, audio_vertex, video_vertex, frames_vertex