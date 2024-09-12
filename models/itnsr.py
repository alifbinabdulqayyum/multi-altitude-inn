# Code adapted from "https://github.com/codyshen0000/ITSRN/blob/main/code/models/ITNSR.py"

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

################


class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G  = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize-1)//2, stride=1),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)

class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G  = growRate
        C  = nConvLayers

        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c*G, G))
        self.convs = nn.Sequential(*convs)

        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C*G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x

class RDN(nn.Module):
    def __init__(self):
        super(RDN, self).__init__()
        G0 = 64
        kSize = 3

        # number of RDB blocks, conv layers, out channels
        self.D, C, G = {
            'A': (20, 6, 32),
            'B': (16, 8, 64),
        }['B']

        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d(1, G0, kSize, padding=(kSize-1)//2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)

        # Redidual dense blocks and dense feature fusion
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(
                RDB(growRate0 = G0, growRate = G, nConvLayers = C)
            )

        # Global Feature Fusion
        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1),
            nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)
        ])
        
        self.out_dim = G0
        

    def forward(self, x):
        f__1 = self.SFENet1(x)
        x  = self.SFENet2(f__1)

        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)

        x = self.GFF(torch.cat(RDBs_out,1))
        x += f__1

        return x

################
class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_list, act='sine'):
        super().__init__()
        # pdb.set_trace()
        self.act = nn.GELU()
        
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            if self.act:
                layers.append(self.act)
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # pdb.set_trace()
        shape = x.shape[:-1]
        x = self.layers(x.contiguous().view(-1, x.shape[-1]))
        return x.view(*shape, -1)
################

def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
        coord_x = -1+(2*i+1)/W
        coord_y = -1+(2*i+1)/H
        normalize to (-1, 1)
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs, indexing='ij'), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

class ITNSR(nn.Module):

    def __init__(self, 
                 local_ensemble=True, 
                 feat_unfold=True, 
                 scale_token=True):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.scale_token = scale_token

        self.encoder = RDN()
        self.imnet = MLP(in_dim=4, out_dim=self.encoder.out_dim*9, hidden_list=[256,256,256,256])
        
        # if embedding_coord is not None:
        #     self.embedding_q = models.make(embedding_coord)
        #     self.embedding_s = models.make(embedding_scale)
        # else:
        #     self.embedding_q = None
        #     self.embedding_s = None

        self.embedding_q = None
        self.embedding_s = None

        if local_ensemble:
            # w = {
            #     'name': 'mlp',
            #     'args': {
            #         'in_dim': 4,
            #         'out_dim': 1,
            #         'hidden_list': [256],
            #         'act': 'gelu'
            #     }
            # }
            # self.Weight = models.make(w)
            self.Weight = MLP(in_dim=4, out_dim=1, hidden_list=[256])

            # score = {
            #     'name': 'mlp',
            #     'args': {
            #         'in_dim': 2,
            #         'out_dim': 1,
            #         'hidden_list': [256],
            #         'act': 'gelu'
            #     }
            # }
            # self.Score = models.make(score)
            self.Score = MLP(in_dim=2, out_dim=1, hidden_list=[256])
 

    def gen_feat(self, inp):
        self.feat = self.encoder(inp)
        return self.feat

    def query_rgb(self, coord, scale=None):

        feat = self.feat

        if self.imnet is None:
            ret = F.grid_sample(feat, coord.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            return ret

        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])
        
        # K
        feat_coord = make_coord(feat.shape[-2:], flatten=False).to(feat.device)  \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        # enhance local features
        if self.local_ensemble:
            # v_lst = [(-1,-1),(-1,0),(-1,1),(0, -1), (0, 0), (0, 1),(1, -1),(1, 0),(1,1)]#
            v_lst = [(i,j) for i in range(-1, 2, 2) for j in range(-1, 2, 2)]
            # v_lst = [(-1,0), (1,0), (0,1), (0, -1), (-2,-2), (-2, 2), (2, -2), (2, 2)]
            eps_shift = 1e-6
            preds = []
            for v in v_lst:
                vx = v[0]
                vy = v[1]
                # project to LR field 
                tx = ((feat.shape[-2] - 1) / (1 - scale[:,0,0])).view(feat.shape[0],  1)
                ty = ((feat.shape[-1] - 1) / (1 - scale[:,0,1])).view(feat.shape[0],  1)
                rx = (2*abs(vx) -1) / tx if vx != 0 else 0
                ry = (2*abs(vy) -1) / ty if vy != 0 else 0
                bs, q = coord.shape[:2]
                coord_ = coord.clone()

                if vx != 0:
                    coord_[:, :, 0] += vx /abs(vx) * rx + eps_shift
                if vy != 0:
                    coord_[:, :, 1] += vy /abs(vy) * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                #Interpolate K to HR resolution  
                value = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                #Interpolate K to HR resolution 
                coord_k = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                #calculate relation of Q-K
                if self.embedding_q:
                    Q = self.embedding_q(coord.contiguous().view(bs * q, -1))
                    K = self.embedding_q(coord_k.contiguous().view(bs * q, -1))
                    rel = Q - K
                    
                    rel[:, 0] *= feat.shape[-2]
                    rel[:, 1] *= feat.shape[-1]
                    inp = rel
                    if self.scale_token:
                        scale_ = scale.clone()
                        scale_[:, :, 0] *= feat.shape[-2]
                        scale_[:, :, 1] *= feat.shape[-1]
                        # scale = scale.view(bs*q,-1)
                        scale_ = self.embedding_s(scale_.contiguous().view(bs * q, -1))
                        inp = torch.cat([inp, scale_], dim=-1)

                else:
                    Q, K = coord, coord_k
                    rel = Q - K
                    rel[:, :, 0] *= feat.shape[-2]
                    rel[:, :, 1] *= feat.shape[-1]
                    inp = rel
                    if self.scale_token:
                        scale_ = scale.clone()
                        scale_[:, :, 0] *= feat.shape[-2]
                        scale_[:, :, 1] *= feat.shape[-1]
                        inp = torch.cat([inp, scale_], dim=-1)

                score = self.Score(rel.view(bs * q, -1)).view(bs, q, -1)
                
                weight = self.imnet(inp.view(bs * q, -1)).view(bs * q, feat.shape[1], -1)
                pred = torch.bmm(value.contiguous().view(bs * q, 1, -1), weight).view(bs, q, -1)
                
                pred +=score
                preds.append(pred)

            preds = torch.stack(preds,dim=-1)

            ret = self.Weight(preds.view(bs*q, -1)).view(bs, q, -1)
        else:
            #V
            bs, q = coord.shape[:2]
            value = F.grid_sample(
                feat, coord.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            #K
            coord_k = F.grid_sample(
                feat_coord, coord.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)

            if self.embedding_q:
                Q = self.embedding_q(coord.contiguous().view(bs * q, -1))
                K = self.embedding_q(coord_k.contiguous().view(bs * q, -1))
                rel = Q - K
                
                rel[:, 0] *= feat.shape[-2]
                rel[:, 1] *= feat.shape[-1]
                inp = rel
                if self.scale_token:
                    scale_ = scale.clone()
                    scale_[:, :, 0] *= feat.shape[-2]
                    scale_[:, :, 1] *= feat.shape[-1]
                    # scale = scale.view(bs*q,-1)
                    scale_ = self.embedding_s(scale_.contiguous().view(bs * q, -1))
                    inp = torch.cat([inp, scale_], dim=-1)

            else:
                Q, K = coord, coord_k
                rel = Q - K
                rel[:, :, 0] *= feat.shape[-2]
                rel[:, :, 1] *= feat.shape[-1]
                inp = rel
                if self.scale_token:
                    scale_ = scale.clone()
                    scale_[:, :, 0] *= feat.shape[-2]
                    scale_[:, :, 1] *= feat.shape[-1]
                    inp = torch.cat([inp, scale_], dim=-1)
            
            
            weight = self.imnet(inp.view(bs * q, -1)).view(bs * q, feat.shape[1], 3)
            pred = torch.bmm(value.contiguous().view(bs * q, 1, -1), weight).view(bs, q, -1)
            ret = pred
        
        return ret

    def forward(self, inp, coord, scale):

        self.gen_feat(inp)
        return self.query_rgb(coord, scale)




