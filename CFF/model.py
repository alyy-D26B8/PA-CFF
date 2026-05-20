# Copyright 2020 Google LLC
#
# Modified by ally, 2026.
# This file is modified from the official IBRNet implementation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import os
from ibrnet.mlp_network import IBRNet
from ibrnet.feature_network import ResUNet
from ibrnet.cff_module import CFFModule


def de_parallel(model):
    return model.module if hasattr(model, 'module') else model

########################################################################################################################
# creation/saving/loading of nerf
########################################################################################################################


class IBRNetModel(object):
    def __init__(self, args, load_opt=True, load_scheduler=True):
        self.args = args
        device = torch.device('cuda:{}'.format(args.local_rank))

        # ===================== networks =====================
        # create coarse IBRNet
        self.net_coarse = IBRNet(args,
                                 in_feat_ch=self.args.coarse_feat_dim,
                                 n_samples=self.args.N_samples).to(device)
        if args.coarse_only:
            self.net_fine = None
        else:
            self.net_fine = IBRNet(args,
                                   in_feat_ch=self.args.fine_feat_dim,
                                   n_samples=self.args.N_samples+self.args.N_importance).to(device)

        # create feature extraction network
        self.feature_net = ResUNet(coarse_out_ch=self.args.coarse_feat_dim,
                                   fine_out_ch=self.args.fine_feat_dim,
                                   coarse_only=self.args.coarse_only).cuda()

        # ===================== CFF module =====================
        use_cff = getattr(args, 'use_cff', False)
        if use_cff:
            self.cff_module = CFFModule(
                resolutions=getattr(args, 'cff_resolutions', [32, 64]),
                feat_dim=getattr(args, 'cff_feat_dim', 8),
                out_dim=getattr(args, 'cff_out_dim', 16),
            ).to(device)
            print('[CFF] Enabled — resolutions={}, feat_dim={}, out_dim={}, bbox={}, params={:,}'.format(
                args.cff_resolutions, args.cff_feat_dim, args.cff_out_dim, args.cff_bbox_size,
                sum(p.numel() for p in self.cff_module.parameters())))
        else:
            self.cff_module = None

        # ===================== freeze options =====================
        freeze_ibrnet = getattr(args, 'freeze_ibrnet', False)
        freeze_feature_net = getattr(args, 'freeze_feature_net', False)

        if freeze_ibrnet:
            print('[Freeze] IBRNet MLP (net_coarse & net_fine) frozen.')
            for p in self.net_coarse.parameters():
                p.requires_grad = False
            if self.net_fine is not None:
                for p in self.net_fine.parameters():
                    p.requires_grad = False

        if freeze_feature_net:
            print('[Freeze] Feature extractor (ResUNet) frozen.')
            for p in self.feature_net.parameters():
                p.requires_grad = False

        # ===================== optimizer =====================
        param_groups = []

        if not freeze_ibrnet:
            param_groups.append({'params': self.net_coarse.parameters()})
            if self.net_fine is not None:
                param_groups.append({'params': self.net_fine.parameters()})

        if not freeze_feature_net:
            param_groups.append({'params': self.feature_net.parameters(),
                                 'lr': args.lrate_feature})

        if self.cff_module is not None:
            lrate_cff = getattr(args, 'lrate_cff', 1e-3)
            param_groups.append({'params': self.cff_module.parameters(),
                                 'lr': lrate_cff})

        assert len(param_groups) > 0, \
            'All modules are frozen and no CFF — nothing to optimise!'

        self.optimizer = torch.optim.Adam(param_groups, lr=args.lrate_mlp)

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=args.lrate_decay_steps,
                                                         gamma=args.lrate_decay_factor)

        # ===================== load checkpoint =====================
        out_folder = os.path.join(args.rootdir, 'out', args.expname)
        self.start_step = self.load_from_ckpt(out_folder,
                                              load_opt=load_opt,
                                              load_scheduler=load_scheduler)

        if args.distributed:
            self.net_coarse = torch.nn.parallel.DistributedDataParallel(
                self.net_coarse,
                device_ids=[args.local_rank],
                output_device=args.local_rank
            )

            self.feature_net = torch.nn.parallel.DistributedDataParallel(
                self.feature_net,
                device_ids=[args.local_rank],
                output_device=args.local_rank
            )

            if self.net_fine is not None:
                self.net_fine = torch.nn.parallel.DistributedDataParallel(
                    self.net_fine,
                    device_ids=[args.local_rank],
                    output_device=args.local_rank
                )

    def switch_to_eval(self):
        self.net_coarse.eval()
        self.feature_net.eval()
        if self.net_fine is not None:
            self.net_fine.eval()
        if self.cff_module is not None:
            self.cff_module.eval()

    def switch_to_train(self):
        self.net_coarse.train()
        self.feature_net.train()
        if self.net_fine is not None:
            self.net_fine.train()
        if self.cff_module is not None:
            self.cff_module.train()

    def save_model(self, filename):
        to_save = {'optimizer': self.optimizer.state_dict(),
                   'scheduler': self.scheduler.state_dict(),
                   'net_coarse': de_parallel(self.net_coarse).state_dict(),
                   'feature_net': de_parallel(self.feature_net).state_dict()
                   }

        if self.net_fine is not None:
            to_save['net_fine'] = de_parallel(self.net_fine).state_dict()

        if self.cff_module is not None:
            to_save['cff_module'] = self.cff_module.state_dict()

        torch.save(to_save, filename)

    def load_model(self, filename, load_opt=True, load_scheduler=True):
        if self.args.distributed:
            to_load = torch.load(filename, map_location='cuda:{}'.format(self.args.local_rank))
        else:
            to_load = torch.load(filename)

        if load_opt:
            try:
                self.optimizer.load_state_dict(to_load['optimizer'])
            except (ValueError, KeyError):
                print('[Warning] Optimizer state mismatch — skipping optimizer load.')

        if load_scheduler:
            try:
                self.scheduler.load_state_dict(to_load['scheduler'])
            except (ValueError, KeyError):
                print('[Warning] Scheduler state mismatch — skipping scheduler load.')

        self.net_coarse.load_state_dict(to_load['net_coarse'])
        self.feature_net.load_state_dict(to_load['feature_net'])

        if self.net_fine is not None and 'net_fine' in to_load.keys():
            self.net_fine.load_state_dict(to_load['net_fine'])

        # CFF: load only when both model and checkpoint have it
        if self.cff_module is not None and 'cff_module' in to_load:
            self.cff_module.load_state_dict(to_load['cff_module'])
            print('[CFF] Loaded CFF weights from checkpoint.')
        elif self.cff_module is not None:
            print('[CFF] No CFF weights in checkpoint — using fresh initialisation.')

    def load_from_ckpt(self, out_folder,
                       load_opt=True,
                       load_scheduler=True,
                       force_latest_ckpt=False):
        '''
        load model from existing checkpoints and return the current step
        :param out_folder: the directory that stores ckpts
        :return: the current starting step
        '''

        # all existing ckpts
        ckpts = []
        if os.path.exists(out_folder):
            ckpts = [os.path.join(out_folder, f)
                     for f in sorted(os.listdir(out_folder)) if f.endswith('.pth')]

        if self.args.ckpt_path is not None and not force_latest_ckpt:
            if os.path.isfile(self.args.ckpt_path):  # load the specified ckpt
                ckpts = [self.args.ckpt_path]

        if len(ckpts) > 0 and not self.args.no_reload:
            fpath = ckpts[-1]
            self.load_model(fpath, load_opt, load_scheduler)
            step = int(fpath[-10:-4])
            print('Reloading from {}, starting at step={}'.format(fpath, step))
        else:
            print('No ckpts found, training from scratch...')
            step = 0

        return step

