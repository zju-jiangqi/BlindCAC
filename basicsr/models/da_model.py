from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
import torch.nn.functional as F
import torch
import torchvision.utils as tvu
from basicsr.metrics import calculate_metric
from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.utils import get_root_logger, imwrite, tensor2img, img2tensor
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel
# from basicsr.models.receptive_cal import *
import copy


def check_image_size(x, ws):
    _, _, h, w = x.size()
    mod_pad_h = (ws - h % ws) % ws
    mod_pad_w = (ws - w % ws) % ws
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    return x

@MODEL_REGISTRY.register()
class DAModel(BaseModel):

    def __init__(self, opt):
        super(DAModel, self).__init__(opt)
        
        # print(torch.cuda.memory_reserved())
        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)
        # print(torch.cuda.max_memory_allocated())
        # input_flop = torch.randn(1,3,1024,1024).cuda()
        # input_psf_flop = torch.randn(1,28,256,256).cuda()
        # begin = time.time()
        # for i in range(0,3):
        #     with torch.no_grad():
        #         # a = torch.cuda.max_memory_allocated()
        #         # print(a)
        #         out = self.net_g.forward(input_flop, input_psf_flop)
        #         # out = self.net_g.forward(input_flop)
        #         # b = torch.cuda.max_memory_allocated()
        #         # print(b)
        #         # print(b-a)
        #     # print(torch.cuda.memory_reserved())
        # end = time.time()
        # inference_time = (end-begin)/3
        # print(inference_time)
        # macs, params = profile(self.net_g, inputs=(input_flop, input_psf_flop))

        # # input_flop = torch.randn(1,3,1024,1024).cuda()
        # # # input_psf_flop = torch.randn(1,28,64,64).cuda()
        # macs, params = profile(self.net_g, inputs=(input_flop, ))

        # gflops = macs/2
        # logger = get_root_logger()
        # logger.info(f'Network, with parameters: {params:,f}, with GFLOPs: {gflops:,f}')
        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        load_key = self.opt['path'].get('param_key_g', None)
        # print(load_key)
        # print( self.net_g.state_dict())
        if load_path is not None:
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), load_key)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            for p in self.net_g_ema.parameters():
                p.requires_grad = False
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        if train_opt.get('trgtv_opt'):
            self.cri_trgtv = build_loss(train_opt['trgtv_opt']).to(self.device)
            self.model_to_device(self.cri_trgtv)
        else:
            self.cri_trgtv = None

        if train_opt.get('trgdc_opt'):
            self.cri_trgdc = build_loss(train_opt['trgdc_opt']).to(self.device)
            self.model_to_device(self.cri_trgdc)

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    # def feed_data(self, data):
    #     self.lq = data['lq'].to(self.device)
    #     # self.psf = data['psf'].to(self.device)
    #     if 'gt' in data:
    #         self.gt = data['gt'].to(self.device)

    def feed_data(self, data):
        self.lqsrc = data['lq'].to(self.device)
        if 'trg' in data:
            self.lqtrg = data['trg'].to(self.device)
            self.gttrg = data['trg'].to(self.device)
        if 'gt' in data:
            self.gtsrc = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        # scaler = GradScaler()
        # with autocast():
        self.output = self.net_g(self.lqsrc)
        self.toutput = self.net_g(self.lqtrg)
        # self.cycle_degredation()

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gtsrc)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix

        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gtsrc)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        # tv loss
        if self.cri_trgtv:
            l_tv = self.cri_trgtv(self.toutput)
            l_total += l_tv
            loss_dict['l_trgtv'] = l_tv
            # l_stv = self.cri_trgtv(self.output_residual)
            # l_g_total += l_stv
            # loss_dict['l_stv'] = l_stv

        if self.cri_trgdc:
            l_dc = self.cri_trgdc(self.toutput)
            l_total += l_dc
            loss_dict['l_trgdc'] = l_dc
            # l_sdc = self.cri_trgdc(self.output_residual)
            # l_g_total += l_sdc
            # loss_dict['l_sdc'] = l_sdc

        l_total.backward()
        # scaler.step(self.optimizer_g)
        # scaler.update()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lqsrc)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lqsrc)
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
        pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            if self.opt['val'].get('grids', False):
                self.grids()
            self.test()
            if self.opt['val'].get('grids', False):
                self.grids_inverse()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            if 'gtsrc' in visuals:
                gt_img = tensor2img([visuals['gtsrc']])
                del self.gtsrc

            # tentative for out of GPU memory
            del self.lqsrc
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], str(current_iter),dataset_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}.png')
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    if name =='niqe' or name =='piqe' or name =='brisque':
                        metric_data = dict(img=sr_img)
                    else:
                        metric_data = dict(img1=sr_img, img2=gt_img)
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
        pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            # print(value)
            log_str += f'\t # {metric}: {value:.4f}\n'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lqsrc.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gtsrc'):
            out_dict['gtsrc'] = self.gtsrc.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)

@MODEL_REGISTRY.register()
class DAmodel_FeMaSR(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)

         # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        
        # self.net_s = build_network(opt['network_s'])
        # self.net_s = self.model_to_device(self.net_s)

        # # define metric functions 
        # if self.opt['val'].get('metrics') is not None:
        #     self.metric_funcs = {}
        #     for _, opt in self.opt['val']['metrics'].items(): 
        #         mopt = opt.copy()
        #         name = mopt.pop('type', None)
        #         mopt.pop('better', None)
        #         # self.metric_funcs[name] = pyiqa.create_metric(name, device=self.device, **mopt)

        # load pre-trained HQ ckpt, frozen decoder and codebook 
        self.LQ_stage = self.opt['network_g'].get('LQ_stage', False) 
        if self.LQ_stage:
            load_path = self.opt['path'].get('pretrain_network_hq', None)
            if load_path is not None:
                load_path = self.opt['path'].get('pretrain_network_hq', None)
                assert load_path is not None, 'Need to specify hq prior model path in LQ stage'

                hq_opt = self.opt['network_g'].copy()
                hq_opt['LQ_stage'] = False
                self.net_hq = build_network(hq_opt)
                self.net_hq = self.model_to_device(self.net_hq)
                self.load_network(self.net_hq, load_path, self.opt['path']['strict_load'])

                self.load_network(self.net_g, load_path, False)
            frozen_module_keywords = self.opt['network_g'].get('frozen_module_keywords', None) 
            if frozen_module_keywords is not None:
                for name, module in self.net_g.named_modules():
                    for fkw in frozen_module_keywords:
                        if fkw in name:
                            for p in module.parameters():
                                p.requires_grad = False
                            break
        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        logger = get_root_logger()
        if load_path is not None:
            logger.info(f'Loading net_g from {load_path}')
            self.load_network(self.net_g, load_path, self.opt['path']['strict_load'])
            
        if self.is_train:
            self.init_training_settings()
            self.use_dis = (self.opt['train']['gan_opt_t']['loss_weight'] != 0) 
            self.net_d_best = copy.deepcopy(self.net_d)
        
        self.net_g_best = copy.deepcopy(self.net_g)

    def init_training_settings(self):
        logger = get_root_logger()
        train_opt = self.opt['train']
        self.net_g.train()
        # self.net_s.train()
        # define network net_d
        self.net_d = build_network(self.opt['network_d'])
        self.net_d = self.model_to_device(self.net_d)
        
        self.net_dt = build_network(self.opt['network_dt'])
        self.net_dt = self.model_to_device(self.net_dt)
        # load pretrained d models
        load_path = self.opt['path'].get('pretrain_network_d', None)
        # print(load_path)
        if load_path is not None:
            logger.info(f'Loading net_dt from {load_path}')
            self.load_network(self.net_dt, load_path, self.opt['path'].get('strict_load_d', True))
            
        self.net_d.train()
        self.net_dt.train()
        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
            self.model_to_device(self.cri_perceptual)
        else:
            self.cri_perceptual = None

        if train_opt.get('trgtv_opt'):
            self.cri_trgtv = build_loss(train_opt['trgtv_opt']).to(self.device)
            self.model_to_device(self.cri_trgtv)
        else:
            self.cri_trgtv = None

        if train_opt.get('trgdc_opt'):
            self.cri_trgdc = build_loss(train_opt['trgdc_opt']).to(self.device)
            self.model_to_device(self.cri_trgdc)
        else:
            self.cri_trgtv = None
            
        if train_opt.get('maxmi_opt'):
            self.cri_maxmi = build_loss(train_opt['maxmi_opt']).to(self.device)
            self.model_to_device(self.cri_maxmi)
        else:
            self.cri_maxmi = None
            
        if train_opt.get('minmi_opt'):
            self.cri_minmi = build_loss(train_opt['minmi_opt']).to(self.device)
            self.model_to_device(self.cri_minmi)
        else:
            self.cri_minmi = None

        if train_opt.get('gan_opt'):
            self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)
            
        if train_opt.get('gan_opt_t'):
            self.cri_gan_t = build_loss(train_opt['gan_opt_t']).to(self.device)

        self.net_d_iters = train_opt.get('net_d_iters', 1)
        self.net_d_init_iters = train_opt.get('net_d_init_iters', 0)

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()
    
    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            optim_params.append(v)
            if not v.requires_grad:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        # optimizer g
        optim_type = train_opt['optim_g'].pop('type')
        optim_class = getattr(torch.optim, optim_type)
        self.optimizer_g = optim_class(optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

        optim_params_d = []
        for k, v in self.net_d.named_parameters():
            optim_params_d.append(v)
            if not v.requires_grad:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        for k, v in self.net_dt.named_parameters():
            optim_params_d.append(v)
            if not v.requires_grad:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        # # optimizer d
        optim_type = train_opt['optim_d'].pop('type')
        optim_class = getattr(torch.optim, optim_type)
        self.optimizer_d = optim_class(optim_params_d, **train_opt['optim_d'])
        self.optimizers.append(self.optimizer_d)

    # def feed_data(self, datasrc, datatrg=None):
    #     self.lqsrc = datasrc['lq'].to(self.device)
    #     if datatrg is not None:
    #         self.lqtrg = datatrg['lq'].to(self.device)
    #     if 'gt' in datasrc:
    #         self.gtsrc = datasrc['gt'].to(self.device)
    #         if datatrg is not None:
    #             self.gttrg = datatrg['gt'].to(self.device)

    def feed_data(self, data):
        self.lqsrc = data['lq'].to(self.device)
        if 'trg' in data:
            self.lqtrg = data['trg'].to(self.device)
            self.gttrg = data['trg'].to(self.device)
        if 'gt' in data:
            self.gtsrc = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        train_opt = self.opt['train']

        for p in self.net_d.parameters():
            p.requires_grad = False
        for p in self.net_dt.parameters():
            p.requires_grad = False
        self.optimizer_g.zero_grad()
        
        # #s2t
        # with torch.no_grad():
        #     self.s2t, _, _, _= self.net_s(self.lqsrc)
        #     # min_max = (0, 1)
        #     # self.s2t = self.s2t.clamp_(*min_max)
        #src_forward
        # self.output_residual, feat_quant = self.net_g(self.lqsrc)
        if self.LQ_stage:
            with torch.no_grad():
                self.gt_rec, _, _, gt_indices = self.net_hq(self.gtsrc)

            self.output_residual, l_codebook, l_semantic, _ = self.net_g(self.lqsrc, gt_indices) 
            # self.output_residual, l_codebook, l_semantic, _, feat_quant = self.net_g(self.lqsrc, gt_indices) 
        # else:
        #     self.output, l_codebook, l_semantic, _, fsrc = self.net_g(self.lqsrc) 
        self.toutput, tl_codebook, tl_semantic, _ = self.net_g(self.lqtrg)
        # self.toutput, tl_codebook, tl_semantic, _, tfeat_quant = self.net_g(self.lqtrg)
        # self.output_residual, l_codebook, _, feat_quant= self.net_g(self.lqsrc)
        #src_forward2
        # self.output_s2t, l_codebook_s2t, _, feat_quant_s2t= self.net_g(self.s2t)
        #trg_forward
        
        
        l_g_total = 0
        loss_dict = OrderedDict()



        # ===================================================
        # codebook loss
        if train_opt.get('codebook_opt', None):
            l_codebook *= train_opt['codebook_opt']['loss_weight'] 
            l_g_total += l_codebook.mean()
            loss_dict['l_codebook'] = l_codebook.mean()
            
            tl_codebook *= train_opt['codebook_opt']['loss_weight'] 
            l_g_total += tl_codebook.mean()
            loss_dict['tl_codebook'] = tl_codebook.mean()
            # l_g_total += l_codebook_s2t.mean()
            # loss_dict['l_codebook_s2t'] = l_codebook_s2t.mean()
            # l_g_totalt += tl_codebook.mean()
            # loss_dict['tl_codebook'] = tl_codebook.mean()

        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output_residual, self.gtsrc)
            # print(l_pix.shape)
            l_g_total += l_pix
            loss_dict['l_pix'] = l_pix
            # l_pix_s2t = self.cri_pix(self.output_s2t, self.gtsrc)
            # l_g_total += l_pix_s2t
            # loss_dict['l_pix_s2t'] = l_pix_s2t


        
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output_residual, self.gtsrc)
            # l_percep_s2t, l_style_s2t = self.cri_perceptual(self.output_s2t, self.gtsrc)
            if l_percep is not None:
                l_g_total += l_percep.mean()
                loss_dict['l_percep'] = l_percep.mean()
                # l_g_total += l_percep_s2t.mean()
                # loss_dict['l_percep_s2t'] = l_percep_s2t.mean()
            if l_style is not None:
                l_g_total += l_style
                loss_dict['l_style'] = l_style
        # l_g_total.mean().backward()
        
        
        # self.toutput_residual, tfeat_quant = self.net_g(self.lqtrg)
        # l_g_total += tl_codebook.mean()
        # loss_dict['tl_codebook'] = tl_codebook.mean()
        # tv loss
        if self.cri_trgtv:
            l_tv = self.cri_trgtv(self.toutput)
            l_g_total += l_tv
            loss_dict['l_trgtv'] = l_tv
            # l_stv = self.cri_trgtv(self.output_residual)
            # l_g_total += l_stv
            # loss_dict['l_stv'] = l_stv

        if self.cri_trgdc:
            l_dc = self.cri_trgdc(self.toutput)
            l_g_total += l_dc
            loss_dict['l_trgdc'] = l_dc
            # l_sdc = self.cri_trgdc(self.output_residual)
            # l_g_total += l_sdc
            # loss_dict['l_sdc'] = l_sdc
        # feature_distange
        # gan loss
        if self.use_dis and current_iter > train_opt['net_d_init_iters']:
            # src_fuse_pred = self.net_d(feat_quant)
            # l_g_gan_srcfuse = self.cri_gan(src_fuse_pred, True, is_disc=False)
            # l_g_total += l_g_gan_srcfuse
            # loss_dict['l_g_gan_srcfuse'] = l_g_gan_srcfuse

            # trg_fuse_pred = self.net_d(feat_quant)
            # l_g_gan_trgfuse = self.cri_gan(trg_fuse_pred, 'half', is_disc=False)
            # # l_g_gan_trgfuse = self.cri_gan(trg_fuse_pred, True, is_disc=False)
            # l_g_total += l_g_gan_trgfuse
            # loss_dict['l_g_gan_srcfuse'] = l_g_gan_trgfuse
            
            # trgs_fuse_pred = self.net_d(tfeat_quant)
            # l_g_gan_trgsfuse = self.cri_gan(trgs_fuse_pred, 'half', is_disc=False)
            # # l_g_gan_trgsfuse = self.cri_gan(trgs_fuse_pred, True, is_disc=False)
            # l_g_total += l_g_gan_trgsfuse
            # loss_dict['l_g_gan_trgfuse'] = l_g_gan_trgsfuse
            
            out_fake_g_preds = self.net_dt(self.output_residual)
            # print(fake_g_pred.size())
            l_g_gans = self.cri_gan_t(out_fake_g_preds, True, is_disc=False)
            l_g_total += l_g_gans
            loss_dict['l_g_gans'] = l_g_gans
            
            # out_fake_g_pred = self.net_dt(self.toutput_residual)
            # # print(fake_g_pred.size())
            # l_g_gan = self.cri_gan_t(out_fake_g_pred, True, is_disc=False)
            # l_g_total += l_g_gan
            # loss_dict['l_g_gan'] = l_g_gan
        
            # out_fake_g_predst = self.net_dt(self.output_s2t)
            # # print(fake_g_pred.size())
            # l_g_ganst = self.cri_gan_t(out_fake_g_predst, True, is_disc=False)
            # l_g_total += l_g_ganst
            # loss_dict['l_g_ganst'] = l_g_ganst
            
        l_g_total.mean().backward()
        self.optimizer_g.step()

        # optimize net_d
        self.fixed_disc = self.opt['train'].get('fixed_disc', False)
        if not self.fixed_disc and self.use_dis and current_iter > train_opt['net_d_init_iters']:
            for p in self.net_d.parameters():
                p.requires_grad = True
            for p in self.net_dt.parameters():
                p.requires_grad = True
            self.optimizer_d.zero_grad()

            # src_fuse_real = self.net_d(tfeat_quant.detach())
            # l_d_srcreal = self.cri_gan(src_fuse_real, True, is_disc=True)
            # loss_dict['l_d_srcreal'] = l_d_srcreal
            # loss_dict['out_d_srcreal'] = torch.mean(src_fuse_real.detach())
            # l_d_srcreal.backward()

            # trg_fuse_fake = self.net_d(feat_quant.detach())
            # l_d_trgfake = self.cri_gan(trg_fuse_fake, False, is_disc=True)
            # loss_dict['l_d_trgfake'] = l_d_trgfake 
            # loss_dict['out_d_trgfake'] = torch.mean(trg_fuse_fake.detach())
            # l_d_trgfake.backward()
            

            real_d_pred = self.net_dt(self.gtsrc)
            l_d_reals = self.cri_gan_t(real_d_pred, True, is_disc=True)
            loss_dict['l_d_reals'] = l_d_reals
            loss_dict['out_d_reals'] = torch.mean(real_d_pred.detach())
            l_d_reals.backward()
            
            fake_d_pred = self.net_dt(self.output_residual.detach())
            l_d_fake = self.cri_gan_t(fake_d_pred, False, is_disc=True)
            loss_dict['l_d_fake'] = l_d_fake
            loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
            l_d_fake.backward()
            
            # fake_d_predt = self.net_dt(self.toutput_residual.detach())
            # l_d_faket = self.cri_gan_t(fake_d_predt, False, is_disc=True)
            # loss_dict['l_d_faket'] = l_d_faket
            # loss_dict['out_d_faket'] = torch.mean(fake_d_predt.detach())
            # l_d_faket.backward()
            
            # fake_d_predst = self.net_dt(self.output_s2t.detach())
            # l_d_fakest = self.cri_gan_t(fake_d_predst, False, is_disc=True)
            # loss_dict['l_d_fakest'] = l_d_fakest
            # loss_dict['out_d_fakest'] = torch.mean(fake_d_predst.detach())
            # l_d_fakest.backward()
            
            self.optimizer_d.step()

        
        self.log_dict = self.reduce_loss_dict(loss_dict)
        
    def test(self):
        self.net_g.eval()
        net_g = self.get_bare_model(self.net_g)
        min_size = 8000 * 8000 # use smaller min_size with limited GPU memory
        lq_input = self.lqsrc
        _, _, h, w = lq_input.shape
        lq_input = check_image_size(lq_input, 64)
        if h*w < min_size:
            self.output = net_g.test(lq_input)
        else:
            self.output = net_g.test_tile(lq_input)
        self.net_g.train()
        
    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, save_as_dir=None):
        logger = get_root_logger()
        logger.info('Only support single GPU validation.')
        self.nondist_validation(dataloader, current_iter, tb_logger, save_img, save_as_dir)


    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, save_as_dir=None):
        logger = get_root_logger()
        logger.info('Only support single GPU validation.')
        self.nondist_validation(dataloader, current_iter, tb_logger, save_img, save_as_dir)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
        pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            if self.opt['val'].get('grids', False):
                self.grids()
            self.test()
            if self.opt['val'].get('grids', False):
                self.grids_inverse()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            if 'gtsrc' in visuals:
                gt_img = tensor2img([visuals['gtsrc']])
                del self.gtsrc

            # tentative for out of GPU memory
            del self.lqsrc
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], str(current_iter),dataset_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}.png')
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    if name =='niqe' or name =='piqe' or name =='brisque':
                        metric_data = dict(img=sr_img)
                    else:
                        metric_data = dict(img1=sr_img, img2=gt_img)
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
        pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
            
    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)
    
    def vis_single_code(self, up_factor=2):
        net_g = self.get_bare_model(self.net_g)
        codenum = self.opt['network_g']['codebook_params'][0][1]
        with torch.no_grad():
            code_idx = torch.arange(codenum).reshape(codenum, 1, 1, 1)
            code_idx = code_idx.repeat(1, 1, up_factor, up_factor)
            output_img = net_g.decode_indices(code_idx) 
            output_img = tvu.make_grid(output_img, nrow=32)

        return output_img.unsqueeze(0)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lqsrc'] = self.lqsrc.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gtsrc'):
            out_dict['gtsrc'] = self.gtsrc.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_network(self.net_d, 'net_d', current_iter)
        self.save_network(self.net_dt, 'net_dt', current_iter)
        # self.save_network(self.net_s, 'net_s', current_iter)
        self.save_training_state(epoch, current_iter)

