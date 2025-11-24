import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
import time
from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
import torch.nn.functional as F
from .base_model import BaseModel
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
import time



@MODEL_REGISTRY.register()
class SRModel(BaseModel):

    def __init__(self, opt):
        super(SRModel, self).__init__(opt)
        
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

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        # self.psf = data['psf'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        # scaler = GradScaler()
        # with autocast():
        self.output = self.net_g(self.lq)
        # self.cycle_degredation()

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

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
                self.output = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq)
            self.net_g.train()


    def test_patch(self):
        """Process image and PSF in overlapping patches with confidence map based fusion.
        The confidence map ensures smooth blending in overlapping regions.
        Includes padding to handle images that don't divide evenly by patch size.
        """
        b, c, h, w = self.lq.size()
        self.original_size = self.lq.size()
        assert b == 1  # 确保batch size为1
        # self.psf= self.abd
        # 保存原始图像和PSF
        self.origin_lq = self.lq.clone()
        # self.origin_psf = self.psf.clone()

        # Get patch size and overlap from config
        patch_size = self.opt['val'].get('patch_size', 256)
        overlap = self.opt['val'].get('overlap', 32)

        # Calculate required padding
        stride = patch_size - overlap
        pad_h = (stride - (h - patch_size) % stride) % stride
        pad_w = (stride - (w - patch_size) % stride) % stride

        # Pad image and PSF if necessary
        if pad_h > 0 or pad_w > 0:
            self.lq = F.pad(self.lq, (0, pad_w, 0, pad_h), mode='reflect')
            # self.psf = F.pad(self.psf, (0, pad_w, 0, pad_h), mode='reflect')
            h_padded, w_padded = self.lq.shape[2:]
        else:
            h_padded, w_padded = h, w

        # Calculate number of patches
        num_row = (h_padded - patch_size) // stride + 1
        num_col = (w_padded - patch_size) // stride + 1

        # Initialize confidence map and output tensor
        confidence_map = torch.zeros((b, 1, h_padded, w_padded)).to(self.device)
        output = torch.zeros((b, c, h_padded, w_padded)).to(self.device)

        # Process each patch
        for i in range(num_row):
            for j in range(num_col):
                # Calculate patch coordinates
                start_i = i * stride
                start_j = j * stride
                end_i = start_i + patch_size
                end_j = start_j + patch_size

                # Extract patches for both image and PSF
                img_patch = self.lq[:, :, start_i:end_i, start_j:end_j]
                # psf_patch = self.psf[:, :, start_i:end_i, start_j:end_j]

                # Generate confidence map for this patch
                patch_confidence = torch.ones((b, 1, patch_size, patch_size)).to(self.device)

                # Apply confidence reduction in overlapping regions
                if overlap > 0:
                    # 创建水平和垂直的权重
                    h_weights = torch.ones(patch_size).to(self.device)
                    v_weights = torch.ones(patch_size).to(self.device)

                    # 计算水平权重
                    if j > 0:  # 左重叠
                        h_weights[:overlap] = torch.linspace(0, 1, overlap).to(self.device)
                    if j < num_col - 1:  # 右重叠
                        h_weights[-overlap:] = torch.linspace(1, 0, overlap).to(self.device)

                    # 计算垂直权重
                    if i > 0:  # 上重叠
                        v_weights[:overlap] = torch.linspace(0, 1, overlap).to(self.device)
                    if i < num_row - 1:  # 下重叠
                        v_weights[-overlap:] = torch.linspace(1, 0, overlap).to(self.device)

                    # 将水平和垂直权重组合
                    h_weights = h_weights.view(1, 1, 1, -1)
                    v_weights = v_weights.view(1, 1, -1, 1)
                    patch_confidence = h_weights * v_weights

                # Process patch through network
                if hasattr(self, 'net_g_ema'):
                    self.net_g_ema.eval()
                    with torch.no_grad():
                        patch_output = self.net_g_ema(img_patch)
                        if isinstance(patch_output, list):
                            patch_output = patch_output[-1]
                else:
                    self.net_g.eval()
                    with torch.no_grad():
                        patch_output = self.net_g(img_patch)
                        if isinstance(patch_output, list):
                            patch_output = patch_output[-1]                        
                    self.net_g.train()                
                # with torch.no_grad():
                #     patch_output = self.net_g(img_patch)


                # Add weighted patch to output
                output[:, :, start_i:end_i, start_j:end_j] += patch_output * patch_confidence
                confidence_map[:, :, start_i:end_i, start_j:end_j] += patch_confidence

        # Verify confidence map
        confidence_sum = confidence_map.sum(dim=0, keepdim=True)
        print('min confidence sum: ', confidence_sum.min().item())
        print('max confidence sum: ', confidence_sum.max().item())
        if torch.any(torch.abs(confidence_sum - 1.0) > 1e-6):
            print("Warning: Confidence map weights do not sum to 1.0 in some regions")
            print(f"Min weight: {confidence_sum.min().item()}, Max weight: {confidence_sum.max().item()}")

        # Normalize output by confidence map
        output = output / confidence_map

        # Remove padding if it was added
        if pad_h > 0 or pad_w > 0:
            output = output[:, :, :h, :w]

        self.output = output
        self.lq = self.origin_lq
        # self.psf = self.origin_psf

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
            if self.opt['val'].get('patch', False):
                self.test_patch()   
            else: 
                if self.opt['val'].get('grids', False):
                    self.grids()          
                self.test()
                if self.opt['val'].get('grids', False):
                    self.grids_inverse()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                del self.gt

            # tentative for out of GPU memory
            del self.lq
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
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)




