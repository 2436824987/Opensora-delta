import argparse
import numpy as np
import os
import random
import torch
import torch.nn as nn
import sys
import time

import argparse
import os
from tqdm import tqdm
import torchvision.transforms as transforms
import collections
sys.setrecursionlimit(10000)
import functools

import argparse
import os
from torch import autocast
from contextlib import contextmanager, nullcontext
import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from scipy import linalg

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from ldm.data.build_dataloader import build_dataloader

# from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor
import logging
from torch.nn.functional import adaptive_avg_pool2d
from mmengine.runner import set_random_seed
# from pytorch_lightning import seed_everything
from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like
# from pytorch_fid.inception import InceptionV3
import copy

# Open-Sora related imports
from pprint import pformat

import colossalai
import torch
import torch.distributed as dist
from colossalai.cluster import DistCoordinator
from mmengine.runner import set_random_seed
from tqdm import tqdm

from opensora.acceleration.parallel_states import set_sequence_parallel_group
from opensora.datasets import save_sample
from opensora.datasets.aspect import get_image_size, get_num_frames
from opensora.models.text_encoder.t5 import text_preprocessing
from opensora.registry import MODELS, SCHEDULERS, build_module
from opensora.utils.config_utils import read_config
from opensora.utils.inference_utils import (
    add_watermark,
    append_generated,
    append_score_to_prompts,
    apply_mask_strategy,
    collect_references_batch,
    dframe_to_frame,
    extract_json_from_prompts,
    extract_prompts_loop,
    get_save_path_name,
    load_prompts,
    merge_prompt,
    prepare_multi_resolution_info,
    refine_prompts_by_openai,
    split_prompt,
)
from opensora.utils.misc import all_exists, create_logger, is_distributed, is_main_process, to_torch_dtype

choice = lambda x: x[np.random.randint(len(x))] if isinstance(
    x, tuple) else choice(tuple(x))

# load safety model
# safety_model_id = "CompVis/stable-diffusion-safety-checker"
# safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
# safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)

def get_activations(data, model, batch_size=50, dims=2048, device='cpu',
                    num_workers=1):

    # model.eval()

    # if batch_size > data.shape[0]:
    #     print(('Warning: batch size is bigger than the data size. '
    #            'Setting batch size to data size'))
    #     batch_size = data.shape[0]

    # pred_arr = np.empty((data.shape[0], dims))
    # start_idx = 0

    # for i in range(0, data.shape[0], batch_size):
    #     if i + batch_size > data.shape[0]:
    #         batch = data[i:, :, :, :]
    #     else:
    #         batch = data[i:i+batch_size, :, :, :]
    #     batch = batch.to(device)

    #     with torch.no_grad():
    #         pred = model(batch)[0]

    #     if pred.size(2) != 1 or pred.size(3) != 1:
    #         pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
        
    #     pred = pred.squeeze(3).squeeze(2).cpu().numpy()

    #     pred_arr[start_idx:start_idx + pred.shape[0]] = pred

    #     start_idx = start_idx + pred.shape[0]
    
    # return pred_arr
    pass

def calculate_activation_statistics(datas, model, batch_size=50, dims=2048,
                                    device='cpu', num_workers=1):
    # act = get_activations(datas, model, batch_size, dims, device, num_workers)
    # mu = np.mean(act, axis=0)
    # sigma = np.cov(act, rowvar=False)
    # return mu, sigma
    pass

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    # mu1 = np.atleast_1d(mu1)
    # mu2 = np.atleast_1d(mu2)
    # # mu2 = np.sum(mu2, axis=1) # To fix the shape issue
    # mu2 = np.mean(mu2, axis=1)

    # sigma1 = np.atleast_2d(sigma1)
    # sigma2 = np.atleast_2d(sigma2)

    # # print("mu1[0]: ", mu1[0])
    # # print("mu2[0][0]: ", mu2[0][0])
    # # print("mu2[1][1]: ", mu2[1][1])
    # # print("training  shape:", mu1.shape)
    # # print("reference shape:", mu2.shape)
    # # print("===========================")
    # # exit(0)

    # assert mu1.shape == mu2.shape, \
    #     'Training and test mean vectors have different lengths'
    # assert sigma1.shape == sigma2.shape, \
    #     'Training and test covariances have different dimensions'

    # diff = mu1 - mu2

    # # Product might be almost singular
    # covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    # if not np.isfinite(covmean).all():
    #     msg = ('fid calculation produces singular product; '
    #            'adding %s to diagonal of cov estimates') % eps
    #     print(msg)
    #     offset = np.eye(sigma1.shape[0]) * eps
    #     covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # # Numerical error might give slight imaginary component
    # if np.iscomplexobj(covmean):
    #     if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
    #         m = np.max(np.abs(covmean.imag))
    #         raise ValueError('Imaginary component {}'.format(m))
    #     covmean = covmean.real

    # tr_covmean = np.trace(covmean)

    # return (diff.dot(diff) + np.trace(sigma1)
    #         + np.trace(sigma2) - 2 * tr_covmean)
    pass

def calculate_fid(data1, ref_mu, ref_sigma, batch_size, device, dims, num_workers=1):
    
    # block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    # model = InceptionV3([block_idx]).to(device)

    # m1, s1 = calculate_activation_statistics(data1, model, batch_size,
    #                                         dims, device, num_workers)
    
    # fid_value = calculate_frechet_distance(m1, s1, ref_mu, ref_sigma)

    pass

class EvolutionSearcher(object):

    def __init__(self, opt, model, text_encoder, vae, time_step, ref_mu, ref_sigma, sampler, dataloader_info, batch_size, dtype, dpm_params=None):
        self.opt = opt
        self.model = model
        self.text_encoder = text_encoder
        self.vae = vae
        self.sampler = sampler
        self.time_step = time_step
        self.dataloader_info = dataloader_info
        self.batch_size = batch_size
        # self.cfg = cfg
        ## EA hyperparameters
        self.max_epochs = opt.max_epochs
        self.select_num = opt.select_num
        self.population_num = opt.population_num
        self.m_prob = opt.m_prob
        self.crossover_num = opt.crossover_num
        self.mutation_num = opt.mutation_num
        self.num_samples = opt.num_sample
        self.ddim_discretize = "uniform"
        ## tracking variable 
        self.keep_top_k = {self.select_num: [], 50: []}
        self.epoch = 0
        self.candidates = []
        self.vis_dict = {}

        self.use_ddim_init_x = opt.use_ddim_init_x

        self.ref_mu = np.load(ref_mu)
        self.ref_sigma = np.load(ref_sigma)

        self.dpm_params = dpm_params
        self.dtype = dtype
    
    def update_top_k(self, candidates, *, k, key, reverse=False):
        assert k in self.keep_top_k
        logging.info('select ......')
        t = self.keep_top_k[k]
        t += candidates
        t.sort(key=key, reverse=reverse)
        self.keep_top_k[k] = t[:k]
    
    def is_legal_before_search(self, cand):
        cand = eval(cand)
        cand = sorted(cand)
        cand = str(cand)
        if cand not in self.vis_dict:
            self.vis_dict[cand] = {}
        info = self.vis_dict[cand]
        if 'visited' in info:
            logging.info('cand: {} has visited!'.format(cand))
            return False
        info['fid'] = self.get_cand_fid(opt=self.opt, cand=eval(cand))
        logging.info('cand: {}, fid: {}'.format(cand, info['fid']))

        info['visited'] = True
        return True
    
    def is_legal(self, cand):
        cand = eval(cand)
        cand = sorted(cand)
        cand = str(cand)
        if cand not in self.vis_dict:
            self.vis_dict[cand] = {}
        info = self.vis_dict[cand]
        if 'visited' in info:
            logging.info('cand: {} has visited!'.format(cand))
            return False
        # if self.RandomForestClassifier.predict_proba(np.asarray(eval(cand), dtype='float')[None, :])[0,1] < self.thres: # 拒绝
        #     logging.info('cand: {} is not legal.'.format(cand))
        #     return False
        info['fid'] = self.get_cand_fid(opt=self.opt, cand=eval(cand))
        logging.info('cand: {}, fid: {}'.format(cand, info['fid']))

        info['visited'] = True
        return True
    
    def get_random_before_search(self, num):
        logging.info('random select ........')
        while len(self.candidates) < num:
            if self.opt.dpm_solver:
                cand = self.sample_active_subnet_dpm()
            else:
                cand = self.sample_active_subnet()
            cand = sorted(cand)
            cand = str(cand)
            if not self.is_legal_before_search(cand):
                continue
            self.candidates.append(cand)
            logging.info('random {}/{}'.format(len(self.candidates), num))
        logging.info('random_num = {}'.format(len(self.candidates)))
    
    def get_random(self, num):
        logging.info('random select ........')
        while len(self.candidates) < num:
            if self.opt.dpm_solver:
                cand = self.sample_active_subnet_dpm()
            else:
                cand = self.sample_active_subnet()
            cand = sorted(cand)
            cand = str(cand)
            if not self.is_legal(cand):
                continue
            self.candidates.append(cand)
            logging.info('random {}/{}'.format(len(self.candidates), num))
        logging.info('random_num = {}'.format(len(self.candidates)))
    
    def get_cross(self, k, cross_num):
        assert k in self.keep_top_k
        logging.info('cross ......')
        res = []
        max_iters = cross_num * 10

        def random_cross():
            cand1 = choice(self.keep_top_k[k])
            cand2 = choice(self.keep_top_k[k])

            new_cand = []
            cand1 = eval(cand1)
            cand2 = eval(cand2)

            for i in range(len(cand1)):
                if np.random.random_sample() < 0.5:
                    new_cand.append(cand1[i])
                else:
                    new_cand.append(cand2[i])

            return new_cand

        while len(res) < cross_num and max_iters > 0:
            max_iters -= 1
            cand = random_cross()
            cand = sorted(cand)
            cand = str(cand)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            logging.info('cross {}/{}'.format(len(res), cross_num))

        logging.info('cross_num = {}'.format(len(res)))
        return res
    
    def get_mutation(self, k, mutation_num, m_prob):
        assert k in self.keep_top_k
        logging.info('mutation ......')
        res = []
        iter = 0
        max_iters = mutation_num * 10

        def random_func():
            cand = choice(self.keep_top_k[k])
            cand = eval(cand)

            candidates = []
            for i in range(self.sampler.ddpm_num_timesteps):
                if i not in cand:
                    candidates.append(i)

            for i in range(len(cand)):
                if np.random.random_sample() < m_prob:
                    new_c = random.choice(candidates)
                    new_index = candidates.index(new_c)
                    del(candidates[new_index])
                    cand[i] = new_c
                    if len(candidates) == 0:  # cand 的长度小于 candidates 的长度
                        break

            return cand

        while len(res) < mutation_num and max_iters > 0:
            max_iters -= 1
            cand = random_func()
            cand = sorted(cand)
            cand = str(cand)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            logging.info('mutation {}/{}'.format(len(res), mutation_num))

        logging.info('mutation_num = {}'.format(len(res)))
        return res
    
    def get_mutation_dpm(self, k, mutation_num, m_prob):
        assert k in self.keep_top_k
        logging.info('mutation ......')
        res = []
        iter = 0
        max_iters = mutation_num * 10

        def random_func():
            cand = choice(self.keep_top_k[k])
            cand = eval(cand)

            candidates = []
            for i in self.dpm_params['full_timesteps']:
                if i not in cand:
                    candidates.append(i)

            for i in range(len(cand)):
                if np.random.random_sample() < m_prob:
                    new_c = random.choice(candidates)
                    new_index = candidates.index(new_c)
                    del(candidates[new_index])
                    cand[i] = new_c
                    if len(candidates) == 0:  # cand 的长度小于 candidates 的长度
                        break

            return cand

        while len(res) < mutation_num and max_iters > 0:
            max_iters -= 1
            cand = random_func()
            cand = sorted(cand)
            cand = str(cand)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            logging.info('mutation {}/{}'.format(len(res), mutation_num))

        logging.info('mutation_num = {}'.format(len(res)))
        return res
    
    def mutate_init_x(self, x0, mutation_num, m_prob):
        logging.info('mutation x0 ......')
        res = []
        iter = 0
        max_iters = mutation_num * 10

        def random_func():
            cand = x0
            cand = eval(cand)

            candidates = []
            for i in range(self.sampler.ddpm_num_timesteps):
                if i not in cand:
                    candidates.append(i)

            for i in range(len(cand)):
                if np.random.random_sample() < m_prob:
                    new_c = random.choice(candidates)
                    new_index = candidates.index(new_c)
                    del(candidates[new_index])
                    cand[i] = new_c
                    if len(candidates) == 0:  # cand 的长度小于 candidates 的长度
                        break

            return cand

        while len(res) < mutation_num and max_iters > 0:
            max_iters -= 1
            cand = random_func()
            cand = sorted(cand)
            cand = str(cand)
            if not self.is_legal_before_search(cand):
                continue
            res.append(cand)
            logging.info('mutation x0 {}/{}'.format(len(res), mutation_num))

        logging.info('mutation_num = {}'.format(len(res)))
        return res

    def mutate_init_x_dpm(self, x0, mutation_num, m_prob):
        logging.info('mutation x0 ......')
        res = []
        iter = 0
        max_iters = mutation_num * 10

        def random_func():
            cand = x0
            cand = eval(cand)

            candidates = []
            for i in self.dpm_params['full_timesteps']:
                if i not in cand:
                    candidates.append(i)

            for i in range(len(cand)):
                if np.random.random_sample() < m_prob:
                    new_c = random.choice(candidates)
                    new_index = candidates.index(new_c)
                    del(candidates[new_index])
                    cand[i] = new_c
                    if len(candidates) == 0:  # cand 的长度小于 candidates 的长度
                        break

            return cand

        while len(res) < mutation_num and max_iters > 0:
            max_iters -= 1
            cand = random_func()
            cand = sorted(cand)
            cand = str(cand)
            if not self.is_legal_before_search(cand):
                continue
            res.append(cand)
            logging.info('mutation x0 {}/{}'.format(len(res), mutation_num))

        logging.info('mutation_num = {}'.format(len(res)))
        return res

    def sample_active_subnet(self):
        original_num_steps = self.sampler.ddpm_num_timesteps
        use_timestep = [i for i in range(original_num_steps)]
        random.shuffle(use_timestep)
        use_timestep = use_timestep[:self.time_step]
        # use_timestep = [use_timestep[i] + 1 for i in range(len(use_timestep))] 
        return use_timestep
    
    def sample_active_subnet_dpm(self):
        use_timestep = copy.deepcopy(self.dpm_params['full_timesteps'])
        random.shuffle(use_timestep)
        use_timestep = use_timestep[:self.time_step + 1]
        # use_timestep = [use_timestep[i] + 1 for i in range(len(use_timestep))] 
        return use_timestep
    
    def get_cand_fid(self, cand=None, opt=None, device='cuda'):
        # z = torch.randn(len(batch_prompts), vae.out_channels, *latent_size, device=device, dtype=dtype)
        # with torch.no_grad():
        #     t1 = time.time()
        #     all_samples = list()
        #     for itr, batch in enumerate(self.dataloader_info['validation_loader']):
        #         # for k, v in batch.items():
        #         #     if torch.is_tensor(v):
        #         #         batch[k] = v.cuda()
        #         prompts = batch['text']
        #         uc = None
        #         if opt.scale != 1.0:
        #             uc = self.model.get_learned_conditioning(self.batch_size * [""])
        #         if isinstance(prompts, tuple):
        #             prompts = list(prompts)
        #         c = self.model.get_learned_conditioning(prompts)
        #         shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
        #         sampled_timestep = np.array(cand)
        #         # TODO: Pass the corresponding params to rf scheduler
        #         samples_ddim, _ = self.sampler.sample(S=opt.time_step,
        #                                             conditioning=c,
        #                                             batch_size=opt.n_samples,
        #                                             shape=shape,
        #                                             verbose=False,
        #                                             unconditional_guidance_scale=opt.scale,
        #                                             unconditional_conditioning=uc,
        #                                             eta=opt.ddim_eta,
        #                                             x_T=start_code,
        #                                             sampled_timestep=sampled_timestep)
        #         x_samples_ddim = self.model.decode_first_stage(samples_ddim)
        #         x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
        #         x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

        #         # x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim)
        #         x_checked_image = x_samples_ddim

        #         x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

        #         for x_sample in x_checked_image_torch:
        #             all_samples.append(x_sample.cpu().numpy())

        #         logging.info('samples: ' + str(len(all_samples)))
            
        #         if len(all_samples) > self.num_samples:
        #             logging.info('samples: ' + str(len(all_samples)))
        #             break
        # sample_time = time.time() - t1
        # # active model
        # t1 = time.time()
        # all_samples = np.array(all_samples)
        # all_samples = torch.Tensor(all_samples)
        # # fid = calculate_fid(data1=all_samples,ref_mu=self.ref_mu, ref_sigma=self.ref_sigma, batch_size=320, dims=2048, device='cuda')
        # # logging.info('FID: ' + str(fid))

        # fid_time = time.time() - t1
        # logging.info('sample_time: ' + str(sample_time) + ', fid_time: ' + str(fid_time))
        # return fid
        pass
    
    def search(self):
        logging.info('population_num = {} select_num = {} mutation_num = {} crossover_num = {} random_num = {} max_epochs = {}'.format(
            self.population_num, self.select_num, self.mutation_num, self.crossover_num, self.population_num - self.mutation_num - self.crossover_num, self.max_epochs))
        if self.use_ddim_init_x is False:
            self.get_random_before_search(self.population_num)

        else:
            if self.opt.dpm_solver:
                init_x = self.dpm_params['init_timesteps']
            else:
                init_x = make_ddim_timesteps(ddim_discr_method=self.ddim_discretize, num_ddim_timesteps=self.time_step,
                                                        num_ddpm_timesteps=self.sampler.ddpm_num_timesteps,verbose=False)
            init_x = sorted(list(init_x))
            self.is_legal_before_search(str(init_x))
            self.candidates.append(str(init_x))
            self.get_random_before_search(self.population_num // 2)
            if self.opt.dpm_solver:
                res = self.mutate_init_x_dpm(x0=str(init_x), mutation_num=self.population_num - self.population_num // 2 - 1, m_prob=0.1)
            else:
                res = self.mutate_init_x(x0=str(init_x), mutation_num=self.population_num - self.population_num // 2 - 1, m_prob=0.1)
            self.candidates += res

        while self.epoch < self.max_epochs:
            logging.info('epoch = {}'.format(self.epoch))
            self.update_top_k(
                self.candidates, k=self.select_num, key=lambda x: self.vis_dict[x]['fid'])
            self.update_top_k(
                self.candidates, k=50, key=lambda x: self.vis_dict[x]['fid'])

            logging.info('epoch = {} : top {} result'.format(
                self.epoch, len(self.keep_top_k[50])))
            for i, cand in enumerate(self.keep_top_k[50]):
                logging.info('No.{} {} fid = {}'.format(
                    i + 1, cand, self.vis_dict[cand]['fid']))
            
            if self.epoch + 1 == self.max_epochs:
                break
            # sys.exit()
            if self.opt.dpm_solver:
                mutation = self.get_mutation_dpm(
                    self.select_num, self.mutation_num, self.m_prob)
            else:
                mutation = self.get_mutation(
                    self.select_num, self.mutation_num, self.m_prob)

            self.candidates = mutation

            cross_cand = self.get_cross(self.select_num, self.crossover_num)
            self.candidates += cross_cand

            self.get_random(self.population_num) #变异+杂交凑不足population size的部分重新随机采样

            self.epoch += 1
