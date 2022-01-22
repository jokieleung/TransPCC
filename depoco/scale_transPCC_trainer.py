#!/usr/bin/env python3

import depoco.datasets.submap_handler as submap_handler
import depoco.evaluation.evaluator as evaluator
import torch
import time
import numpy as np
import torch.nn as nn
import torch.optim as optim

import os
from torch.utils.tensorboard import SummaryWriter
from ruamel import yaml
import argparse
import depoco.architectures.network_blocks as network
import chamfer3D.dist_chamfer_3D

import depoco.utils.point_cloud_utils as pcu
import subprocess

# import depoco.utils.checkpoint as chkpt
import depoco.architectures.loss_handler as loss_handler
from tqdm.auto import trange, tqdm
torch.autograd.set_detect_anomaly(True)

class DepocoNetTrainer():
    def __init__(self, config):
        t_start = time.time()
        # parameters
        config['git_commit_version'] = str(subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD']).strip())
        self.config = config
        self.experiment_id = self.config["train"]["experiment_id"]
        # self.load_experiment_id = self.config["train"]["load_experiment_id"]
        # self.submaps = submap_handler.SubmapHandler(self.config)
        t_sm = time.time()
        self.submaps = submap_handler.SubMapParser(config)
        print(f'Loaded Submaps ({time.time()-t_sm}s')

        self.max_nr_pts = self.config["train"]["max_nr_pts"]
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        ### Load Encoder and Decoder ####
        t_model = time.time()
        self.encoder_model = None
        self.decoder_model = None
        self.getModel(config)
        print(f'Loaded Model ({time.time()-t_model}s)')

        ##################################
        ########## Loss Attributes #######
        ##################################
        self.cham_loss = chamfer3D.dist_chamfer_3D.chamfer_3DDist()
        self.pairwise_dist = nn.PairwiseDistance()
        self.l2_loss = nn.MSELoss(reduction='mean')
        self.w_transf2map = self.config['train']['loss_weights']['transf2map']
        self.w_map2transf = self.config['train']['loss_weights']['map2transf']

        print(f'Init Trainer ({time.time()-t_start}s)')

    def getModel(self, config: dict):
        """Loads the model specified in self.config
        """
        arch = self.config["network"]
        print(f"network architecture: {arch}")
        self.encoder_model = network.Encoder_PointTrans(
            config['network']['encoder_blocks'])
        self.decoder_model = network.Decoder_PointTrans(
            config['network']['decoder_blocks'])
        print(self.encoder_model)
        print(self.decoder_model)

    def loadModel(self, best: bool = True, device='cuda', out_dir=None):
        if out_dir is None:
            out_dir = self.config["network"]['out_dir']
        enc_path = out_dir+self.experiment_id+'/enc'
        dec_path = out_dir+self.experiment_id+'/dec'
        # enc_path = out_dir+self.load_experiment_id+'/enc'
        # dec_path = out_dir+self.load_experiment_id+'/dec'
        enc_path += '_best.pth' if best else '.pth'
        dec_path += '_best.pth' if best else '.pth'
        print("load", enc_path, ",", dec_path)
        if(os.path.isfile(enc_path) and os.path.isfile(dec_path)):
            # if(os.path.isfile(model_file)):
            self.encoder_model.load_state_dict(torch.load(
                enc_path, map_location=lambda storage, loc: storage))

            self.decoder_model.load_state_dict(torch.load(
                dec_path, map_location=lambda storage, loc: storage))
            self.encoder_model.to(device)
            self.decoder_model.to(device)
        else:
            print(10*'!', 'Cannot load model', 10*'!')

    def saveModel(self, best: bool = False):
        out_dir = self.config["network"]['out_dir']
        enc_path = out_dir+self.experiment_id+'/enc'
        dec_path = out_dir+self.experiment_id+'/dec'
        enc_path += '_best.pth' if best else '.pth'
        dec_path += '_best.pth' if best else '.pth'
        torch.save(self.encoder_model.state_dict(), enc_path)
        torch.save(self.decoder_model.state_dict(), dec_path)

    def saveYaml(self, out_dir="network_files/"):
        config_path = out_dir+self.experiment_id+'/'+self.experiment_id+'.yaml'
        if not os.path.exists(out_dir+self.experiment_id):
            os.makedirs(out_dir+self.experiment_id, exist_ok=True)
        with open(config_path, 'w') as f:
            saver = yaml.YAML()
            saver.dump(self.config, f)

    def test(self, best=True):
        # TEST
        return self.evaluate(self.submaps.getTestSet(),
                             load_model=True,
                             best_model=best,
                             reference_points='map',
                             compute_memory=True, evaluate=True)

    def getNetworkParams(self):
        return list(self.encoder_model.parameters()) + list(self.decoder_model.parameters())

    def getScheduler(self, optimizer, len_data_loader, batch_size):
        number_epochs = self.config['train']['max_epochs']
        steps_per_epoch = int(len_data_loader / batch_size)
        max_lr = self.config["train"]["optimizer"]["max_lr"]
        div_factor = max_lr / \
            self.config["train"]["optimizer"]["start_lr"]
        final_div_factor = self.config["train"]["optimizer"]["start_lr"] / \
            self.config["train"]["optimizer"]["end_lr"]
        pct_start = self.config["train"]['optimizer']["pct_incr_cycle"]
        anneal_strategy = self.config["train"]['optimizer']["anneal_strategy"]
        return optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=steps_per_epoch, epochs=number_epochs, pct_start=pct_start, anneal_strategy=anneal_strategy, div_factor=div_factor, final_div_factor=final_div_factor)

    def getLogWriter(self, logdir):
        if(os.path.isdir(logdir)):
            filelist = [f for f in os.listdir(logdir)]
            for f in filelist:
                os.remove(os.path.join(logdir, f))
        return SummaryWriter(logdir)

    def train(self, verbose=True):
        ###### Setup ######
        if self.config['train']['load_pretrained']:
            self.loadModel(best=True)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.encoder_model.to(self.device)
        self.decoder_model.to(self.device)
        number_epochs = self.config['train']['max_epochs']
        output_path = self.config['network']['out_dir']
        writer = self.getLogWriter(
            output_path+'log/'+self.experiment_id+"/")
        batch_size = min(
            (self.config["train"]["batch_size"], self.submaps.getTrainSize()))

        ###### Init optimizer ########
        lr = self.config["train"]["optimizer"]["max_lr"]

        # learning strategy 1 : Adam 
        optimizer = optim.Adam(self.getNetworkParams(), lr=lr, amsgrad=False,weight_decay=self.config["train"]["optimizer"]["weight_decay"])
        scheduler = self.getScheduler(
            optimizer, self.submaps.getTrainSize(), batch_size)

        # # learning strategy 2 : SGD , still have a NAN problem marked by Jokie 220119
        # optimizer = optim.SGD(self.getNetworkParams(), lr=lr, momentum=self.config["train"]["optimizer"]["momentum"],weight_decay=self.config["train"]["optimizer"]["weight_decay"])
        # scheduler = optim.lr_scheduler.MultiStepLR(
        #     optimizer, milestones=[int(number_epochs*0.6), int(number_epochs*0.8)], gamma=0.1)

        self.saveYaml(out_dir=output_path)

        time_start = time.time()
        optimizer.zero_grad()
        r_batch = 0
        best_loss = 1e10
        n_pct_time = time.time()

        validation_it = 0
        nr_batches = int(self.submaps.getTrainSize()/batch_size)

        LEARNING_RATE_CLIP = 1e-5
        
        ####################################
        ####### Start training #############
        ####################################
        print('Start Training: #self.submaps: %d, #batches: %f' %
              (self.submaps.getTrainSize(), nr_batches))
        for epoch in range(number_epochs):
            running_loss = 0
            batch = 0

            # LR time decay by Jokie 220118
            curr_lr = max(self.config["train"]["optimizer"]["max_lr"] * (self.config["train"]["optimizer"]['lr_decay'] ** (epoch // self.config["train"]["optimizer"]['step_size'])), LEARNING_RATE_CLIP)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            for i, input_dict in enumerate(self.submaps.getTrainSet()):
                # if i >= 5:  # debug evaluation
                #     break
                if i >= (nr_batches * batch_size):  # drop last batch
                    continue
                ######## Preprocess #######
                input_dict['points'] = input_dict['points'].to(self.device)
                input_dict['features'] = input_dict['features'].to(self.device)
                input_points = input_dict['points']
                scale = input_dict['scale'].to(self.device)
                # print('[train] scale value:',input_dict['scale'])

                ####### Encoding and decoding #########
                t1 = time.time()
                # print('input points : ', input_dict.copy()['points'].unsqueeze(0))
                xyz, features, xyz_and_feats = self.encoder_model(input_dict.copy()['points'].unsqueeze(0))
                #print('after encoder points shape: ', xyz.shape)
                #print('after encoder features shape: ', features.shape)
                #for po, fe in xyz_and_feats:
                #    print('[encoder] each block points shape:', po.shape)
                #    print('[encoder] each block features shape:', fe.shape)
                xyz, features, intermedia_xyzs = self.decoder_model(xyz, features, xyz_and_feats)
                #print('after decoder points shape: ', xyz.shape)
                #print('after decoder features shape: ', features.shape)
                #print('after decoder input_points shape: ', input_points.shape)
                xyz = xyz.squeeze(0)
                features = features.squeeze(0)
                translation = features[:, :3]
                #print('after decoder translation shape: ', translation.shape)
                samples = xyz
                #####################
                ###### Loss #########
                #####################
                # print('[Train] Scale: ', scale)
                loss = self.getTrainLoss(input_points, samples, translation, scale)
                loss += loss_handler.PCCRegularizer(
                    intermedia_xyzs,
                    weight=self.config['train']['loss_weights']['upsampling_reg'],
                    gt_points=input_points)

                # print(loss)
                loss.backward()
                running_loss += loss.item()
                ########################################
                ######### Gradient Accumulation ########
                ########################################
                if ((i % batch_size) == (batch_size-1)):
                    # print('[Training] cur i: ' , i)
                    r_batch += 1
                    batch += 1
                    optimizer.step()

                    optimizer.zero_grad()
                    # print('[Training] total running loss: ' , running_loss)
                    running_loss /= batch_size

                    # comment when not using the oneCycleSchedulerLR by Jokie 220118
                    # scheduler.step()
                    # curr_lr = scheduler.get_lr()[0]

                    # Write Log
                    writer.add_scalar('learning loss',
                                      running_loss,
                                      r_batch)
                    writer.add_scalar('learning rate',
                                      curr_lr,
                                      r_batch)
                    if verbose:
                        print('[%d, %5d] loss: %.10f, time: %.4f lr: %.10f' %
                              (epoch + 1, batch, running_loss, time.time()-time_start, curr_lr))

                    time_start = time.time()
                    self.saveModel(best=False)

                    running_loss = 0

            #############################
            ##### validation ############
            #############################
            if (epoch % self.config['train']['validation']['report_rate']) is (self.config['train']['validation']['report_rate']-1):
                ts_val = time.time()
                valid_dict = self.evaluate(
                    dataloader=self.submaps.getValidSet())
                    # dataloader=self.submaps.getValidSet())
                chamf_dist = valid_dict['reconstruction_error']
                
                writer.add_scalar('v: chamfer distance',
                                  chamf_dist,
                                  r_batch)
                if chamf_dist < best_loss:
                    self.saveModel(best=True)
                    best_loss = chamf_dist
                if verbose:
                    print('[%d, valid] rec_err: %.10f, time: %.4f' %
                          (epoch + 1, chamf_dist, time.time()-ts_val))

                validation_it += 1
            ###########################################
            ##### verbose every 10 percent ############
            ###########################################
            if(pcu.isEveryNPercent((epoch), max_it=number_epochs, percent=10)):
                n_pct = (epoch+1)/number_epochs*100
                time_est = (time.time() - n_pct_time)/n_pct*(100-n_pct)
                print("%4d%s in %ds, estim. time left: %ds (%dmin), best loss: %.10f" % (
                    n_pct, "%", time.time() - n_pct_time, time_est, time_est/60, best_loss))

    def getTrainLoss(self, gt_points: torch.tensor, samples, translations, scale ):
        loss = torch.tensor(
            0.0, dtype=torch.float32, device=self.device)  # init loss
        samples_transf = samples + translations

        # print("[train]before scale samples_transf, ", samples_transf)
        # print("[train]before scale gt_points, ", gt_points)

        # Scale to metric space
        samples_tran =samples_transf * scale
        gt_poi = gt_points* scale

        # print("[train]after scale samples_transf, ", samples_tran)
        # print("[train]after scale gt_points, ", gt_poi)
        

        # print("[train]gt_points shape, ", gt_poi.shape)
        # print("[train]samples_transf shape, ", samples_tran.shape)

        # Chamfer Loss between input and samples+T
        d_map2transf, d_transf2map, idx3, idx4 = self.cham_loss(
            gt_poi.unsqueeze(0), samples_tran.unsqueeze(0))
        # d_map2transf, d_transf2map, idx3, idx4 = self.cham_loss(
        #     gt_points.unsqueeze(0), samples_transf.unsqueeze(0))
        loss += (self.w_map2transf * d_map2transf.mean() +
                 self.w_transf2map * d_transf2map.mean())
        return loss

    def getValidLoss(self, gt_points: torch.tensor, samples, ):
        loss = torch.tensor(
            0.0, dtype=torch.float32, device=self.device)  # init loss
        # samples_transf = samples + translations

        # # Scale to metric space
        # samples_tran =samples_transf * scale
        # # samples *= scale
        # # translation *= scale
        # gt_poi = gt_points* scale

        # Chamfer Loss between input and samples+T
        d_map2transf, d_transf2map, idx3, idx4 = self.cham_loss(
            gt_points.unsqueeze(0), samples.unsqueeze(0))
        # d_map2transf, d_transf2map, idx3, idx4 = self.cham_loss(
        #     gt_points.unsqueeze(0), samples_transf.unsqueeze(0))
        loss += (self.w_map2transf * d_map2transf.mean() +
                 self.w_transf2map * d_transf2map.mean())
        return loss

    def evaluate(self, dataloader,
                 load_model=False,
                 best_model=False,
                 reference_points='points',
                 compute_memory=True, # original: False 
                 evaluate=True): # original: False 
        loss_evaluator = evaluator.Evaluator(self.config)
        self.encoder_model.eval()
        self.decoder_model.eval()

        cur_n = 0
        running_eval_loss = 0

        with torch.no_grad():
            if load_model:
                self.loadModel(best=best_model)
                self.encoder_model.to(self.device)
                self.decoder_model.to(self.device)
                print('loaded best:', best_model)
            for i, input_dict in enumerate(tqdm(dataloader)):
                map_idx = input_dict['idx']
                # print('map:', map_idx)
                scale = input_dict['scale'].to(self.device)
                # print('[evaluation] scale shape', scale.shape)
                # print('[evaluation] scale', scale)
                input_dict['features'] = input_dict['features'].to(self.device)
                input_dict['points'] = input_dict['points'].to(self.device)
                
                
                # print('[Evaluation] input points: ', input_dict.copy()['points'])
                # print('[Evaluation] input points shape: ', input_dict.copy()['points'].shape)
                ####### Cast to float16 if necessary #######
                xyz, features, xyz_and_feats = self.encoder_model(input_dict.copy()['points'].unsqueeze(0))
                #out_dict = self.encoder_model(input_dict.copy())
                if self.config['evaluation']['float16']:
                    xyz = xyz.half().float()
                    features = features.half().float()
                ####### Compute  Memory #######
                if compute_memory:
                    nbytes = 2 if self.config['evaluation']['float16'] else 4
                    mem = (xyz.numel() +
                           features.numel())*nbytes
                    loss_evaluator.eval_results['memory'].append(mem)
                    loss_evaluator.eval_results['bpp'].append(
                        mem/input_dict['map'].shape[0]*8)
                ############# Decoder ##################
                xyz, features, intermedia_xyzs = self.decoder_model(xyz, features, xyz_and_feats)
                xyz = xyz.squeeze(0)
                features = features.squeeze(0)
                # Computer the decoder memory 
                nbytes = 2 if self.config['evaluation']['float16'] else 4
                mem = (xyz.numel() +
                        features.numel())*nbytes
                loss_evaluator.eval_results['decoder_memory'].append(mem)

                translation = features[:, :3]
                samples = xyz
                samples_transf = samples+translation

                ###################################
                gt_points = input_dict[reference_points].to(self.device)

                # print('[Evaluation] gt_points: ', gt_points)
                # print('[Evaluation] samples_transf: ', samples_transf)
                # print('[Evaluation] gt_points shape: ', gt_points.shape)

                # Scale to metric space having bug: will product scale value 2times. Find in 220119 by Jokie
                # samples_transf *= scale
                # samples *= scale
                # translation *= scale
                # gt_points *= scale

                # Scale to metric space without self quoting
                samples_tran = samples_transf * scale
                gt_pts = gt_points * scale

                # print('[Evaluation] Scale: ', scale)
                # print('[Evaluation] Scaled gt_points: ', gt_pts)
                # print('[Evaluation] Scaled samples_transf: ', samples_tran)

                reconstruction_error = loss_evaluator.chamferDist(
                    gt_points=gt_pts, source_points=samples_tran)
                loss_evaluator.eval_results['mapwise_reconstruction_error'].append(
                    reconstruction_error.item())

                # debug used only by Jokie 220119
                # print('[evalution] gt self loss: ', self.getValidLoss(gt_points, gt_points))
                # print('[evalution] cur stage loss: ', reconstruction_error.item())
                # print('[evalution] cur stage n: ', cur_n)
                # print('[evalution] cur self.running_loss : ', running_eval_loss)
                # print('[evalution] cur my cal loss: ', running_eval_loss / cur_n)

                if evaluate:  # Full evaluation for testing
                    feat_ind = np.cumsum(
                        [0]+self.config['grid']['feature_dim'])
                    normal_idx = pcu.findList(
                        self.config['grid']['features'], value='normals')
                    normal_idx = (feat_ind[normal_idx], feat_ind[normal_idx+1])
                    gt_normals = input_dict[reference_points +
                                            '_attributes'][:, normal_idx[0]:normal_idx[1]].cuda()
                    loss_evaluator.evaluate(
                        gt_points=gt_points, source_points=samples_transf, gt_normals=gt_normals)

            chamfer_dist = loss_evaluator.getRunningLoss()
            # chamfer_dist = running_eval_loss / cur_n
            # running_eval_loss = 0
            # cur_n = 0


            loss_evaluator.eval_results['reconstruction_error'] = chamfer_dist
            # print('loss_evaluator.eval_results: ', loss_evaluator.eval_results)
            print('mean bpp : ', np.mean(loss_evaluator.eval_results['bpp']))
            print('mean encoder memory (bytes): ', np.mean(loss_evaluator.eval_results['memory']))
            print('mean decoder memory (bytes): ', np.mean(loss_evaluator.eval_results['decoder_memory']))
            print('mean chamfer_dist_abs: ', np.mean(loss_evaluator.eval_results['chamfer_dist_abs']))
            print('mean chamfer_dist_plane: ', np.mean(loss_evaluator.eval_results['chamfer_dist_plane']))
            print('mean iou: ', np.mean(loss_evaluator.eval_results['iou']))
        self.encoder_model.train()
        self.decoder_model.train()

        return loss_evaluator.eval_results

    def encodeDecode(self, input_dict, float_16=True):
        map_idx = input_dict['idx']
        # print('map:', map_idx)
        scale = input_dict['scale']
        input_dict['features'] = input_dict['features'].to(self.device)
        input_dict['points'] = input_dict['points'].to(self.device)

        ####### Cast to float_16 if necessary #######
        xyz, features, xyz_and_feats = self.encoder_model(input_dict.copy()['points'].unsqueeze(0))
        #out_dict = self.encoder_model(input_dict.copy())
        if float_16:
            xyz = xyz.half().float()
            features = features.half().float()
        nr_emb = xyz.shape
        ############# Decoder ##################
        xyz, features, intermedia_xyzs = self.decoder_model(xyz, features, xyz_and_feats)
        xyz = xyz.squeeze(0)
        features = features.squeeze(0)
        translation = features[:, :3]
        samples = xyz
        samples_transf = samples+translation

        samples_trans =  samples_transf * scale
        return samples_trans, nr_emb


if __name__ == "__main__":
    print('Hello')
    parser = argparse.ArgumentParser("./sample_net_trainer.py")
    parser.add_argument(
        '--config', '-cfg',
        type=str,
        required=False,
        default='config/depoco.yaml',
        help='configitecture yaml cfg file. See /config/config for example',
    )
    FLAGS, unparsed = parser.parse_known_args()

    print('passed flags')
    config = yaml.safe_load(open(FLAGS.config, 'r'))
    print('loaded yaml flags')
    trainer = DepocoNetTrainer(config)
    print('initialized  trainer')
    trainer.train(verbose=True)
