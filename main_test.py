import os
import time
import mlflow
import pickle
import numpy as np
from copy import deepcopy

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from parse import *
from dataset import load_cf_data, implicit_CF_dataset, implicit_CF_dataset_test
from evaluation import Evaluator
import modeling.backbone as backbone
import modeling.KD as KD
from utils import seed_all, avg_dict, Logger, Drawer, Var_calc, Var_calcer

def main(args):
    # Dataset
    num_users, num_items, train_pairs, valid_pairs, test_pairs, train_dict, valid_dict, test_dict, train_matrix, user_pop, item_pop = load_cf_data(args.dataset)
    trainset = implicit_CF_dataset(args.dataset, num_users, num_items, train_pairs, train_matrix, train_dict, user_pop, item_pop, args.num_ns, args.neg_sampling_on_all)
    validset = implicit_CF_dataset_test(num_users, num_items, valid_dict)
    testset = implicit_CF_dataset_test(num_users, num_items, test_dict)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

    # Backbone
    all_backbones = [e.lower() for e in dir(backbone)]
    if args.backbone.lower() in all_backbones:
        all_teacher_args, all_student_args = deepcopy(args), deepcopy(args)
        all_teacher_args.__dict__.update(teacher_args.__dict__)
        all_student_args.__dict__.update(student_args.__dict__)
        Teacher = getattr(backbone, dir(backbone)[all_backbones.index(args.backbone.lower())])(trainset, all_teacher_args).cuda()
        Student = getattr(backbone, dir(backbone)[all_backbones.index(args.backbone.lower())])(trainset, all_student_args).cuda()
    else:
        logger.log(f'Invalid backbone {args.backbone}.')
        raise(NotImplementedError, f'Invalid backbone {args.backbone}.')

    if args.model.lower() == "scratch":
        if args.train_teacher:
            model = KD.Scratch(args, Teacher).cuda()
        else:
            model = KD.Scratch(args, Student).cuda()
    else:
        T_path = os.path.join("checkpoints", args.dataset, args.backbone, f"scratch-{teacher_args.embedding_dim}", "BEST_EPOCH.pt")
        Teacher.load_state_dict(torch.load(T_path))
        all_models = [e.lower() for e in dir(KD)]
        if args.model.lower() in all_models:
            model = getattr(KD, dir(KD)[all_models.index(args.model.lower())])(args, Teacher, Student).cuda()
        else:
            logger.log(f'Invalid model {args.model}.')
            raise(NotImplementedError, f'Invalid model {args.model}.')

    # Optimizer
    optimizer = optim.Adam(model.get_params_to_update())

    # Evaluator
    evaluator = Evaluator(args)
    best_model, best_epoch = deepcopy(model.param_to_save), -1
    ckpts = []

    # Test Teacher first
    if args.model.lower() != "scratch":
        logger.log('-' * 40 + "Teacher" + '-' * 40, pre=False)
        tmp_evaluator = Evaluator(args)
        tmp_model = KD.Scratch(args, Teacher).cuda()
        is_improved, early_stop, eval_results, elapsed = tmp_evaluator.evaluate_while_training(tmp_model, -1, train_loader, validset, testset)
        Evaluator.print_final_result(logger, tmp_evaluator.eval_dict)
        logger.log('-' * 88, pre=False)

        if args.draw_teacher:
            ccdf_path = os.path.join("draw_logs", args.dataset, args.backbone, args.model.lower())
            drawer = Drawer(args, ccdf_path)
            drawer.plot_CCDF4negs(tmp_model, train_loader, validset, testset, "teacher-ccdf.png", args.draw_mxK)
            return

    # num_mat = 
        # get distribution of teacher's predictions
    
    # get the variance of users' predictions within each epoch
    model_variance = None
    variance_calculator = None
    if args.model.lower() == "rrdvar" or args.model.lower() == "dcdvar":
        variance_calculator = Var_calc(args, train_loader)  
        model_variance = variance_calculator.get_rating_variance()
        model.set_model_variance(model_variance)
    
    if "rrdvk" in args.model.lower():
        variance_calculator = Var_calcer(args, train_loader, "per_calu_len")
        model_variance, _ = variance_calculator.get_rating_variance()
        item_idx = model.item_idx_init()
        model.set_model_variance(model_variance, item_idx)
        # variance_calculator.update_rating_variance(model, 0) 
    
    ccdf_path = os.path.join("draw_logs", args.dataset, args.backbone, args.model.lower())
    drawer = None
    if args.draw_student:
        drawer = Drawer(args, ccdf_path)


    false_neg_val_list = []
    # x_axis = []
    x_axis = [0]
    neg_hard_val_list = []
    for i in range(args.D_num):
        neg_hard_val_list.append([])

    false_neg_sum, false_neg_sum_2 = 0, 0
    false_neg_variance = [0]

    neg_hard_sum, neg_hard_sum_2 = [], []
    neg_hard_variance = []
    for i in range(args.D_num):
        neg_hard_sum.append(0)
        neg_hard_sum_2.append(0)
        neg_hard_variance.append([0])




    for epoch in range(args.epochs):
        logger.log(f'Epoch [{epoch + 1}/{args.epochs}]')
        tic1 = time.time()
        logger.log('Negative sampling...')
        train_loader.dataset.negative_sampling()

        logger.log("Model's personal time...")
        model.do_something_in_each_epoch(epoch)

        epoch_loss, epoch_base_loss, epoch_kd_loss = [], [], []
        logger.log('Training...')
        
        for idx, (batch_user, batch_pos_item, batch_neg_item) in enumerate(train_loader):
            batch_user = batch_user.cuda()      # batch_size
            batch_pos_item = batch_pos_item.cuda()  # batch_size
            batch_neg_item = batch_neg_item.cuda()  # batch_size, num_ns
            
            # Forward Pass
            model.train()
            loss, base_loss, kd_loss = model(batch_user, batch_pos_item, batch_neg_item)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache() # clear memory
            epoch_loss.append(loss.detach())
            epoch_base_loss.append(base_loss)
            epoch_kd_loss.append(kd_loss)

        epoch_loss = torch.mean(torch.stack(epoch_loss)).item()
        epoch_base_loss = torch.mean(torch.stack(epoch_base_loss)).item()
        epoch_kd_loss = torch.mean(torch.stack(epoch_kd_loss)).item()

        toc1 = time.time()

        false_neg = model.get_false_neg()
        neg_hard = model.get_neg_hard()

        user_idx = torch.arange(num_users)

        # if epoch % 25 == 0 and args.draw_student:
        #     neg_hard_val = model.student.get_user_item_ratings(user_idx, neg_hard.cuda())
        #     neg_hard_val = neg_hard_val.view(-1, args.D_num, args.D_size) # num_users X D_num X D_size
        #     neg_hard_val = neg_hard_val.mean(dim = 0)
        #     neg_hard_val = neg_hard_val.mean(dim = -1) # D_num
        #     for i in range(args.D_num):
        #         neg_hard_val_list[i].append(neg_hard_val[i].item())

        #     false_neg_val = model.student.get_user_item_ratings(user_idx, false_neg.cuda()) # num_users X false_neg_num
        #     false_neg_val = false_neg_val.mean()
        #     false_neg_val_list.append(false_neg_val.item())

        #     x_axis.append(epoch)
        step = 1
        if epoch % step == 0 and args.draw_student:
            neg_hard_val = model.student.get_user_item_ratings(user_idx, neg_hard.cuda())
            # neg_hard_val = neg_hard_val.view(-1, args.D_num, args.D_size) # num_users X D_num X D_size
            neg_hard_val = neg_hard_val.mean(dim = 0)
            # neg_hard_val = neg_hard_val.mean(dim = -1) # D_num
            for i in range(args.D_num):
                neg_hard_val_list[i].append(neg_hard_val[i].item())

            false_neg_val = model.student.get_user_item_ratings(user_idx, false_neg.cuda()) # num_users X false_neg_num
            false_neg_val = false_neg_val.mean()
            false_neg_val_list.append(false_neg_val.item())


            if epoch != 0:
                epoch_time = epoch // step

                false_neg_sum += false_neg_val.item()
                false_neg_sum_2 += false_neg_val.item() ** 2

                for i in range(args.D_num):
                    neg_hard_sum[i] += neg_hard_val[i].item()
                    neg_hard_sum_2[i] += neg_hard_val[i].item() ** 2
                    
                if epoch % 5 == 0:
                    variance = false_neg_sum_2 / epoch_time - (false_neg_sum / epoch_time) ** 2
                    false_neg_variance.append(variance)
                    false_neg_sum_2, false_neg_sum = 0, 0 # reset
                    for i in range(args.D_num):
                        variance = neg_hard_sum_2[i] / epoch_time - (neg_hard_sum[i] / epoch_time) ** 2
                        neg_hard_variance[i].append(variance)
                        neg_hard_sum_2[i], neg_hard_sum[i] = 0, 0 # reset
                    x_axis.append(epoch)

        # evaluation
        if epoch % args.eval_period == 0:
            logger.log("Evaluating...")
            is_improved, early_stop, eval_results, elapsed = evaluator.evaluate_while_training(model, epoch, train_loader, validset, testset)
            evaluator.print_result_while_training(logger, epoch_loss, epoch_base_loss, epoch_kd_loss, eval_results, is_improved=is_improved, train_time=toc1-tic1, test_time=elapsed)
            if early_stop:
                break
            if is_improved:
                best_model = deepcopy(model.param_to_save)
                best_epoch = epoch
        
        if epoch in [10, 30, 50, 100, 200, 500] and epoch != 0 and args.draw_student and args.draw_type.lower() == "ccdf":
            drawer.plot_CCDF4negs(model, train_loader, validset, testset, "epoch {}".format(epoch), args.draw_mxK)
        
        # save intermediate checkpoints
        if not args.no_save and args.ckpt_interval != -1 and epoch % args.ckpt_interval == 0 and epoch != 0:
            ckpts.append(deepcopy(model.param_to_save))
    label_num_list = [1, 2, 4, 8]
    if args.draw_student:
        
        # Y_list = []
        # for i in label_num_list:
        #     Y_list.append(neg_hard_val_list[i-1])
        # Y_list.append(false_neg_val_list)
        label_list = []
        for i in label_num_list:
            label_list.append(f"hard_neg, D={i}")
        label_list.append("false_neg")

        # drawer.plot_multi_curves("Epoch", "mean scores", x_axis, Y_list, label_list, f"student-{args.draw_type.lower()}-{args.draw_name}.{args.savetype}", args.savetype)


        variance_list = []
        for i in label_num_list:
            variance_list.append(neg_hard_variance[i-1])
        variance_list.append(false_neg_variance)
        drawer.plot_multi_curves("Epoch", "variance", x_axis, variance_list, label_list, f"student-{args.draw_type.lower()}-variance_with_timing-{args.draw_name}.{args.savetype}", args.savetype)

        # drawer.plot_all_subfig(f"student-{args.draw_type.lower()}-{args.draw_name}.{args.savetype}", args.x_name, args.y_name, args.savetype)
    eval_dict = evaluator.eval_dict
    Evaluator.print_final_result(logger, eval_dict)
    Evaluator.print_final_result(ans_logger, eval_dict)


    if not args.no_save:
        embedding_dim = Teacher.embedding_dim if args.train_teacher else Student.embedding_dim
        save_dir = os.path.join("checkpoints", args.dataset, args.backbone, f"{args.model.lower()}-{embedding_dim}")
        os.makedirs(save_dir, exist_ok=True)
        torch.save(best_model, os.path.join(save_dir, "BEST_EPOCH.pt"))
        for idx, ckpt in enumerate(ckpts):
            if (idx + 1) * args.ckpt_interval >= best_epoch:
                break
            torch.save(ckpt, os.path.join(save_dir, f"EPOCH_{(idx + 1) * args.ckpt_interval}.pt"))

    return eval_dict


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    logger = Logger(args, args.no_log)
    ans_logger = Logger(args, args.no_log, True)

    if args.run_all:
        args_copy = deepcopy(args)
        eval_dicts = []

        for seed in range(5):
            args = deepcopy(args_copy)
            args.seed = seed
            seed_all(args.seed)
            logger.log_args(teacher_args, "TEACHER")
            if not args.train_teacher:
                logger.log_args(student_args, "STUDENT")
            logger.log_args(args, "ARGUMENTS")
            eval_dicts.append(main(args))
        
        avg_eval_dict = avg_dict(eval_dicts)

        logger.log('=' * 60)
        Evaluator.print_final_result(logger, avg_eval_dict, prefix="avg ")
    else:
        ans_logger.prelog()
        logger.log_args(teacher_args, "TEACHER")
        ans_logger.log_args(teacher_args, "TEACHER")
        if not args.train_teacher:
            logger.log_args(student_args, "STUDENT")
            ans_logger.log_args(student_args, "STUDENT")
        logger.log_args(args, "ARGUMENTS")
        ans_logger.log_args(args, "ARGUMENTS")
        seed_all(args.seed)
        main(args)
