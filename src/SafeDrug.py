import argparse
import os
import time
from collections import defaultdict

import dill
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from models import SafeDrugModel
from util import llprint, multi_label_metric, ddi_rate_score

torch.manual_seed(1203)
np.random.seed(2048)
# torch.set_num_threads(30)
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# setting
model_name = "SafeDrug"
# resume_path = 'saved/{}/Epoch_49_TARGET_0.06_JA_0.5183_DDI_0.05854.model'.format(model_name)
resume_path = "Epoch_49_TARGET_0.06_JA_0.5183_DDI_0.05854.model"

if not os.path.exists(os.path.join("saved", model_name)):
    os.makedirs(os.path.join("saved", model_name))

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument("--Test", action="store_true", default=False, help="test mode")
parser.add_argument("--model_name", type=str, default=model_name, help="model name")
parser.add_argument("--resume_path", type=str, default=resume_path, help="resume path")
parser.add_argument("--lr", type=float, default=5e-4, help="learning rate")
parser.add_argument("--target_ddi", type=float, default=0.06, help="target ddi")
parser.add_argument("--kp", type=float, default=0.05, help="coefficient of P signal")
parser.add_argument("--dim", type=int, default=64, help="dimension")
parser.add_argument("--cuda", type=int, default=1, help="which cuda")

args = parser.parse_args()


# evaluate
def eval(model, data_eval, voc_size, epoch, med_kg_voc, diag_kg_voc, kg_edge, device):
    model.eval()

    smm_record = []
    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    """自己加入一些loss来看"""
    loss_bce, loss_multi = [[] for _ in range(2)]

    med_cnt, visit_cnt = 0, 0

    for step, input in enumerate(data_eval):
        y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []
        for adm_idx, adm in enumerate(input):
            target_output, _ = model(input[: adm_idx + 1], med_kg_voc, diag_kg_voc, kg_edge)

            """自己加的loss，在输出时候用来看loss的改变，不训练"""
            loss_bce_target = np.zeros((1, voc_size[2]))
            loss_bce_target[:, adm[2]] = 1

            loss_multi_target = np.full((1, voc_size[2]), -1)
            for idx, item in enumerate(adm[2]):
                loss_multi_target[0][idx] = item

            with torch.no_grad():
                loss_bce1 = F.binary_cross_entropy_with_logits(
                    target_output, torch.FloatTensor(loss_bce_target).to(device)
                ).cpu()
                loss_multi1 = F.multilabel_margin_loss(
                    F.sigmoid(target_output), torch.LongTensor(loss_multi_target).to(device)
                ).cpu()

            loss_bce.append(loss_bce1)
            loss_multi.append(loss_multi1)
            """"""

            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[adm[2]] = 1
            y_gt.append(y_gt_tmp)

            # prediction prod
            target_output = F.sigmoid(target_output).detach().cpu().numpy()[0]
            y_pred_prob.append(target_output)

            # prediction med set
            y_pred_tmp = target_output.copy()
            y_pred_tmp[y_pred_tmp >= 0.5] = 1
            y_pred_tmp[y_pred_tmp < 0.5] = 0
            y_pred.append(y_pred_tmp)

            # prediction label
            y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
            y_pred_label.append(sorted(y_pred_label_tmp))
            visit_cnt += 1
            med_cnt += len(y_pred_label_tmp)

        smm_record.append(y_pred_label)
        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(
            np.array(y_gt), np.array(y_pred), np.array(y_pred_prob)
        )

        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)
        llprint("\rtest step: {} / {}".format(step, len(data_eval)))

    # ddi rate
    ddi_rate = ddi_rate_score(smm_record, path="../data/output/ddi_A_final.pkl")

    """列表转np"""
    loss_multi = np.array(loss_multi)
    loss_bce = np.array(loss_bce)

    llprint(
        "\nDDI Rate: {:.4}, Jaccard: {:.4},  PRAUC: {:.4}, AVG_PRC: {:.4}, AVG_RECALL: {:.4}, AVG_F1: {:.4},"
        "AVG_Lmulti: {:.4}, AVG_Lbce: {:.4}, AVG_MED: {:.4}\n".format(
            ddi_rate,
            np.mean(ja),
            np.mean(prauc),
            np.mean(avg_p),
            np.mean(avg_r),
            np.mean(avg_f1),
            np.mean(loss_multi),
            np.mean(loss_bce),
            med_cnt / visit_cnt,
        )
    )

    return (
        ddi_rate,
        np.mean(ja),
        np.mean(prauc),
        np.mean(avg_p),
        np.mean(avg_r),
        np.mean(avg_f1),
        med_cnt / visit_cnt,
    )


def main():
    # load data
    data_path = "../data/output/records_final.pkl"
    voc_path = "../data/output/voc_final.pkl"

    ddi_adj_path = "../data/output/ddi_A_final.pkl"
    ddi_mask_path = "../data/output/ddi_mask_H.pkl"
    molecule_path = "../data/output/atc3toSMILES.pkl"

    device = torch.device("cpu")
    # device = torch.device("cuda")

    ddi_adj = dill.load(open(ddi_adj_path, "rb"))
    ddi_mask_H = dill.load(open(ddi_mask_path, "rb"))
    data = dill.load(open(data_path, "rb"))
    molecule = dill.load(open(molecule_path, "rb"))

    voc = dill.load(open(voc_path, "rb"))
    diag_voc, pro_voc, med_voc = voc["diag_voc"], voc["pro_voc"], voc["med_voc"]

    """自己写的新字典"""
    basic_info_voc_path = "../data/output/basic_info_voc_final.pkl"
    voc2 = dill.load(open(basic_info_voc_path, "rb"))
    marital_voc, ethnicity_voc, gender_voc, expire_voc, age_voc = voc2["marital_voc"], voc2["ethnicity_voc"], voc2[
        "gender_voc"], voc2["expire_voc"], voc2["age_voc"]
    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word), len(marital_voc.idx2word),
                len(ethnicity_voc.idx2word), len(gender_voc.idx2word), len(expire_voc.idx2word),
                len(age_voc.idx2word))
    """"""

    """自己写的药物和图谱中药物的映射"""
    med_kg_voc = dill.load(open("../data/output/med_map.pkl", "rb"))
    diag_kg_voc = dill.load(open("../data/output/diseases_map.pkl", "rb"))
    kg_edge = dill.load(open("../data/output/new_kg_edges.pkl", "rb"))
    """"""
    data = data[:5]  # 本地电脑上测试用5个数据

    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point: split_point + eval_len]
    data_eval = data[split_point + eval_len:]

    # MPNNSet, N_fingerprint, average_projection = buildMPNN(
    #     molecule, med_voc.idx2word, 2, device
    # )

    model = SafeDrugModel(
        voc_size,
        ddi_adj,
        ddi_mask_H,
        emb_dim=args.dim,
        device=device,
    )
    # model.load_state_dict(torch.load(open(args.resume_path, 'rb')))

    if args.Test:
        model.load_state_dict(torch.load(open(args.resume_path, "rb")))
        model.to(device=device)
        tic = time.time()

        ddi_list, ja_list, prauc_list, f1_list, med_list = [], [], [], [], []
        # ###
        # for threshold in np.linspace(0.00, 0.20, 30):
        #     print ('threshold = {}'.format(threshold))
        #     ddi, ja, prauc, _, _, f1, avg_med = eval(model, data_test, voc_size, 0, threshold)
        #     ddi_list.append(ddi)
        #     ja_list.append(ja)
        #     prauc_list.append(prauc)
        #     f1_list.append(f1)
        #     med_list.append(avg_med)
        # total = [ddi_list, ja_list, prauc_list, f1_list, med_list]
        # with open('ablation_ddi.pkl', 'wb') as infile:
        #     dill.dump(total, infile)
        # ###
        result = []
        for _ in range(10):
            test_sample = np.random.choice(
                data_test, round(len(data_test) * 0.8), replace=True
            )
            ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = eval(
                model, data_eval, voc_size, 0, med_kg_voc, diag_kg_voc, kg_edge, device
            )
            result.append([ddi_rate, ja, avg_f1, prauc, avg_med])

        result = np.array(result)
        mean = result.mean(axis=0)
        std = result.std(axis=0)

        outstring = ""
        for m, s in zip(mean, std):
            outstring += "{:.4f} $\pm$ {:.4f} & ".format(m, s)

        print(outstring)

        print("test time: {}".format(time.time() - tic))
        return

    model.to(device=device)
    # print('parameters', get_n_params(model))
    # exit()
    optimizer = Adam(list(model.parameters()), lr=args.lr)

    # start iterations
    history = defaultdict(list)
    best_epoch, best_ja = 0, 0

    # EPOCH = 50
    EPOCH = 3  # 测试用3个epoch
    for epoch in range(EPOCH):
        tic = time.time()
        print("\nepoch {} --------------------------".format(epoch + 1))

        model.train()
        for step, input in enumerate(data_train):

            loss = 0
            for idx, adm in enumerate(input):

                seq_input = input[: idx + 1]
                loss_bce_target = np.zeros((1, voc_size[2]))
                loss_bce_target[:, adm[2]] = 1

                loss_multi_target = np.full((1, voc_size[2]), -1)
                for idx, item in enumerate(adm[2]):
                    loss_multi_target[0][idx] = item

                result, loss_ddi = model(seq_input, med_kg_voc, diag_kg_voc, kg_edge)

                loss_bce = F.binary_cross_entropy_with_logits(
                    result, torch.FloatTensor(loss_bce_target).to(device)
                )
                loss_multi = F.multilabel_margin_loss(
                    F.sigmoid(result), torch.LongTensor(loss_multi_target).to(device)
                )

                result = F.sigmoid(result).detach().cpu().numpy()[0]
                result[result >= 0.5] = 1
                result[result < 0.5] = 0
                y_label = np.where(result == 1)[0]
                current_ddi_rate = ddi_rate_score(
                    [[y_label]], path="../data/output/ddi_A_final.pkl"
                )

                if current_ddi_rate <= args.target_ddi:
                    loss = 0.95 * loss_bce + 0.05 * loss_multi
                else:
                    beta = min(0, 1 + (args.target_ddi - current_ddi_rate) / args.kp)
                    loss = (
                            beta * (0.95 * loss_bce + 0.05 * loss_multi)
                            + (1 - beta) * loss_ddi
                    )

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

            llprint("\rtraining step: {} / {}".format(step, len(data_train)))

        print()
        tic2 = time.time()
        ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = eval(
            model, data_eval, voc_size, epoch, med_kg_voc, diag_kg_voc, kg_edge, device
        )
        print(
            "training time: {}, test time: {}".format(
                time.time() - tic, time.time() - tic2
            )
        )

        history["ja"].append(ja)
        history["ddi_rate"].append(ddi_rate)
        history["avg_p"].append(avg_p)
        history["avg_r"].append(avg_r)
        history["avg_f1"].append(avg_f1)
        history["prauc"].append(prauc)
        history["med"].append(avg_med)

        if epoch >= 5:
            print(
                "ddi: {}, Med: {}, Ja: {}, F1: {}, PRAUC: {}".format(
                    np.mean(history["ddi_rate"][-5:]),
                    np.mean(history["med"][-5:]),
                    np.mean(history["ja"][-5:]),
                    np.mean(history["avg_f1"][-5:]),
                    np.mean(history["prauc"][-5:]),
                )
            )

        torch.save(
            model.state_dict(),
            open(
                os.path.join(
                    "saved",
                    args.model_name,
                    "Epoch_{}_TARGET_{:.2}_JA_{:.4}_DDI_{:.4}.model".format(
                        epoch, args.target_ddi, ja, ddi_rate
                    ),
                ),
                "wb",
            ),
        )

        if epoch != 0 and best_ja < ja:
            best_epoch = epoch
            best_ja = ja

        print("best_epoch: {}".format(best_epoch + 1))

    dill.dump(
        history,
        open(
            os.path.join(
                "saved", args.model_name, "history_{}.pkl".format(args.model_name)
            ),
            "wb",
        ),
    )


if __name__ == "__main__":
    main()
