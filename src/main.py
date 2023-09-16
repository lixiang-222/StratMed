import argparse
import math
import time
from collections import defaultdict

import dill
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from models import AdvancedModel, BasicModel
from util import llprint, multi_label_metric, ddi_rate_score, Regularization, result_output

torch.manual_seed(1203)
np.random.seed(2048)

# Training settings
parser = argparse.ArgumentParser()
# 模式参数
parser.add_argument("--debug", default=False, help="debug mode")
parser.add_argument("--Test", action="store_true", default=False, help="test mode")
parser.add_argument("--pretrain", default=False, help="re-pretrain this time")

# 设备参数
parser.add_argument("--cuda", type=int, default=1, help="which cuda")
parser.add_argument("--resume_path_trained", default="trained/trained_model.pt", help="trained model")
parser.add_argument("--resume_path_pretrained", default="trained/pretrained_model.pt", help="pretrained model")

# 训练参数
parser.add_argument("--lr", type=float, default=5e-4, help="learning rate")
parser.add_argument("--regular", type=float, default=0.005, help="regularization parameter")
parser.add_argument("--target_ddi", type=float, default=0.06, help="target ddi")
parser.add_argument("--kp", type=float, default=0.05, help="coefficient of P signal")
parser.add_argument("--dim", type=int, default=64, help="dimension")

# 分层参数
parser.add_argument("--bucket_grad", type=int, default=2, help="grad in all bucket")
parser.add_argument("--med_bucket_top", type=int, default=60, help="top layer number in med-med bucket")
parser.add_argument("--diag_bucket_top", type=int, default=150, help="top layer number in diag-med bucket")
parser.add_argument("--proc_bucket_top", type=int, default=150, help="top layer number in proc-med bucket")
parser.add_argument("--med_max_weight", type=float, default=0.8, help="the relevance weight in med-med graph")
parser.add_argument("--diag_max_weight", type=float, default=0.6, help="the relevance weight in diag-med graph")
parser.add_argument("--proc_max_weight", type=float, default=0.6, help="the relevance weight in proc-med graph")

args = parser.parse_args()


# evaluate
def eval(model, data_eval, voc_size, epoch, device):
    model.eval()

    smm_record = []
    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]

    """自己加入一些loss来看"""
    loss_bce, loss_multi, loss = [[] for _ in range(3)]

    med_cnt, visit_cnt = 0, 0

    for step, input in enumerate(data_eval):
        y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []
        for adm_idx, adm in enumerate(input):
            target_output, _ = model(input[: adm_idx + 1])

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
                loss1 = 0.95 * loss_bce1.item() + 0.05 * loss_multi1.item()

            loss_bce.append(loss_bce1)
            loss_multi.append(loss_multi1)
            loss.append(loss1)
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

    # """列表转np"""
    # loss_multi = np.array(loss_multi)
    # loss_bce = np.array(loss_bce)
    # loss = np.array(loss)

    llprint(
        "\nDDI Rate: {:.4}, Jaccard: {:.4},  PRAUC: {:.4}, AVG_PRC: {:.4}, AVG_RECALL: {:.4}, AVG_F1: {:.4},"
        "AVG_Loss: {:.4}, AVG_MED: {:.4}\n".format(
            ddi_rate,
            np.mean(ja),
            np.mean(prauc),
            np.mean(avg_p),
            np.mean(avg_r),
            np.mean(avg_f1),
            np.mean(loss),
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
        np.mean(loss),
        med_cnt / visit_cnt,
    )


def pre_training(data_train, data_eval, voc_size, ddi_adj, ddi_mask_H, device):
    pretrained_model = BasicModel(
        voc_size,
        ddi_adj,
        ddi_mask_H,
        emb_dim=args.dim,
        device=device,
    )
    pretrained_model.to(device=device)

    if not args.pretrain or args.Test:
        pretrained_model.load_state_dict(torch.load(args.resume_path_pretrained, map_location=device))
        return pretrained_model

    else:
        regular = Regularization(pretrained_model, args.regular, p=0)  # 正则化模型

        optimizer = Adam(list(pretrained_model.parameters()), lr=args.lr)

        # start iterations
        best_epoch, best_ja = 0, 0
        best_model = None

        EPOCH = 20
        for epoch in range(EPOCH):
            tic = time.time()
            print("\nepoch {} --------------------------".format(epoch))

            pretrained_model.train()
            for step, patient in enumerate(data_train):

                loss = 0
                for idx, adm in enumerate(patient):

                    seq_input = patient[: idx + 1]
                    loss_bce_target = np.zeros((1, voc_size[2]))
                    loss_bce_target[:, adm[2]] = 1

                    loss_multi_target = np.full((1, voc_size[2]), -1)
                    for idx, item in enumerate(adm[2]):
                        loss_multi_target[0][idx] = item

                    result, loss_ddi = pretrained_model(seq_input)

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

                    loss += regular(pretrained_model)  # 正则化

                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step()

                llprint("\rtraining step: {} / {}".format(step, len(data_train)))
            print()
            tic2 = time.time()
            print("\n验证集结果：")
            ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_loss, avg_med = eval(
                pretrained_model, data_eval, voc_size, epoch, device
            )
            print(
                "training time: {}, test time: {}".format(
                    time.time() - tic, time.time() - tic2
                )
            )

            if epoch != 0:
                if best_ja < ja:
                    best_epoch = epoch
                    best_ja = ja
                    best_model = pretrained_model
                print(
                    "best_epoch: {}, best_ja: {:.4}".format(best_epoch, best_ja))
        #
        torch.save(best_model.state_dict(), args.resume_path_pretrained)
        return best_model


def relevance_mining(records, voc_size):
    if args.debug:
        return dill.load(open("../data/output/ehr_adj.pkl", "rb"))

    # 药物-所有的ehr
    ehr_adj = np.zeros((voc_size[0] + voc_size[1] + voc_size[2], voc_size[0] + voc_size[1] + voc_size[2]))
    ehr_adj_med_diag = np.zeros((voc_size[2], voc_size[0]))
    ehr_adj_med_proc = np.zeros((voc_size[2], voc_size[1]))
    ehr_adj_med_med = np.zeros((voc_size[2], voc_size[2]))

    for patient in records:
        for adm in patient:
            for i in range(len(adm[2])):
                for j in range(len(adm[0])):
                    node1 = adm[2][i]
                    node2 = adm[0][j]
                    ehr_adj_med_diag[node1, node2] += 1
                for k in range(len(adm[1])):
                    node1 = adm[2][i]
                    node2 = adm[1][k]
                    ehr_adj_med_proc[node1, node2] += 1
                for l in range(i + 1, len(adm[2])):
                    node1 = adm[2][i]
                    node2 = adm[2][l]
                    ehr_adj_med_med[node1, node2] += 1
                    ehr_adj_med_med[node2, node1] += 1

    def calculate_layers(a, k, amount):
        n = math.floor(math.log((amount / a - 1) / (k - 1), k))
        return n

    def stratification(data, a, k, amount):
        sorted_values = np.sort(data.flatten())
        # 计算位置
        index_of_5 = np.where(sorted_values == 5)[0][0]
        index_list = [index_of_5]
        layer = calculate_layers(a, k, amount - index_of_5 - 1)

        for i in range(layer):
            index_list.append(amount - a * (k ** i))
        index_list = sorted(index_list)
        # 获取阈值
        threshold = sorted_values[index_list]
        # 分层
        bucket = np.digitize(data, threshold)
        return bucket, layer + 2

    # 初始个数与梯度
    k = args.bucket_grad
    med_med_bucket, med_layer = stratification(ehr_adj_med_med, args.med_bucket_top, k, voc_size[2] * voc_size[2])
    med_diag_bucket, diag_layer = stratification(ehr_adj_med_diag, args.diag_bucket_top, k, voc_size[0] * voc_size[2])
    med_proc_bucket, proc_layer = stratification(ehr_adj_med_proc, args.proc_bucket_top, k, voc_size[1] * voc_size[2])

    print("med-med")
    for i in range(med_layer):
        print(i, np.count_nonzero(med_med_bucket == i))

    print("med-diag")
    for i in range(diag_layer):
        print(i, np.count_nonzero(med_diag_bucket == i))

    print("med-proc")
    for i in range(proc_layer):
        print(i, np.count_nonzero(med_proc_bucket == i))

    for med, line in enumerate(med_diag_bucket):
        for diag, weight in enumerate(line):
            ehr_adj[diag][med + voc_size[0] + voc_size[1]] = weight
            ehr_adj[med + voc_size[0] + voc_size[1]][diag] = weight

    for med, line in enumerate(med_proc_bucket):
        for proc, weight in enumerate(line):
            ehr_adj[proc + voc_size[0]][med + voc_size[0] + voc_size[1]] = weight
            ehr_adj[med + voc_size[0] + voc_size[1]][proc + voc_size[0]] = weight

    for med, line in enumerate(med_med_bucket):
        for med2, weight in enumerate(line):
            ehr_adj[med + voc_size[0] + voc_size[1]][med2 + voc_size[0] + voc_size[1]] = weight
            ehr_adj[med2 + voc_size[0] + voc_size[1]][med + voc_size[0] + voc_size[1]] = weight

    layers = [diag_layer, proc_layer, med_layer]

    dill.dump([ehr_adj, layers], open('../data/output/ehr_adj.pkl', 'wb'))

    return ehr_adj, layers


def main():
    # load data
    data_path = "../data/output/records_final.pkl"
    voc_path = "../data/output/voc_final.pkl"

    ddi_adj_path = "../data/output/ddi_A_final.pkl"
    ddi_mask_path = "../data/output/ddi_mask_H.pkl"
    # ehr_adj_path = "../data/output/ehr_adj_final.pkl"

    if torch.cuda.is_available():
        device = torch.device("cuda:{}".format(args.cuda))
    else:
        device = torch.device("cpu")

    ddi_adj = dill.load(open(ddi_adj_path, "rb"))
    ddi_mask_H = dill.load(open(ddi_mask_path, "rb"))
    data = dill.load(open(data_path, "rb"))
    if args.debug:
        data = data[:5]

    voc = dill.load(open(voc_path, "rb"))
    diag_voc, pro_voc, med_voc = voc["diag_voc"], voc["pro_voc"], voc["med_voc"]

    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point: split_point + eval_len]
    data_eval = data[split_point + eval_len:]

    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))

    print("*********pre-training*********")
    pretrained_model = pre_training(data_train, data_eval, voc_size, ddi_adj, ddi_mask_H, device)

    print("*********relevance mining*********")
    ehr_adj, bucket_layer = relevance_mining(data_train, voc_size)

    model = AdvancedModel(
        voc_size,
        ddi_adj,
        ddi_mask_H,
        ehr_adj,
        bucket_layer,
        emb_dim=args.dim,
        device=device,
        pretrained_embeddings=pretrained_model.embeddings
    )

    if args.Test:
        model.load_state_dict(torch.load(open(args.resume_path_trained, 'rb'), map_location=device))
        tic = time.time()

        ddi_list, ja_list, prauc_list, f1_list, med_list = [], [], [], [], []
        result = []
        print()
        print(" ----- 10 rounds of bootstrapping test ----- ")
        for _ in range(10):
            test_sample = np.random.choice(data_eval, size=round(len(data_eval) * 0.8), replace=True)
            ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, _, avg_med = eval(model, test_sample, voc_size, 0, device)
            result.append([ddi_rate, ja, avg_f1, prauc, avg_med])

        result = np.array(result)
        mean = result.mean(axis=0)
        std = result.std(axis=0)

        outstring = ""
        for m, s in zip(mean, std):
            outstring += "{:.4f} $\pm$ {:.4f} & ".format(m, s)

        print()
        print(" ----- final (mean $\pm$ std) ----- ")
        print(outstring)

        print('test time: {}'.format(time.time() - tic))
        return

    print("***********training*********")

    model.to(device=device)

    regular = Regularization(model, args.regular, p=0)  # 正则化模型

    # print('parameters', get_n_params(model))
    # exit()
    optimizer = Adam(list(model.parameters()), lr=args.lr)

    # start iterations
    history = defaultdict(list)
    history_on_train = defaultdict(list)
    best = {"epoch": 0, "ja": 0, "ddi": 0, "prauc": 0, "f1": 0, "med": 0, 'model': None}

    EPOCH = 15
    if args.debug:
        EPOCH = 5
    for epoch in range(EPOCH):
        tic = time.time()
        print("\nepoch {} --------------------------".format(epoch))

        model.train()
        for step, patient in enumerate(data_train):

            loss = 0
            for idx, adm in enumerate(patient):

                seq_input = patient[: idx + 1]
                loss_bce_target = np.zeros((1, voc_size[2]))
                loss_bce_target[:, adm[2]] = 1

                loss_multi_target = np.full((1, voc_size[2]), -1)
                for idx, item in enumerate(adm[2]):
                    loss_multi_target[0][idx] = item

                result, loss_ddi = model(seq_input)

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

                loss += regular(model)  # 跟上面配套

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

            llprint("\rtraining step: {} / {}".format(step, len(data_train)))
        print()
        tic2 = time.time()
        print("\n验证集结果：")
        ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_loss, avg_med = eval(
            model, data_eval, voc_size, epoch, device
        )
        print("\n训练集结果：")
        _, ja_on_train, _, _, _, _, avg_loss_on_train, _ = eval(model, data_train, voc_size, epoch, device)

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
        history['loss'].append(avg_loss)

        history_on_train["ja"].append(ja_on_train)
        history_on_train["loss"].append(avg_loss_on_train)

        if epoch != 0:
            if best['ja'] < ja:
                best['epoch'] = epoch
                best['ja'] = ja
                best['model'] = model
                best['ddi'] = ddi_rate
                best['prauc'] = prauc
                best['f1'] = avg_f1
                best['med'] = avg_med
            print(
                "best_epoch: {}, best_ja: {:.4}".format(best['epoch'], best['ja']))

    torch.save(best['model'].state_dict(), "results/trained_model_ja{:.4}.pt".format(best['ja']))

    result_output(history, history_on_train, best, regular)


if __name__ == "__main__":
    main()
