import argparse
import dill
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch
import torch.nn.functional as F
from collections import defaultdict
from torch.optim import Adam

from models import BasicModel
from util import llprint, multi_label_metric, ddi_rate_score, Regularization

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


def main():
    # load data
    data_path = "../data/output/records_final.pkl"
    voc_path = "../data/output/voc_final.pkl"

    ddi_adj_path = "../data/output/ddi_A_final.pkl"
    ddi_mask_path = "../data/output/ddi_mask_H.pkl"

    # device = torch.device("cuda:{}".format(args.cuda))
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    ddi_adj = dill.load(open(ddi_adj_path, "rb"))
    ddi_mask_H = dill.load(open(ddi_mask_path, "rb"))
    data = dill.load(open(data_path, "rb"))
    # data = data[:5]

    voc = dill.load(open(voc_path, "rb"))
    diag_voc, pro_voc, med_voc = voc["diag_voc"], voc["pro_voc"], voc["med_voc"]

    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point: split_point + eval_len]
    data_eval = data[split_point + eval_len:]

    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))

    model = BasicModel(
        voc_size,
        ddi_adj,
        ddi_mask_H,
        emb_dim=args.dim,
        device=device,
    )

    model.to(device=device)

    # regular = Regularization(model, 0.01, p=0)  # L2正则化模型

    optimizer = Adam(list(model.parameters()), lr=args.lr)

    # start iterations
    history = defaultdict(list)
    history_ja_on_train = defaultdict(list)
    best_epoch, best_ja = 0, 0

    EPOCH = 15
    # EPOCH = 5
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
                # loss += regular(model)  # 跟上面配套
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

        history_ja_on_train["ja"].append(ja_on_train)
        history_ja_on_train["loss"].append(avg_loss_on_train)

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

        if epoch != 0:
            if best_ja < ja:
                best_epoch = epoch
                best_ja = ja
                torch.save(
                    model.state_dict(),
                    open(
                        os.path.join(
                            "saved",
                            args.model_name,
                            "model_ja_{:.4}.pt".format(best_ja)
                        ),
                        "wb",
                    ),
                )
            print(
                "best_epoch: {}, best_ja: {:.4}".format(best_epoch, best_ja))

    # 画ja的图
    max_eval_ja, max_eval_ja_index = max(history["ja"]), history["ja"].index(max(history["ja"]))
    max_train_ja, max_train_ja_index = max(history_ja_on_train["ja"]), history_ja_on_train["ja"].index(
        max(history_ja_on_train["ja"]))

    # 创建一个新的图形
    plt.figure()

    # 绘制每个列表的线
    plt.plot(history["ja"], label="eval")
    plt.plot(history_ja_on_train["ja"], label="train")

    # 在每个最大值的节点旁边标注其对应的值
    plt.text(max_eval_ja_index, max_eval_ja, str(max_eval_ja), color='red', ha='center', va='bottom')
    plt.text(max_train_ja_index, max_train_ja, str(max_train_ja), color='red', ha='center', va='bottom')

    # 添加标题和标签
    plt.title("OverFitting-ja")
    plt.xlabel('epoch')
    plt.ylabel('ja_value')
    # 显示图例
    plt.legend()

    # 做的loss系列
    # 创建一个新的图形
    plt.figure()

    # 绘制每个列表的线
    plt.plot(history["loss"], label="eval")
    plt.plot(history_ja_on_train["loss"], label="train")

    # 添加标题和标签
    plt.title('OverFitting-loss')
    plt.xlabel('epoch')
    plt.ylabel('loss_value')

    # 显示图例
    plt.legend()
    # 显示图形
    plt.show()


if __name__ == "__main__":
    main()
