import pickle

import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go
import sklearn
from sklearn.metrics import accuracy_score, auc, confusion_matrix, f1_score, precision_recall_curve, roc_curve

import wandb


def klue_re_micro_f1(preds, labels):
    """KLUE-RE micro f1 (except no_relation)"""
    label_list = [
        "no_relation",
        "org:top_members/employees",
        "org:members",
        "org:product",
        "per:title",
        "org:alternate_names",
        "per:employee_of",
        "org:place_of_headquarters",
        "per:product",
        "org:number_of_employees/members",
        "per:children",
        "per:place_of_residence",
        "per:alternate_names",
        "per:other_family",
        "per:colleagues",
        "per:origin",
        "per:siblings",
        "per:spouse",
        "org:founded",
        "org:political/religious_affiliation",
        "org:member_of",
        "per:parents",
        "org:dissolved",
        "per:schools_attended",
        "per:date_of_death",
        "per:date_of_birth",
        "per:place_of_birth",
        "per:place_of_death",
        "org:founded_by",
        "per:religion",
    ]

    no_relation_label_idx = label_list.index("no_relation")
    label_indices = list(range(len(label_list)))
    label_indices.remove(no_relation_label_idx)
    return f1_score(labels, preds, average="micro", labels=label_indices) * 100.0


def klue_re_auprc(probs, labels):
    """KLUE-RE AUPRC (with no_relation)"""
    labels = np.eye(30)[labels]

    score = np.zeros((30,))
    for c in range(30):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = sklearn.metrics.precision_recall_curve(targets_c, preds_c)
        score[c] = sklearn.metrics.auc(recall, precision)
    return np.average(score) * 100.0


def graphs(labels, preds, probs):
    with open("./dataset/dict_label_to_num.pkl", "rb") as f:
        label_mapping = pickle.load(f)

    cm = confusion_matrix(labels, preds)

    fig1 = ff.create_annotated_heatmap(z=cm, x=list(range(30)), y=list(range(30)), colorscale="YlGnBu", showscale=True)
    fig1.update_layout(
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        xaxis=dict(tickmode="linear"),  # x축 레이블을 일렬로 나열
        yaxis=dict(tickmode="linear", autorange="reversed"),  # y축 레이블을 역순으로 나열
    )

    fig2 = go.Figure()
    fig3 = go.Figure()
    labels = np.eye(30)[labels]

    for c in range(30):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = precision_recall_curve(targets_c, preds_c)
        pr_auc = auc(recall, precision)
        fig2.add_trace(
            go.Scatter(
                x=recall,
                y=precision,
                mode="lines",
                name="{0} (AUC = {1:.2f})".format(list(label_mapping.keys())[c], pr_auc),
            )
        )
        fpr, tpr, _ = roc_curve(targets_c, preds_c)
        roc_auc = auc(fpr, tpr)
        fig3.add_trace(
            go.Scatter(
                x=fpr, y=tpr, mode="lines", name="{0} (AUC = {1:.2f})".format(list(label_mapping.keys())[c], roc_auc)
            )
        )

    fig2.update_layout(
        title="Multiclass Precision-Recall Curve",
        xaxis_title="Recall",
        yaxis_title="Precision",
    )

    fig2.update_yaxes(scaleanchor="x", scaleratio=1)
    fig2.update_xaxes(constrain="domain")

    fig3.update_layout(
        title="Multiclass ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate"
    )

    fig3.add_shape(type="line", line=dict(dash="dash"), x0=0, x1=1, y0=0, y1=1)

    fig3.update_yaxes(scaleanchor="x", scaleratio=1)
    fig3.update_xaxes(constrain="domain")

    wandb.log({"Confusion_Matrix": fig1, "PR-Curve": fig2, "ROC-Curve": fig3})


def compute_metrics(pred):
    """validation을 위한 metrics function"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    probs = pred.predictions
    # calculate accuracy using sklearn's function
    f1 = klue_re_micro_f1(preds, labels)
    auprc = klue_re_auprc(probs, labels)
    acc = accuracy_score(labels, preds)  # 리더보드 평가에는 포함되지 않습니다.

    graphs(labels, preds, probs)  # Plotting

    return {
        "micro f1 score": f1,
        "auprc": auprc,
        "accuracy": acc,
    }
