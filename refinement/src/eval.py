from sklearn.metrics import accuracy_score

def evaluation(eval_iter, model, criterion, device) -> float:
    avg_loss, acc_sum, acc_step = 0.0, 0.0, 0

    for i, batch in enumerate(eval_iter):
        model.eval()
        input_ids, token_type_ids, attention_mask, mlm_labels, labels = [w.to(device) for w in batch]

        logit, weight= model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        _, predictions = criterion(logit, mlm_labels, labels, weight)

        acc = accuracy_score(predictions, labels.cpu().detach().numpy())
        acc_step += 1
        acc_sum += acc
    return acc_sum/acc_step